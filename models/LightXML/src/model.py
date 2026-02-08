import tqdm
import time
import cProfile
import numpy as np
# from apex import amp

import torch
from torch import nn


class LightXML(nn.Module):
    def __init__(self, n_labels, bert, group_y=None, feature_layers=2, dropout=0.5, update_count=1,
                 candidates_topk=10, 
                 use_swa=False, swa_warmup_epoch=10, swa_update_step=200, hidden_dim=300):
        super(LightXML, self).__init__()
        """
        Initialize the LightXML model.

        Args:
            n_labels: total number of fine-grained labels.
            bert: transformer encoder used as text backbone.
            group_y: optional mapping of group → label indices (enables two-stage mode).
            feature_layers: number of last BERT layers to concatenate.
            dropout: dropout probability for pooled features.
            update_count: gradient accumulation steps.
            candidates_topk: number of top groups retained per sample.
            use_swa: enable Stochastic Weight Averaging for smoother convergence.
            swa_warmup_epoch: epochs to skip before SWA starts (avoid noisy early weights).
            swa_update_step: how often to update SWA weights (in training steps).
            hidden_dim: dimensionality of label embedding space (bottleneck).
        """
        
        # ---- SWA (Stochastic Weight Averaging) configuration ----
        # Averaging weights during late training improves generalization.
        # Warmup avoids averaging unstable early weights.
        self.use_swa = use_swa 
        self.swa_warmup_epoch = swa_warmup_epoch 
        self.swa_update_step = swa_update_step
        self.swa_state = {} # stores running averages of parameters

        # Number of gradient accumulation steps before optimizer step
        self.update_count = update_count
        
        self.candidates_topk = candidates_topk

        print('swa', self.use_swa, self.swa_warmup_epoch, self.swa_update_step, self.swa_state)
        print('update_count', self.update_count)

        # ---- Backbone and feature projection ----
        self.bert = bert
        self.feature_layers, self.drop_out = feature_layers, nn.Dropout(dropout)

        # ---- Two-stage setup: group → label ----
        # If group_y is provided, model learns a group-level head (coarse)
        # and a label-embedding head (fine). Otherwise, a single full-label head is used.
        self.group_y = group_y
        if self.group_y is not None:
            self.group_y_labels = group_y.shape[0]
            print('hidden dim:', hidden_dim)
            print('label group numbers:', self.group_y_labels)

            # ---- Coarse group prediction head ----
            self.l0 = nn.Linear(self.feature_layers * self.bert.hidden_size, self.group_y_labels)

            # ---- Bottleneck projection ----
            self.l1 = nn.Linear(self.feature_layers * self.bert.hidden_size, hidden_dim)

            # ---- Fine label embeddings ----
            self.embed = nn.Embedding(n_labels, hidden_dim)
            nn.init.xavier_uniform_(self.embed.weight)

            # ==============================================================
            # 🔹 Replace Python list-of-arrays `group_y` with flat GPU tensors
            # ==============================================================

            # Flatten all labels into a single tensor
            group_lengths = [len(arr) for arr in group_y]
            group_labels_flat = torch.tensor(
                np.concatenate(group_y, axis=0), dtype=torch.long
            )

            # Compute prefix-sum pointer array (start indices per group)
            group_ptr = torch.tensor(
                [0] + list(np.cumsum(group_lengths)), dtype=torch.long
            )

            # Register them as buffers so they move automatically with .cuda(), .to(), etc.
            self.register_buffer("group_labels_flat", group_labels_flat)
            self.register_buffer("group_ptr", group_ptr)

            # You can now safely drop the old Python structure to free memory
            del self.group_y
        else:
            self.l0 = nn.Linear(self.feature_layers*self.bert.hidden_size, n_labels)

    def get_candidates(self, group_logits, group_gd=None):
        """
        Select candidate fine-grained labels based on coarse (group-level) predictions.

        Args:
            group_logits: Tensor of shape [B, num_groups]; raw outputs from group head.
            group_gd: Optional ground-truth group labels [B, num_groups] (used only in training).
                    Acts as guidance to ensure non-zero gradients on true groups.

        Returns:
            indices:  [B, top_k] array of selected group indices per sample.
            candidates:  [B, padded_C] array of candidate fine label ids for each sample.
            candidates_scores:  [B, padded_C] array of corresponding group-level scores.
        """
        
        # probs on GPU (no grad needed for topk)
        probs = torch.sigmoid(group_logits)
        if group_gd is not None:
            probs = probs + group_gd  # assumes same shape, float

        # top-k on GPU
        scores, idx = torch.topk(probs, k=self.candidates_topk, dim=1)        # [B, K]

        # Expand groups -> labels via flat map
        # self.group_ptr: [G+1], self.group_labels_flat: [sum_labels], both CUDA LongTensors
        # For each sample, gather the (start,end) per top-k group and slice/concat
        starts = self.group_ptr[idx]                      # [B, K]
        ends   = self.group_ptr[idx + 1]                  # [B, K]
        lens   = (ends - starts)                          # [B, K]
        maxC   = lens.sum(dim=1).max().item()            # worst-case padded length

        # Build a padded candidates tensor [B, maxC]; same for group prior scores
        B, K = idx.shape
        candidates = torch.empty(B, maxC, dtype=torch.long, device=idx.device)
        priors     = torch.empty(B, maxC, device=idx.device)

        # Vectorized ragged fill: one pass per (B) instead of per (B,K,label)
        # (If you want fully vectorized, you can precompute cumulative offsets per row.)
        for b in range(B):
            ofs = 0
            for k in range(K):
                s, e = starts[b, k].item(), ends[b, k].item()
                seg = self.group_labels_flat[s:e]                       # [Lk]
                Lk  = seg.numel()
                candidates[b, ofs:ofs+Lk] = seg
                priors[b, ofs:ofs+Lk]     = scores[b, k]                # broadcast
                ofs += Lk
            # pad tail with last value (“edge”) if needed
            if ofs < maxC:
                candidates[b, ofs:] = candidates[b, ofs-1]
                priors[b, ofs:]     = priors[b, ofs-1]

        return idx, candidates, priors

    def forward(self, input_ids, labels=None, group_labels=None, candidates=None):
        """
        Two modes:
        - Single-stage (no groups): direct multi-label prediction over all labels.
        - Two-stage (with groups): (1) predict groups, (2) prune to label candidates,
            (3) score candidates via label embeddings. Train with joint (fine + group) loss.

        Args:
            input_ids: token ids for the text encoder.
            labels: [B, n_labels] multi-hot (fine labels). If None → inference mode.
            group_labels: [B, n_groups] multi-hot (supervises coarse retriever).
            candidates: dense list of label ids per sample (used only to extract true ids at train time).
        """
        is_training = labels is not None
        
        # 1) Encode text and build a richer representation by concatenating the last
        #    `feature_layers` CLS embeddings. This captures multi-level features.
        outs = self.bert(input_ids, output_layers=True)
        out = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        out = self.drop_out(out)
        
        # 2) Coarse head: predict group logits (or full labels if single-stage).
        
        # --- Single-stage path (no group pruning; simpler baseline) ---
        group_logits = self.l0(out)
        if self.group_y is None:
            logits = group_logits # directly over all labels
            if is_training:
                loss_fn = torch.nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels) # optimize only fine-label BCE
                return logits, loss
            else:
                return logits
        # --- Two-stage path (group → candidates → fine scoring) ---
        if is_training:
            # Extract the set of true label ids per sample by masking `candidates` with `labels`.
            # We keep counts per sample to later remap supervision onto pruned candidate lists.
            l = labels.to(dtype=torch.bool)
            target_candidates = torch.masked_select(candidates, l).detach().cpu()
            target_candidates_num = l.sum(dim=1).detach().cpu()
        
        # 3) Build candidate label sets using coarse predictions.
        #    During training, add group ground-truth as guidance to avoid missing positives.
        with torch.no_grad():
            groups, candidates, group_candidates_scores = self.get_candidates(group_logits,
                                                                          group_gd=group_labels if is_training else None)
        if is_training:
            # 4) Remap global one-hot supervision to the candidate space per sample.
            #    If a true label was pruned out, force-insert it by overwriting a slot
            #    (ensures positives are present → gradients for fine head).
            bs = 0
            new_labels = []
            for i, n in enumerate(target_candidates_num.numpy()):
                be = bs + n
                c = set(target_candidates[bs: be].numpy())
                c2 = candidates[i]
                new_labels.append(torch.tensor([1.0 if i in c else 0.0 for i in c2 ]))
                
                # Force recall: if some true labels were not in candidates, overwrite zeros.
                if len(c) != new_labels[-1].sum():
                    s_c2 = set(c2)
                    for cc in list(c):
                        if cc in s_c2:
                            continue
                        for j in range(new_labels[-1].shape[0]):
                            if new_labels[-1][j].item() != 1:
                                c2[j] = cc
                                new_labels[-1][j] = 1.0
                                break
                bs = be
            labels = torch.stack(new_labels).cuda()
            
        # Tensorize candidates and their group priors on GPU for batched scoring.
        candidates, group_candidates_scores =  torch.LongTensor(candidates).cuda(), torch.Tensor(group_candidates_scores).cuda()

        # 5) Fine scorer: project text to label-embedding space and score candidates
        #    by dot product with their learned embeddings (efficient bilinear scoring).
        emb = self.l1(out)
        embed_weights = self.embed(candidates) # N, sampled_size, H
        emb = emb.unsqueeze(-1)
        logits = torch.bmm(embed_weights, emb).squeeze(-1)

        if is_training:
            # 6) Joint objective:
            #    - Fine-level BCE over candidate logits (precision within groups)
            #    - Group-level BCE over coarse logits (recall of relevant groups)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels) + loss_fn(group_logits, group_labels)
            return logits, loss
        else:
            # 7) Inference: convert fine logits to probs and weight by group priors.
            #    This blends coarse relevance with fine ranking quality.
            candidates_scores = torch.sigmoid(logits)
            candidates_scores = candidates_scores * group_candidates_scores
            return group_logits, candidates, candidates_scores

    def save_model(self, path):
        # Temporarily swap in SWA-averaged weights for saving, then restore current ones
        self.swa_swap_params()
        torch.save(self.state_dict(), path)
        self.swa_swap_params()

    def swa_init(self):
        # Initialize SWA state with the first model snapshot
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        # Update running average of model weights (SWA)
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(beta, p.data.cpu())

    def swa_swap_params(self):
        # Swap current model weights with SWA-averaged weights (used for eval/save)
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].cuda()

    def get_accuracy(self, candidates, logits, labels):
        # Compute Precision@1/3/5 given model logits and true labels
        if candidates is not None:
            candidates = candidates.detach().cpu()
        scores, indices = torch.topk(logits.detach().cpu(), k=10)

        acc1, acc3, acc5, total = 0, 0, 0, 0
        for i, l in enumerate(labels):
            l = set(np.nonzero(l)[0]) # true label indices
            # Map top-k candidate indices to label IDs (if candidate set given)
            if candidates is not None:
                labels = candidates[i][indices[i]].numpy()
            else:
                labels = indices[i, :5].numpy()

            acc1 += len(set([labels[0]]) & l)
            acc3 += len(set(labels[:3]) & l)
            acc5 += len(set(labels[:5]) & l)
            total += 1

        return total, acc1, acc3, acc5

    def one_epoch(self, epoch, dataloader, optimizer,
                  mode='train', eval_loader=None, eval_step=20000, log=None):
        """
        Run one full epoch of training, evaluation, or testing.

        Args:
            epoch: current epoch index.
            dataloader: main DataLoader for this pass.
            optimizer: optimizer (used only in 'train' mode).
            mode: 'train', 'eval', or 'test'.
            eval_loader: optional validation loader for periodic evaluation.
            eval_step: step interval for intermediate evaluation during training.
            log: optional logger for periodic metric printing.

        Returns:
            - In eval: (group p@1/3/5, fine p@1/3/5)
            - In test: predicted scores and labels
            - In train: accumulated loss
        """
        bar = tqdm.tqdm(total=len(dataloader))
        # Metric accumulators (group-level and fine-level)
        p1, p3, p5 = 0, 0, 0
        g_p1, g_p3, g_p5 = 0, 0, 0
        total, acc1, acc3, acc5 = 0, 0, 0, 0
        g_acc1, g_acc3, g_acc5 = 0, 0, 0
        train_loss = 0

        # Set training or eval mode
        if mode == 'train':
            self.train()
        else:
            self.eval()
            
        # Initialize SWA after warmup epoch
        if self.use_swa and epoch == self.swa_warmup_epoch and mode == 'train':
            self.swa_init()

        # Use SWA weights during evaluation
        if self.use_swa and mode == 'eval':
            self.swa_swap_params()

        pred_scores, pred_labels = [], []
        bar.set_description(f'{mode}-{epoch}')

        with torch.set_grad_enabled(mode == 'train'):
            for step, data in enumerate(dataloader):
                batch = tuple(t for t in data)
                have_group = len(batch) > 4 # check if group labels exist
                
                # check if group labels exist
                inputs = {'input_ids':      batch[0].cuda(),
                          'attention_mask': batch[1].cuda(),
                          'token_type_ids': batch[2].cuda()}
                
                # check if group labels exist
                if mode == 'train':
                    inputs['labels'] = batch[3].cuda()
                    if self.group_y is not None:
                        inputs['group_labels'] = batch[4].cuda()
                        inputs['candidates'] = batch[5].cuda()
                        
                # Forward pass
                outputs = self(**inputs)
                bar.update(1)
                
                # ------------------------- TRAINING -------------------------
                if mode == 'train':
                    loss = outputs[1]
                    loss /= self.update_count
                    train_loss += loss.item()
                    
                    # Backprop with mixed precision (Apex AMP)
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                    # Gradient accumulation: update every `update_count` steps
                    if step % self.update_count == 0:
                        optimizer.step()
                        self.zero_grad()
                    
                    # Periodic evaluation to monitor progress
                    if step % eval_step == 0 and eval_loader is not None and step != 0:
                        results = self.one_epoch(epoch, eval_loader, optimizer, mode='eval')
                        p1, p3, p5 = results[3:6]
                        g_p1, g_p3, g_p5 = results[:3]
                        
                        # Log both fine and group precision metrics
                        if self.group_y is not None:
                            log.log(f'{epoch:>2} {step:>6}: {p1:.4f}, {p3:.4f}, {p5:.4f}'
                                    f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}')
                        else:
                            log.log(f'{epoch:>2} {step:>6}: {p1:.4f}, {p3:.4f}, {p5:.4f}')
                        # NOTE: we don't reset model to train mode and keep model in eval mode
                        # which means all dropout will be remove after `eval_step` in every epoch
                        # this tricks makes LightXML converge fast
                        # self.train()

                    # Periodically update SWA parameters
                    if self.use_swa and step % self.swa_update_step == 0:
                        self.swa_step()

                    bar.set_postfix(loss=loss.item())
                    
                # ------------------------- EVALUATION (no group mode) -------------------------
                elif self.group_y is None:
                    logits = outputs
                    if mode == 'eval':
                        labels = batch[3]
                        _total, _acc1, _acc3, _acc5 =  self.get_accuracy(None, logits, labels.cpu().numpy())
                        total += _total; acc1 += _acc1; acc3 += _acc3; acc5 += _acc5
                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5
                        bar.set_postfix(p1=p1, p3=p3, p5=p5)
                    elif mode == 'test':
                        pred_scores.append(logits.detach().cpu())
                        
                # ------------------------- EVALUATION (with groups) -------------------------
                else:
                    group_logits, candidates, logits = outputs

                    if mode == 'eval':
                        labels = batch[3]
                        group_labels = batch[4]

                        # Fine-level precision
                        _total, _acc1, _acc3, _acc5 = self.get_accuracy(candidates, logits, labels.cpu().numpy())
                        total += _total; acc1 += _acc1; acc3 += _acc3; acc5 += _acc5
                        p1 = acc1 / total
                        p3 = acc3 / total / 3
                        p5 = acc5 / total / 5

                        # Group-level precision
                        _, _g_acc1, _g_acc3, _g_acc5 = self.get_accuracy(None, group_logits, group_labels.cpu().numpy())
                        g_acc1 += _g_acc1; g_acc3 += _g_acc3; g_acc5 += _g_acc5
                        g_p1 = g_acc1 / total
                        g_p3 = g_acc3 / total / 3
                        g_p5 = g_acc5 / total / 5
                        bar.set_postfix(p1=p1, p3=p3, p5=p5, g_p1=g_p1, g_p3=g_p3, g_p5=g_p5)
                    elif mode == 'test':
                        # Collect top-100 predictions for evaluation metrics (e.g., P@k)
                        _scores, _indices = torch.topk(logits.detach().cpu(), k=100)
                        _labels = torch.stack([candidates[i][_indices[i]] for i in range(_indices.shape[0])], dim=0)
                        pred_scores.append(_scores.cpu())
                        pred_labels.append(_labels.cpu())

        # Swap back to original weights after SWA evaluation
        if self.use_swa and mode == 'eval':
            self.swa_swap_params()
        bar.close()

        # Return results depending on mode
        if mode == 'eval':  
            return g_p1, g_p3, g_p5, p1, p3, p5
        elif mode == 'test':
            return torch.cat(pred_scores, dim=0).numpy(), torch.cat(pred_labels, dim=0).numpy() if len(pred_labels) != 0 else None
        elif mode == 'train':
            return train_loss
