# Adapted from https://github.com/xmc-aalto/cascadexml/blob/main/src/CascadeXML.py
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def feat_maker(recipe, bert_outs):
    feats = [None] * len(recipe)
    for i, idx_list in enumerate(recipe):
        if isinstance(idx_list, int):
            feats[i] = bert_outs[idx_list][:, 0]
        
        elif isinstance(idx_list, tuple):
            feats[i] = torch.cat([bert_outs[idx][:, 0] for idx in idx_list], dim=1)

        else:
            raise ValueError("Invalid feat recipe")
    return feats
    

class CascadeXML(nn.Module):
    def __init__(
        self,
        taxonomy,
        emb_dim,
        device,
        backbone,
        clusters
        ):
        super(CascadeXML, self).__init__()
        self.backbone = backbone
        self.taxonomy = taxonomy
        self.emb_dim = emb_dim
        self.device = device
        # Hyperparameters for this model
        self.candidates_topk = [50, 50]
        self.candidates_topk = self.candidates_topk.split(',') if isinstance(self.candidates_topk, str) else self.candidates_topk
        assert isinstance(self.candidates_topk, list), "topK should be a list with at least 2 integers"
        self.return_shortlist = False
        self.rw_loss = [1, 1, 1, 1]
        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
        embed_drops = [0.2, 0.25, 0.4, 0.5]
        self.layers = [(0, 1), 2, 3]

        max_cluster = [max([len(c) for c in clusters[i]]) for i in range(len(clusters))]
        # Number of clusters per level, last level has the root of the taxonomy excluded
        self.num_ele = [len(g) for g in clusters] + [self.taxonomy.n_nodes-1]

        # Meta classifiers
        self.Cn = nn.ModuleList([nn.Embedding(self.num_ele[0], self.emb_dim)])
        self.Cn_bias = nn.ModuleList([nn.Embedding(self.num_ele[0], 1)])

        # For an even number of clusters on second level
        if self.num_ele[-2] % 2 == 1:
            # Add a padding cluster on level 2
            pad_idx_cluster_level2 = len(clusters[1])
            for i in range(self.num_ele[0]):
                clusters[-2][i] = np.pad(clusters[-2][i], (0, max_cluster[-2]-len(clusters[-2][i])),
                                        constant_values=pad_idx_cluster_level2).astype(np.int32)
            self.Cn.append(nn.Embedding(self.num_ele[1]+1, self.emb_dim, padding_idx=pad_idx_cluster_level2))
            self.Cn_bias.append(nn.Embedding(self.num_ele[1]+1, 1, padding_idx=pad_idx_cluster_level2))

            # Then necessarily add a padding label on last level, so that it goes in the pad cluster on level 2
            pad_idx_cluster_level3 = self.num_ele[-1]
            clusters[1].append(np.array([pad_idx_cluster_level3, pad_idx_cluster_level3], dtype=np.int32))
            for i in range(self.num_ele[-2]):
                clusters[-1][i] = np.pad(clusters[-1][i], (0, max_cluster[-1]-len(clusters[-1][i])), 
                                        constant_values=pad_idx_cluster_level3).astype(np.int32)
            self.Cn.append(nn.Embedding(self.num_ele[2]+1, self.emb_dim, padding_idx=pad_idx_cluster_level3))
            self.Cn_bias.append(nn.Embedding(self.num_ele[2]+1, 1, padding_idx=pad_idx_cluster_level3))

        else:
            # For last level of the clustering
            for i in range(self.num_ele[-2]):
                clusters[-1][i] = np.pad(clusters[-1][i], (0, max_cluster[-1]-len(clusters[-1][i])), 
                                            constant_values=self.num_ele[-1]).astype(np.int32)
            self.Cn.append(nn.Embedding(self.num_ele[1], self.emb_dim))
            self.Cn_bias.append(nn.Embedding(self.num_ele[1], 1))
            # Add a padding level if needed
            if self.num_ele[2] % 2 == 1:
                self.Cn.append(nn.Embedding(self.num_ele[2]+1, self.emb_dim, padding_idx=-1))
                self.Cn_bias.append(nn.Embedding(self.num_ele[2]+1, 1, padding_idx=-1))
            else:
                self.Cn.append(nn.Embedding(self.num_ele[2], self.emb_dim))
                self.Cn_bias.append(nn.Embedding(self.num_ele[2], 1))


        # clusters = [np.stack(c) for c in clusters]
        _clusters = [torch.LongTensor(c).to(device) for c in clusters]  # device set by buffers
        for i, c in enumerate(_clusters):
            self.register_buffer(f"clusters_{i}", c)
        self.clusters = [getattr(self, f"clusters_{i}") for i in range(len(_clusters))]
        self.num_ele = [len(g) for g in self.clusters] + [self.taxonomy.n_nodes-1]
        
        print(f'label group numbers: {self.num_ele}; top_k: {self.candidates_topk}')
        print(f'Layers used: {self.layers}; Dropouts: {embed_drops}')
        print(self.Cn)
        assert len(self.layers) == len(self.num_ele)
                
        concat_size = len(self.layers[0])*self.emb_dim
        self.Cn_hidden = nn.Sequential(
                              nn.Dropout(0.2),
                              nn.Linear(concat_size, self.emb_dim)
                        )
        
        self.embed_drops = nn.ModuleList([nn.Dropout(p) for p in embed_drops])
        self.init_classifier_weights()


    def init_classifier_weights(self):
        nn.init.xavier_uniform_(self.Cn_hidden[1].weight)

        for C in self.Cn:
            nn.init.xavier_uniform_(C.weight)
        self.Cn[-1].weight[-1].data.fill_(0)

        for bias in self.Cn_bias:
            bias.weight.data.fill_(0)

    def reinit_weights(self):
        for C in self.Cn[:-1]:
            nn.init.xavier_uniform_(C.weight)

    def get_candidates(self, group_scores, prev_cands, level, group_gd=None):
        TF_scores = group_scores.clone()
        if group_gd is not None:
            TF_scores += group_gd
        scores, indices = torch.topk(TF_scores, k=min(self.candidates_topk[level - 1], TF_scores.shape[1]))
        if self.is_training:
            scores = group_scores[torch.arange(group_scores.shape[0]).view(-1,1).to(self.device), indices]
        indices = prev_cands[torch.arange(indices.shape[0]).view(-1,1).to(self.device), indices]
        candidates = self.clusters[level - 1][indices] 
        candidates_scores = torch.ones_like(candidates) * scores[...,None] 

        return indices, candidates.flatten(1), candidates_scores.flatten(1)


    def forward(self, input_ids, all_labels=None, return_out=False):
        self.is_training = all_labels is not None
        
        bert_outs = self.backbone(input_ids, output_embeddings=True, output_layers=True)

        if return_out:
            # out = torch.stack(bert_outs).detach().clone().cpu() #Attention Maps
            # out = [bert_outs[idx][:, 0].detach().clone().cpu() for idx in range(13)] #CLS Token
            out = bert_outs[-1][:, 0].detach().clone().cpu() #last embedding
            del bert_outs
            return out

        outs = feat_maker(self.layers, bert_outs)

        prev_logits, prev_labels, all_losses, all_probs, all_probs_weighted, all_candidates = None, None, [], [], [], []
        # Loop over the level in the hierarchical clustering
        for i, (embed, feat) in enumerate(zip(self.Cn, outs)):
            if i == 0:
                feat = self.Cn_hidden(feat)
            feat = self.embed_drops[i](feat).unsqueeze(-1)

            if self.is_training:
                labels = all_labels[i]

            if i == 0:
                candidates = torch.arange(embed.num_embeddings)[None].expand(feat.shape[0], -1)
            else:
                shortlisted_clusters, candidates, group_candidates_scores = self.get_candidates(prev_logits, prev_cands, level=i, group_gd=prev_labels)

            candidates = candidates.to(input_ids.device)

            new_labels, new_cands, new_group_cands = [], [], []
            for j in range(input_ids.shape[0]):
                if i == len(self.Cn) - 1:
                    new_cands.append(candidates[j][torch.where(candidates[j] != self.num_ele[i])[0]])
                    new_group_cands.append(group_candidates_scores[j][torch.where(candidates[j] != self.num_ele[i])[0]])
                else:
                    new_cands.append(candidates[j])

                if self.is_training:
                    ext = labels[j].to(candidates.device)
                    lab_bin = (new_cands[-1][..., None] == ext).any(-1).float()
                    new_labels.append(lab_bin)

            if self.is_training:
                labels = pad_sequence(new_labels, True, 0).to(input_ids.device)
            if i == len(self.Cn) - 1:
                candidates = pad_sequence(new_cands, True, self.num_ele[i])
                group_candidates_scores = pad_sequence(new_group_cands, True, 0.)

                if self.return_shortlist:
                    return candidates

            candidates = candidates.to(embed.weight.device)
            embed_weights = embed(candidates)
            logits = torch.bmm(embed_weights, feat.to(embed.weight.device)).squeeze(-1)
            logits = (logits + self.Cn_bias[i].to(embed.weight.device)(candidates).squeeze(-1)).to(input_ids.device) 

            if i == len(self.Cn) - 1:
                candidates_scores = torch.where(logits == 0., -np.inf, logits.double()).float().sigmoid() #Handling padding
            else:
                candidates_scores = torch.sigmoid(logits) 

            weighted_scores = candidates_scores * group_candidates_scores if i != 0 else candidates_scores

            all_probs.append(candidates_scores)
            all_probs_weighted.append(weighted_scores)
            all_candidates.append(candidates)

            prev_logits = candidates_scores.detach()
            prev_cands = candidates

            if self.is_training:
                all_losses.append(self.loss_fn(logits, labels))
                prev_labels = labels

        if self.is_training:
            sum_loss = 0.
            for i, l in enumerate(all_losses):
                sum_loss += l * self.rw_loss[i]
            return all_probs, all_candidates, sum_loss
        else:
            # List of probs, candidates per level in the clustering
            return all_probs, all_candidates, all_probs_weighted
