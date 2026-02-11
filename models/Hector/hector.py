from models.Hector.Hectree import HecTree as Tree
from models.Hector.model import EncoderDecoder, make_model

from models.Hector.preprocess import build_init_embedding
from models.Hector.optimizer import rate, DenseSparseAdam, GradientClipper, arate

import torch
from torch.optim.lr_scheduler import ExponentialLR

from models.Hector.loss import MLabelSmoothing, SimpleLossCompute, CustomPrecisionLoss, quick_precision_at_1
import models.Hector.config as Cf



### TODO :
###     implement eval function



class Hector() : 

    def __init__(self, src_vocab, tgt_vocab, abstract_dict, taxonomies, 
                 device  = torch.device('cpu'),
                 #adaptive_patience,
                 Number_src_blocs = 6, Number_tgt_blocs = 6, dim_src_embedding = 300,
                  dim_tgt_embedding = 600, dim_feed_forward = 2048, number_of_heads = 12, dropout = 0.1, 
                  learning_rate = 1e-4, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8, weight_decay=0.01, gamma=.99998,
                  accum_iter = 20, loss_smoothing = 0.01,
                  max_padding_document = 128, max_number_of_labels = 20,
                  with_bias = False,
                  **kwargs
                ) -> None:


        self._model :EncoderDecoder = None

        
        print("Building Taxonomies")
        all_label_tokens = sorted(list(tgt_vocab.get_stoi().values()))


        self.tree = None

   


        assert len(taxonomies) ==1 , "Hector only supports single taxonomy"

        tax = taxonomies[0]
        root, children_dict = tax
        h = Tree(root_label=root,children_dict=children_dict, all_tokens=all_label_tokens)
        self.tree = h
        lvl = h.get_max_level()
        max_level = lvl


        self._max_level = max_level

        
        self.tree.set_max_level(max_level)

        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

        
            

        

        print("Initializing embeddings")

        if not Cf.QUICK_DEBUG :
            emb_src_init, emb_tgt_init = build_init_embedding( vocab_src=src_vocab, vocab_tgt=tgt_vocab,
                                                           d_src=dim_src_embedding,
                                                           d_tgt=dim_tgt_embedding, abstract_dict=abstract_dict)
        
        else : 
            emb_src_init = None
            emb_tgt_init = None

        print("Building Model")

        self._init_model(src_vocab, tgt_vocab, N_src = Number_src_blocs, N_tgt = Number_tgt_blocs, d_src=dim_src_embedding,
                          d_tgt = dim_tgt_embedding, d_ff = dim_feed_forward, h = number_of_heads, dropout = dropout,
                            emb_src_init = emb_src_init, emb_tgt_init = emb_tgt_init, lvl_mask= None,  with_bias=with_bias)
        

        
            


        self._pad_tgt =  tgt_vocab[Cf.padding_token]
        self._pad_src = src_vocab[Cf.padding_token]
        self._max_padding_src = max_padding_document
        self._max_padding_tgt = max_number_of_labels



        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.gamma = gamma

        self.get_optimizer()
        
        

        
        self._clippy = GradientClipper(start_value=1. * accum_iter)


        self.criterion = MLabelSmoothing(
            vocab_size=len(tgt_vocab),
            padding_idx=self._pad_tgt,
            smoothing=loss_smoothing
            )
        
        self._gpu_target = None if device in (None, "cpu") else device
        if isinstance(device, str):
            print(f"Using device: {device}")
            self.device = torch.device(device)
        elif isinstance(device, int):
            self.device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device or torch.device("cpu")

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)

        self._model.to(self.device)
        self.criterion = self.criterion.to(self.device)


        
        self.loss_function = SimpleLossCompute(self.criterion)

        self._metric = CustomPrecisionLoss(index_pady=self._pad_tgt, nprec=[1,2,3,4,5,10,20])

        self._training_steps = 0
        self._accum_iter = accum_iter


        print("Ready")

    def train(self):
        self._model.train()

    def eval(self):
        self._model.eval()

    def get_optimizer(self):
        self._optimizer = DenseSparseAdam(
            self._model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1,self.beta2),
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )

        self._lr_scheduler = ExponentialLR(
            optimizer=self._optimizer,
            gamma=self.gamma
        )
         


    def _init_model(self,src_vocab, tgt_vocab, N_src, N_tgt, d_src, d_tgt, d_ff, h, dropout, emb_src_init=None, emb_tgt_init=None,
                     lvl_mask=None, with_bias = False, **kwargs):

        self._model = make_model(src_vocab, tgt_vocab, N_src, N_tgt, d_src, d_tgt, d_ff, h, dropout, emb_src_init,
                                  emb_tgt_init, lvl_mask, with_bias=with_bias)

    
    def _forward(self,  src, tgt, src_mask, tgt_mask, child_mask):

        return self._model.forward(src, tgt, src_mask, tgt_mask, child_mask)
    



    def _prepare_all_paths(self, labels):
        """
        Return a list of triples (path_list, children_list, mask_list), all as Python lists.
        No padding and NO tensors here; _prepare_batch will handle padding & tensor conversion.

        If the document has no valid paths (e.g., only the root), return [].
        """

        labs = [int(l) for l in labels]
        labs = [l for l in labs if l!= self._pad_tgt]
        output = self.tree.labels_to_paths(labs)



        # Convert to plain lists (ints for ids, floats for masks); no padding here
        processed_output = []

        for (path, children, mask_level) in output :
            new_path = torch.LongTensor(path+([self._pad_tgt]*(self._max_level-len(path)))).unsqueeze(0)
            new_children = torch.LongTensor(children+([self._pad_tgt]*(self._max_padding_tgt-len(children)))).unsqueeze(0)
            new_mask_level = torch.FloatTensor(mask_level).unsqueeze(0)
            processed_output.append((new_path,new_children,new_mask_level))
        
        return processed_output









    def _prepare_batch(self, documents_tokens, labels ):  # 0 = <blank>
        """
        :input documents_tokens : 2D tensor of shape (batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :input labels_tokens : 2D tensor of shape (batch_size, max_padding_tgt) with labels ids for each documents


        :variable src: 2D tensor of shape (new_batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :variable path: 2D tensor of shape (new_batch_size, max_padding_tgt) with paths up to current level
        :variable kinder: 2D tensor of shape (new_batch_size, max_padding_tgt) with children of path
        :variable cmask: 2D tensor of shape (new_batch_size, tgt_vocab_size) where all labels except for children of
                    the current path are masked with 0s
        
        """

        GLOBAL_TASK_ID = -1
        src, tgt, kinder, masks = [], [], [], []

        for index_document in range(len(documents_tokens)):
            paths = self._prepare_all_paths(labels[index_document])
            if not paths:
                # Skip this doc 
                continue
            for (path,children,mask) in paths : 
                
                tgt.append(path)
                kinder.append(children)
                masks.append(mask)
                
            src.append( documents_tokens[index_document].repeat(len(paths),1) )

        # If the whole batch had no valid docs  return Nones (caller must handle)
        if not src:
            return None, None, None, None, None, None, None

        src = torch.cat(src,dim=0)
        tgt = torch.cat(tgt,dim=0)
        kinder = torch.cat(kinder,dim=0)
        masks = torch.cat(masks,dim=0)

        src_mask = (src != self._pad_src).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        tgt_mask = (tgt != self._pad_tgt).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        
        

        if kinder is not None:
            ntokens = (kinder != self._pad_tgt).data.sum()
        else:
            ntokens = None

        
        
        return (
            src,
            tgt,
            kinder,
            masks,
            src_mask,
            tgt_mask,
            ntokens,
        )

    

    def _prepare_test_batch(self, documents_tokens, paths, lvl_mask, **kwargs ):  # 0 = <blank>
        """
        :input documents_tokens : 2D tensor of shape (batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :input labels_tokens : 2D tensor of shape (batch_size, max_padding_tgt) with labels ids for each documents


        :variable src: 2D tensor of shape (new_batch_size, max_padding_src) with document tokens ids
                    each row starts with 1 (<s> token id) and ends with 2 (</s> token id)
        :variable path: 2D tensor of shape (new_batch_size, max_padding_tgt) with paths up to current level
        :variable kinder: 2D tensor of shape (new_batch_size, max_padding_tgt) with children of path
        :variable cmask: 2D tensor of shape (new_batch_size, tgt_vocab_size) where all labels except for children of
                    the current path are masked with 0s
        
        """

        

        src = []
        tgt = []
        masks = []


        for index_document in range(len(documents_tokens)):

            doc = documents_tokens[index_document].unsqueeze(0)
            path = paths[index_document]
            mask = lvl_mask[index_document]    

            new_path = torch.LongTensor(path+([self._pad_tgt]*(self._max_level-len(path)))).unsqueeze(0)
            new_mask_level = torch.FloatTensor(mask).unsqueeze(0)
            



            src.append( doc )
            tgt.append(new_path)
            masks.append(new_mask_level)

 
        
        src = torch.cat(src,dim=0)
        tgt = torch.cat(tgt,dim=0)

        masks = torch.cat(masks,dim=0)


        

        
        src_mask = (src != self._pad_src).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        tgt_mask = (tgt != self._pad_tgt).unsqueeze(-2)  # padding mask (non-masked positions: True, masked: False)
        
        
        device = getattr(self, "device", None)
        if device is None:
            # Fallback: keep current behavior
            if self._gpu_target is None:
                return src, tgt, masks, src_mask, tgt_mask
            device = torch.device("cuda", self._gpu_target) if torch.cuda.is_available() else torch.device("cpu")

        # If device is a string like "mps" / "cuda" / "cpu", turn it into a torch.device
        if isinstance(device, str):
            device = torch.device(device)

        return (
            src.to(device),
            tgt.to(device),
            masks.to(device),
            src_mask.to(device),
            tgt_mask.to(device),
        )




    def apply_loss(self):

        self._optimizer
        
        _ = self._clippy.clip_gradient(self._model)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._lr_scheduler.step()
        
     

    def train_on_batch(self, documents_tokens,  labels, train_batch_size = 1024):

        out = self._prepare_batch(documents_tokens,  labels)
        if out[0] is None:  # all documents skipped
            # Return a zero loss tensor on the correct device to keep loops simple
            return torch.tensor(0.0, device=self._model.device if hasattr(self._model, "device") else None)
        
        src, tgt, kinder, masks, src_mask, tgt_mask, ntokens = out
        

        device = getattr(self, "device", None)
        if device is None:
            # Fallback: keep current behavior
            if self._gpu_target is None:
                return src, tgt, kinder, masks, src_mask, tgt_mask, ntokens
            device = torch.device("cuda", self._gpu_target) if torch.cuda.is_available() else torch.device("cpu")

        # If device is a string like "mps" / "cuda" / "cpu", turn it into a torch.device
        if isinstance(device, str):
            device = torch.device(device)

        for i in range(0,len(src),train_batch_size):
        
            
            src_batch = src[i:i+train_batch_size].to(device)
            tgt_batch = tgt[i:i+train_batch_size].to(device)
            kinder_batch = kinder[i:i+train_batch_size].to(device)
            masks_batch = masks[i:i+train_batch_size].to(device)
            src_mask_batch = src_mask[i:i+train_batch_size].to(device)
            tgt_mask_batch = tgt_mask[i:i+train_batch_size].to(device)
            ntokens_batch = ntokens.to(device)
               
        

            out = self._forward(src_batch, tgt_batch, src_mask_batch, tgt_mask_batch, masks_batch)
            loss, loss_node = self.loss_function(x=out, y=kinder_batch, norm=ntokens_batch, level_mask=masks_batch)
            loss_node.backward()

            del src_batch, tgt_batch, kinder_batch, masks_batch, src_mask_batch, tgt_mask_batch, ntokens_batch, out
            
        self._training_steps += 1
        if self._training_steps % self._accum_iter == 0:
            self.apply_loss()
        return loss.detach()


    




    def eval_batch(self, documents_tokens, labels, eval_batch_size = 1024):

        out = self._prepare_batch(documents_tokens,  labels)
        if out[0] is None:  # all documents skipped
            # Return a zero loss tensor on the correct device to keep loops simple
            return torch.tensor(0.0, device=self._model.device if hasattr(self._model, "device") else None)
        
        src, tgt, kinder, masks, src_mask, tgt_mask, ntokens = out
        

        device = getattr(self, "device", None)
        if device is None:
            # Fallback: keep current behavior
            if self._gpu_target is None:
                return src, tgt, kinder, masks, src_mask, tgt_mask, ntokens
            device = torch.device("cuda", self._gpu_target) if torch.cuda.is_available() else torch.device("cpu")

        # If device is a string like "mps" / "cuda" / "cpu", turn it into a torch.device
        if isinstance(device, str):
            device = torch.device(device)


        total_loss = 0
        total_prec = 0


        for i in range(0,len(src),eval_batch_size):
        
            
            src_batch = src[i:i+eval_batch_size].to(device)
            tgt_batch = tgt[i:i+eval_batch_size].to(device)
            kinder_batch = kinder[i:i+eval_batch_size].to(device)
            masks_batch = masks[i:i+eval_batch_size].to(device)
            src_mask_batch = src_mask[i:i+eval_batch_size].to(device)
            tgt_mask_batch = tgt_mask[i:i+eval_batch_size].to(device)
            ntokens_batch = ntokens.to(device)
               
        

            out = self._forward(src_batch, tgt_batch, src_mask_batch, tgt_mask_batch, masks_batch)
            loss, _ = self.loss_function(x=out, y=kinder_batch, norm=ntokens_batch, level_mask=masks_batch)
            

            total_loss += loss.item()*len(src_batch)
            prec = quick_precision_at_1(out, kinder_batch, masks_batch)
            total_prec+= prec

            

            del src_batch, tgt_batch, kinder_batch, masks_batch, src_mask_batch, tgt_mask_batch, ntokens_batch, out

        loss = total_loss / len(src)
        precision = total_prec / len(src)
        return loss, precision
        

    def completion_batch(self, documents_tokens, labels, eval_batch_size = 1024):

        out = self._prepare_batch(documents_tokens,  labels)
        if out[0] is None:  # all documents skipped
            # Return a zero loss tensor on the correct device to keep loops simple
            return torch.tensor(0.0, device=self._model.device if hasattr(self._model, "device") else None)
        
        src, tgt, kinder, masks, src_mask, tgt_mask, ntokens = out
        

        device = getattr(self, "device", None)
        if device is None:
            # Fallback: keep current behavior
            if self._gpu_target is None:
                return src, tgt, kinder, masks, src_mask, tgt_mask, ntokens
            device = torch.device("cuda", self._gpu_target) if torch.cuda.is_available() else torch.device("cpu")

        # If device is a string like "mps" / "cuda" / "cpu", turn it into a torch.device
        if isinstance(device, str):
            device = torch.device(device)

        all_outputs = []
        all_kinders = []


        for i in range(0,len(src),eval_batch_size):
        
            
            src_batch = src[i:i+eval_batch_size].to(device)
            tgt_batch = tgt[i:i+eval_batch_size].to(device)
            kinder_batch = kinder[i:i+eval_batch_size].to(device)
            masks_batch = masks[i:i+eval_batch_size].to(device)
            src_mask_batch = src_mask[i:i+eval_batch_size].to(device)
            tgt_mask_batch = tgt_mask[i:i+eval_batch_size].to(device)
            ntokens_batch = ntokens.to(device)
               
        

            out = self._forward(src_batch, tgt_batch, src_mask_batch, tgt_mask_batch, masks_batch)

            all_outputs.append(out)
            all_kinders.append(kinder_batch)
            del src_batch, tgt_batch, kinder_batch, masks_batch, src_mask_batch, tgt_mask_batch, ntokens_batch
        out = torch.cat(all_outputs,dim=0)
        kinder = torch.cat(all_kinders,dim=0)


        return out, kinder

    
    
    def test_batch(self, documents_tokens, paths, lvl_mask):

        src,tgt,   masks, src_mask, tgt_mask = self._prepare_test_batch(documents_tokens, paths, lvl_mask)

        out = self._forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, child_mask=masks)

        return out
        



    def freeze(self):
        """
        do nothing for Hector
        """
        pass


    def get_generator_weights(self):
        weight = self._model.get_generator_weights()
        return weight
    





