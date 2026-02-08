from models.Tamlec.Tamlec_tree import TAMLEC_Tree as Tree
from models.Tamlec.model import EncoderDecoder, make_model

from models.Tamlec.preprocess import build_init_embedding
from models.Tamlec.optimizer import rate, DenseSparseAdam, GradientClipper, arate

import torch
from torch.optim.lr_scheduler import ExponentialLR

from models.Tamlec.loss import MLabelSmoothing, SimpleLossCompute, CustomPrecisionLoss, quick_precision_at_1
import models.Tamlec.config as Cf

import numpy as np



### TODO :
###     implement eval function



class Tamlec() : 

    def __init__(self, src_vocab, tgt_vocab, path_to_glove, abstract_dict, taxonomies, 
                 width_adaptive = False, # new param, with true activate the width_adaptive loss
                 decoder_adaptative = 0, # new param, number of decoding layers that are adapted to the task. 0 for no additional adaptation
                 tasks_size = None, # new param, list of the size of the training set of each task, to adapt the loss. 
                 device  = torch.device('cpu'),
                 #adaptive_patience,
                 Number_src_blocs = 6, Number_tgt_blocs = 6, dim_src_embedding = 300,
                  dim_tgt_embedding = 600, dim_feed_forward = 2048, number_of_heads = 12, dropout = 0.1, 
                  learning_rate = 1e-4, beta1 = 0.9, beta2 = 0.99, epsilon = 1e-8, weight_decay=0.01, gamma=.99998,
                  accum_iter = 20, loss_smoothing = 0.01,
                  max_padding_document = 128, max_number_of_labels = 20,
                  with_bias = False,
                ) -> None:


        self._model :EncoderDecoder= None

        print("Using NEW TAMLEC implementation")
        
        print("Building Taxonomies")
        n_task = len(taxonomies)
        all_label_tokens = sorted(list(tgt_vocab.get_stoi().values()))


        self.tamlec_forest = []

        max_level = 0

        widths_collection = []

        for tax in taxonomies :
            root, children_dict = tax
            h = Tree(root_label=root,children_dict=children_dict, all_tokens=all_label_tokens)
            self.tamlec_forest.append(h)
            lvl = h.get_max_level()
            max_level = max(max_level,lvl)
            widths_collection.append(h.get_width())

        if not width_adaptive : widths_collection = None

        self._max_level = max_level

        for h in self.tamlec_forest :
            h.set_max_level(max_level)

        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab

        
            

        

        print("Initializing embeddings")

        if not Cf.QUICK_DEBUG :
            emb_src_init, emb_tgt_init = build_init_embedding(path_to_glove, vocab_src=src_vocab, vocab_tgt=tgt_vocab,
                                                           d_src=dim_src_embedding,
                                                           d_tgt=dim_tgt_embedding, abstract_dict=abstract_dict)
        
        else : 
            emb_src_init = None
            emb_tgt_init = None

        print("Building Model")

        self._init_model(src_vocab, tgt_vocab, N_src = Number_src_blocs, N_tgt = Number_tgt_blocs, d_src=dim_src_embedding,
                          d_tgt = dim_tgt_embedding, d_ff = dim_feed_forward, h = number_of_heads, dropout = dropout,
                            emb_src_init = emb_src_init, emb_tgt_init = emb_tgt_init, lvl_mask= None, n_tasks = n_task,
                            N_custom_target=decoder_adaptative, with_bias=with_bias)
        


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
        
        

        if tasks_size is None :
            weights = None 
        else : 
            mweight = max(tasks_size)
            weights = [ mweight /w for w in tasks_size  ]
        self._clippy = GradientClipper(start_value=1. * accum_iter, weights = weights)


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
        self.loss_function = SimpleLossCompute(self.criterion,widths = widths_collection, weights = weights)

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
                     lvl_mask=None, n_tasks=1, N_custom_target = 0, with_bias = False):

        self._model = make_model(src_vocab, tgt_vocab, N_src, N_tgt, d_src, d_tgt, d_ff, h, dropout, emb_src_init,
                                  emb_tgt_init, lvl_mask, n_tasks, N_custom_target=N_custom_target,
                                  with_bias=with_bias)

    
    def _forward(self,  src, tgt, src_mask, tgt_mask, child_mask, task_id):

        return self._model.forward(src, tgt, src_mask, tgt_mask, child_mask, task_id)
    

    def apply_loss(self,task_id):

        DEBUG = False
        #print("Applying loss for task {}".format(task_id))

        if DEBUG :
            w1 = self._model.decoders_custom[task_id].layers[0].get_weights()["feed_forward"]["sublayer 0"]["w"]
            a1 = self._model.adapter.proj.weight
            c1 = self._model.encoder.layers[0].feed_forward.get_weights()["sublayer 0"]["w"]

        self._optimizer
        
        norm = self._clippy.clip_gradient(self._model, task_id)
        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._lr_scheduler.step()
        
        if DEBUG :
            w2 = self._model.decoders_custom[task_id].layers[0].get_weights()["feed_forward"]["sublayer 0"]["w"]
            a2 = self._model.adapter.proj.weight
            c2 = self._model.encoder.layers[0].feed_forward.get_weights()["sublayer 0"]["w"]

            print("Gradient clipped :", norm[1].item())
            print("Weights change after optimization step (first 5 values) :")
            print("Decoder weights difference:", np.linalg.norm(w2 - w1).item())
            print("Adapter weights difference", torch.norm(a2 - a1).item())
            print("Encoder weights difference", np.linalg.norm(c2 - c1).item())


    def _prepare_all_paths(self, labels, task_id):
        labs = [int(l) for l in labels]
        labs = [l for l in labs if l!= self._pad_tgt]
        output = self.tamlec_forest[task_id].labels_to_paths(labs)

        processed_output = []

        for (path, children, mask_level) in output :
            new_path = torch.LongTensor(path+([self._pad_tgt]*(self._max_level-len(path)))).unsqueeze(0)
            new_children = torch.LongTensor(children+([self._pad_tgt]*(self._max_padding_tgt-len(children)))).unsqueeze(0)
            new_mask_level = torch.FloatTensor(mask_level).unsqueeze(0) # TODO : why separate masks and children ?
            processed_output.append((new_path,new_children,new_mask_level))
        
        return processed_output







    def _prepare_batch(self, documents_tokens, labels, task_id ):  # 0 = <blank>
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
        kinder = []
        masks = []

        for index_document in range(len(documents_tokens)):
            paths = self._prepare_all_paths(labels[index_document], task_id)
            #paths  = paths_and_children[index_document].get(str(task_id), [])
            for (path,children,mask) in paths : 
                tgt.append(path)
                kinder.append(children)
                masks.append(mask)
                
            src.append( documents_tokens[index_document].repeat(len(paths),1) )

 
        
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

        
        return src, tgt,  kinder, masks, src_mask, tgt_mask, ntokens


    

    def _prepare_test_batch(self, documents_tokens, paths, lvl_mask, id_task ):  # 0 = <blank>
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

        return src.to(device),tgt.to(device),  masks.to(device), src_mask.to(device), tgt_mask.to(device)





    

    def train_on_batch(self, documents_tokens, labels,task_id, train_batch_size = 1024 ):

        #print("Training on batch for task {}".format(task_id))

        

        src,tgt,  kinder, masks, src_mask, tgt_mask, ntokens = self._prepare_batch(documents_tokens=documents_tokens,labels=labels, task_id=task_id)

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
               
        

        
            out = self._forward(src=src_batch, tgt=tgt_batch, src_mask=src_mask_batch, tgt_mask=tgt_mask_batch, child_mask=masks_batch, task_id=task_id)
            loss, loss_node = self.loss_function(x=out, y=kinder_batch, norm=ntokens_batch, level_mask=masks_batch, task_id=task_id)
            loss_node.backward()

            del src_batch, tgt_batch, kinder_batch, masks_batch, src_mask_batch, tgt_mask_batch, ntokens_batch, out

        
       
        self._training_steps+=1 
        if self._training_steps % self._accum_iter == 0 :
            self.apply_loss(task_id=task_id)

        
        return loss.detach()


    




    def eval_batch(self, documents_tokens, labels,task_id,eval_batch_size=1024):

        src,tgt,  kinder, masks, src_mask, tgt_mask, ntokens = self._prepare_batch(documents_tokens=documents_tokens,labels=labels, task_id=task_id)

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
               
        

        
            out = self._forward(src=src_batch, tgt=tgt_batch, src_mask=src_mask_batch, tgt_mask=tgt_mask_batch, child_mask=masks_batch, task_id=task_id)
            loss, _ = self.loss_function(x=out, y=kinder_batch, norm=ntokens_batch, level_mask=masks_batch,task_id=task_id)
            total_loss += loss.item()*len(src_batch)
            prec = quick_precision_at_1(out, kinder_batch, masks_batch)
            total_prec+= prec
            del src_batch, tgt_batch, kinder_batch, masks_batch, src_mask_batch, tgt_mask_batch, ntokens_batch, out

        loss = total_loss / len(src)
        precision = total_prec / len(src)
        #print("Eval precision @1:", precision)

        return loss,precision
    

    def completion_batch(self, documents_tokens, labels,task_id):

        src,tgt,  kinder, masks, src_mask, tgt_mask, ntokens = self._prepare_batch(documents_tokens=documents_tokens,labels=labels, task_id=task_id)

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

        for i in range(0,len(src),1024):
        
            
            src_batch = src[i:i+1024].to(device)
            tgt_batch = tgt[i:i+1024].to(device)
            kinder_batch = kinder[i:i+1024].to(device)
            masks_batch = masks[i:i+1024].to(device)
            src_mask_batch = src_mask[i:i+1024].to(device)
            tgt_mask_batch = tgt_mask[i:i+1024].to(device)
            ntokens_batch = ntokens.to(device)
               
        

        
            out = self._forward(src=src_batch, tgt=tgt_batch, src_mask=src_mask_batch, tgt_mask=tgt_mask_batch, child_mask=masks_batch, task_id=task_id)
            all_outputs.append(out)
            all_kinders.append(kinder_batch)
            del src_batch, tgt_batch, kinder_batch, masks_batch, src_mask_batch, tgt_mask_batch, ntokens_batch
        out = torch.cat(all_outputs,dim=0)
        kinder = torch.cat(all_kinders,dim=0)

        return out, kinder

 
        
        
        
    
    def test_batch(self, documents_tokens, paths, lvl_mask, id_task):
        
        src,tgt,   masks, src_mask, tgt_mask = self._prepare_test_batch(documents_tokens, paths, lvl_mask, id_task)

        out = self._forward(src=src, tgt=tgt, src_mask=src_mask, tgt_mask=tgt_mask, child_mask=masks, task_id=id_task)

        return out
        



    def freeze(self):
        """
        freeze all layers except the task specifics
        """
        self._model.freeze()


    def get_generator_weights(self,task_id : int):
        weight = self._model.get_generator_weights(task_id=task_id)
        return weight
    
    def get_custom_decoder_weights(self,task_id : int):
        w = self._model.get_custom_decoder_weights(task_id=task_id)
        return w





