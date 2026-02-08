from tqdm import tqdm
import torch
import numpy as np
import time
import warnings

from algorithms.base_algorithm import AbstractAlgorithm
from misc.utils.time import format_time_diff
from models.FastXML.fastxml.trainer import Trainer
from models.FastXML.fastxml.fastxml import Inferencer
from models.FastXML.fastxml.weights import propensity
from scipy.sparse import csr_matrix
from pathlib import Path

class FastxmlAlg(AbstractAlgorithm):
    """FastXML algorithm wrapper backed by the FastXML trainer/inferencer."""
    def __init__(
            self,  
            metrics_handler,
            taxonomy,
            embeddings_loader, # Embeddings loader to initialize the BaseBertModel
            model_path: Path = Path(), # Path to a model to load the model from
            **kwargs,
            ): 
        """Initialize FastXML trainer and embedding utilities."""
        self.taxonomy = taxonomy
        self.model_path = model_path
        # Padding index set at 0, freeze=True to disable training on embeddings, it does not matter in here
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings_loader, padding_idx=0, freeze=True)
        print(f"> Loading model...")
        self.inferencer = None
        self.model = Trainer(
            # Number of trees in the ensemble
            n_trees=50,
            max_leaf_size=200,
            max_labels_per_leaf=200,
            n_jobs=2,
            # Number of iterations (max_iter in sklearn)
            n_epochs=10,
            # ['log', 'hinge']
            loss='hinge',
            # ['fastxml', 'dsimec']
            # I think that 'fastxml' put a 'l1' regularization while 'dsimec' a 'l2'
            optimization='dsimec',
            # False for FastXML and PFastXML, True for PFastreXML
            leaf_classifiers=True,
            # ['auto', 'sgd', 'liblinear']
            engine='liblinear',
            # C param for classifier (SVM or log regression)
            C=1,
            leaf_probs=True,
            verbose=True,
        )
        self.metrics_handler = metrics_handler

    def run_init(self) -> None:
        """Optional extra initialization hook (tokenizers, caches, etc.)."""
        pass
    
    def train(self, dataloaders):
        """
        Training loop.
        Args:
            dataloaders (dict): Dictionary with keys: 
                'tasks_train': torch.utils.data.DataLoader loading task nodes (subtrees of the taxonomy) - Train split,
                'tasks_validation': torch.utils.data.DataLoader loading task nodes (subtrees of the taxonomy) - Validation split,
                'tasks_test': torch.utils.data.DataLoader loading task nodes (subtrees of the taxonomy) - Test split,
                'embeddings': torch tensor of per token embeddings (e.g. <pad> -> all zeroes, 'dog' -> pretrained embedding vector),
        """
        def safe_doc_embedding(sample_tensor):
            ids = sample_tensor[sample_tensor != 0]
            if ids.numel() == 0:
                # Fallback: zero vector (or use UNK embedding if you prefer)
                return torch.zeros(self.embeddings.embedding_dim)
                # Alternative using UNK (id 1):
                # return self.embeddings(torch.tensor([1], device=ids.device)).squeeze(0)
            emb = self.embeddings(ids)
            return emb.mean(dim=0)
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        split = 'train'
        
        # Always set to False, no fine-tuning for this method
        start_train = time.perf_counter()

        # Get all data at the same time
        sparse_input = []
        all_labels = []
        for input_data, labels, _ in tqdm(dataloaders[f"global_{split}"], leave=False):
            for sample_tensor, sample_label in zip(input_data, labels):
                # Do not take <PAD> tokens in the average
                doc_embedding = safe_doc_embedding(sample_tensor)
                sparse_input.append(
                    csr_matrix(doc_embedding.detach().cpu().numpy()[None, :]).astype(np.float32)
                )                   
                # Remove the taxonomy root
                sample_label.remove(0)
                all_labels.append(sample_label)

        # Train the model
        weights = propensity(all_labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(sparse_input, all_labels, weights)

        self.model.save(str(self.model_path.with_name('model')))

        stop_train = time.perf_counter()
        print(f"> End training after {format_time_diff(start_train, stop_train)}")

    def save_model(self) -> None:
        """FastXML models are saved via Trainer.save during training."""
        pass
        
    def load_model(self) -> None:
        """
        Bind an Inferencer to the trained FastXML model (used for eval/inference).
        """
        model_path = str(self.model_path.with_name('model'))
        self.inferencer = Inferencer(model_path, leaf_probs=True)
    
    def load_model_finetuned(self) -> None:
        """FastXML does not support finetuned models."""
        pass
    
    @torch.no_grad()
    def inference_eval(self, input_data, **kwargs):
        """
        Single-batch inference helper returning dense scores [B, n_labels].
        Converts input token ids to average embedding -> sparse row -> predict -> dense vector.
        """ 

        def safe_doc_embedding(sample_tensor: torch.Tensor) -> torch.Tensor:
            ids = sample_tensor[sample_tensor != 0]
            if ids.numel() == 0:
                return torch.zeros(self.embeddings.embedding_dim, device=sample_tensor.device)
            emb = self.embeddings(ids)
            return emb.mean(dim=0)

        # input_data is a batch of token-id tensors [B, seq_len]
        batch_sparse = []
        for sample_tensor in input_data:
            doc_embedding = safe_doc_embedding(sample_tensor)
            batch_sparse.append(csr_matrix(doc_embedding.detach().cpu().numpy()[None, :]).astype(np.float32))

        assert self.inferencer is not None, "inferencer should not be None, ensure a model is loaded or trained"
        # FastXML inferencer can take a list of sparse rows
        pred_dicts = self.inferencer.predict(batch_sparse, fmt='dict') 
        out = []
        for sample_dict in pred_dicts:
            pred = torch.tensor(list(sample_dict.keys()), dtype=torch.int64)
            score = torch.tensor(list(sample_dict.values()), dtype=torch.float32)
            dense = torch.zeros(self.taxonomy.n_nodes, dtype=torch.float32)
            if len(pred) > 0:
                dense.scatter_(dim=0, index=pred, src=score)
            out.append(dense)
        return torch.stack(out, dim=0)  # [B, n_labels]

    def finetune(self, dataloaders):
        raise NotImplementedError("FastXML does not support fine-tuning.")
