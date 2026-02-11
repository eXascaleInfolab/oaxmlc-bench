from tqdm import tqdm
import torch
import numpy as np
import time
from scipy.sparse import csr_matrix
import warnings
import multiprocessing

from algorithms.base_algorithm import AbstractAlgorithm
from misc.utils.time import format_time_diff
from models.Parabel.parabel import ParabelModel
from scipy.sparse import vstack
from pathlib import Path
from models import Parabel


class ParabelAlg(AbstractAlgorithm):
    """Parabel algorithm wrapper using sparse embeddings."""
    def __init__(
            self,  
            metrics_handler,
            taxonomy,
            embeddings_loader, # Embeddings loader to initialize the BaseBertModel
            model_path: Path = Path(), # Path to a model to load the model from
            device: str = 'cpu', # Device to use (cpu, cuda:0,
            verbose=True,
            **kwargs
            ): 
        """Initialize Parabel model and embedding utilities."""
        # Set the start method to 'spawn' to reduce number of files descriptors inherited by the children processes
        # So we can have more processes running at the same time during the evaluation
        current = multiprocessing.get_start_method(allow_none=True)
        if current is None:
            multiprocessing.set_start_method('spawn')       # OK: first and only time
        elif current != 'spawn':
            if verbose:
                print(f"> Warning: start method already '{current}', not changing to 'spawn'.")
        
        self.taxonomy = taxonomy
        self.device = device
        self.model_path = model_path
        # Padding index set at 0, freeze=True to disable training on embeddings, it does not matter in here
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings_loader, padding_idx=0, freeze=True)
        del embeddings_loader
        print(f"> Initialize model...")
        self.model = ParabelModel()
        self.metrics_handler = metrics_handler
        self.search_width = 10
        # Number of processes that compute the evaluation in multiprocessing
        self.n_proc = 32
        
        # Mapping back from title to idx
        self.title_to_idx = {}
        for label, title in self.taxonomy.label_to_title.items():
            self.title_to_idx[title] = self.taxonomy.label_to_idx[label]
        
    def load_model(self):
        """Load a saved Parabel model from disk."""
        self.model = ParabelModel().load_model(self.model_path)

    def load_model_finetuned(self):
        """Parabel does not support finetuned models."""
        pass
    
    def run_init(self)-> None:
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
        training = True
        # Always set to False, no fine-tuning for this method
        finetuning = False
        # At evaluation, the maximum number of nodes that will be considered
        # for checking for possible label assignment to the data point at each level of the tree. 
        search_width = self.search_width
        # Number of processes that compute the evaluation in multiprocessing
        n_proc = self.n_proc
        start_train = time.perf_counter()

        
        # Get all data at the same time
        sparse_input = []
        all_labels = []
        for input_data, labels, _, _ in tqdm(dataloaders[f"global_{split}"], leave=False):
            for sample_tensor, sample_label in zip(input_data, labels):
                # Do not take <PAD> tokens in the average
                doc_embedding = safe_doc_embedding(sample_tensor)
                sparse_input.append(
                    csr_matrix(doc_embedding.detach().cpu().numpy()[None, :]).astype(np.float32)
                )
                # Get list of label names
                label_names = [self.taxonomy.label_to_title[self.taxonomy.idx_to_label[lab]] for lab in sample_label if lab != 0]
                # Add a TAB character as expected in the model
                all_labels.append('\t'.join(label_names))
        sparse_input = vstack(sparse_input, format='csr', dtype=np.float32)
        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.train(sparse_input, all_labels, max_labels_per_leaf=200, convert_X=False, verbose=True)
        self.model.save_model(self.model_path)

        stop_train = time.perf_counter()
        print(f"> End training after {format_time_diff(start_train, stop_train)}")

    def save_model(self, path):
        """Save the Parabel model to the given path."""
        self.model.save_model(path)

    def inference_eval(self, input_data, **kwargs):
        """Run inference and return dense prediction scores."""
        def safe_doc_embedding(sample_tensor):
            ids = sample_tensor[sample_tensor != 0]
            if ids.numel() == 0:
                # Fallback: zero vector (or use UNK embedding if you prefer)
                return torch.zeros(self.embeddings.embedding_dim)
                # Alternative using UNK (id 1):
                # return self.embeddings(torch.tensor([1], device=ids.device)).squeeze(0)
            emb = self.embeddings(ids)
            return emb.mean(dim=0)
        # Construct sparse matrices and the one-hot labels
        sparse_input = []
        for sample_tensor in input_data:
            # Do not take <PAD> tokens in the average
            doc_embedding = safe_doc_embedding(sample_tensor)
            sparse_input.append(
                csr_matrix(doc_embedding.detach().cpu().numpy()[None, :]).astype(np.float32)
            )
            
        # Get the predictions
        with multiprocessing.Pool(processes=self.n_proc) as pool:
            all_dict_preds = pool.starmap(self.model.predict, zip(sparse_input, len(sparse_input)*[self.search_width]))
        #all_dict_preds = []
        #for sample in tqdm(sparse_input, leave=False):
        #    dict_pred = self.model.predict(sample, self.search_width)
        #    all_dict_preds.append(dict_pred)
        # Transform to one-hot encoding
        all_predictions = []
        for dict_pred in tqdm(all_dict_preds, leave=False):
            pred_sorted = list(dict_pred.keys())
            pred_sorted = [self.title_to_idx[lab] for lab in pred_sorted]
            pred_probabilities = list(dict_pred.values())
            pred = torch.tensor(pred_sorted, dtype=torch.int64)
            score = torch.tensor(pred_probabilities)
            assert len(pred) == len(score)
            pred_sample = torch.zeros(self.taxonomy.n_nodes, dtype=torch.float32)
            pred_sample.scatter_(dim=0, index=pred, src=score)
            all_predictions.append(pred_sample)
        all_predictions = torch.stack(all_predictions, dim=0)
        return all_predictions
    
    def finetune(self, dataloaders):
        raise NotImplementedError("Parabel does not support fine-tuning.")
