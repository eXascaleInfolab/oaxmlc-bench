
import torch
from torch.utils.data import Dataset
import json, orjson
from tqdm import tqdm

class GlobalDataset(Dataset):
    """Standard dataset."""

    def __init__(self, indices, relevant_labels, one_hot_labels, cfg, train_dataset):
        """Constructor for GlobalDataset class.

        Constructs a dataset specific to be used with models created from PyTorch.

        Args:
            indices (list): A list containing indices of files to load as data.
            relevant_labels (tensor): A tensor containing the column indices that are relevant for all documents, i.e. all labels minus the root and the root of all sub-trees. It is applied as "slice" for the labels.
            one_hot_labels (boolean): If the labels need to be transformed to one-hot encoding.
            cfg (dict): Dict config of the framework.
            train_dataset (boolean): A flag telling if this is a training set or not. This is to avoid training on samples belonging to the selected task in few-shot experiments

        """
        super().__init__()
        self.cfg = cfg
        # Contains paths to file belonging to the dataset
        self.files = []
        self.relevant_labels = relevant_labels
        self.one_hot_labels = one_hot_labels
        self.taxonomy = self.cfg['taxonomy']
        subroot = self.taxonomy.idx_to_label[self.cfg['task_to_subroot'][self.cfg['selected_task']]]
        label_idx_in_selected_task = set([self.taxonomy.label_to_idx[subroot]] + [self.taxonomy.label_to_idx[node] for node in self.taxonomy.all_children(subroot)])
        for idx in tqdm(indices, leave=False, desc="Loading global dataset"):
            file_path = self.cfg['paths']['data'] / f"{idx}.json"
            with open(file_path, 'r') as f:
                _, labels = json.load(f)
                # If sample is in selected task and this is set as the training dataset, do not add it to the dataset
                if len(label_idx_in_selected_task.intersection(labels)) != 0 and self.cfg['fewshot_exp'] and train_dataset:
                    continue
                else:
                    self.files.append(file_path)

    def __len__(self):
        return len(self.files)

    # idx is the document/sample index
    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            text_tokenized, label = orjson.loads(f.read())
        text_tokenized = torch.tensor(text_tokenized, dtype=torch.int32)
        # Construct one-hot labels or keep integer labels
        if self.one_hot_labels:
            # Cast to tensor since label is a list
            label = torch.zeros(self.taxonomy.n_nodes).scatter_(0, torch.tensor(label, dtype=torch.int64), 1.)

        return text_tokenized, label, self.relevant_labels
    
class GlobalDatasetHectorTamlec(Dataset):
    """Standard dataset."""

    def __init__(self, indices, relevant_labels, one_hot_labels, cfg, train_dataset):
        """Constructor for GlobalDataset class.

        Constructs a dataset specific to be used with models created from PyTorch.

        Args:
            indices (list): A list containing indices of files to load as data.
            relevant_labels (tensor): A tensor containing the column indices that are relevant for all documents, i.e. all labels minus the root and the root of all sub-trees. It is applied as "slice" for the labels.
            one_hot_labels (boolean): If the labels need to be transformed to one-hot encoding.
            cfg (dict): Dict config of the framework.
            train_dataset (boolean): A flag telling if this is a training set or not. This is to avoid training on samples belonging to the selected task in few-shot experiments

        """
        super().__init__()
        self.cfg = cfg
        # Contains paths to file belonging to the dataset
        self.files = []
        self.relevant_labels = relevant_labels
        self.one_hot_labels = one_hot_labels
        self.taxonomy = self.cfg['taxonomy']
        #self.paths = []
        
        subroot = self.taxonomy.idx_to_label[self.cfg['task_to_subroot'][self.cfg['selected_task']]]
        label_idx_in_selected_task = set([self.taxonomy.label_to_idx[subroot]] + [self.taxonomy.label_to_idx[node] for node in self.taxonomy.all_children(subroot)])
        for idx in tqdm(indices, leave=False, desc="Loading global dataset"):
            file_path = self.cfg['paths']['data'] / f"{idx}.json"
            with open(file_path, 'r') as f:
                _, labels = json.load(f)
                # If sample is in selected task and this is set as the training dataset, do not add it to the dataset
                if len(label_idx_in_selected_task.intersection(labels)) != 0 and self.cfg['fewshot_exp'] and train_dataset:
                    continue
                else:
                    #self.paths.append(self.cfg['paths']['paths_per_doc'] / f"{idx}.json")
                    self.files.append(file_path)

    def __len__(self):
        return len(self.files)

    # idx is the document/sample index
    def __getitem__(self, idx):
        #doc_paths = orjson.loads(self.paths[idx].read_bytes())
        with open(self.files[idx], 'rb') as f:
            text_tokenized, label = orjson.loads(f.read())
        text_tokenized = torch.tensor(text_tokenized, dtype=torch.int32)
        # Construct one-hot labels or keep integer labels
        if self.one_hot_labels:
            # Cast to tensor since label is a list
            label = torch.zeros(self.taxonomy.n_nodes).scatter_(0, torch.tensor(label, dtype=torch.int64), 1.)

        return text_tokenized, label, self.relevant_labels