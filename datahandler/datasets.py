import torch
from torch.utils.data import Dataset
import numpy as np
import copy
import json, orjson
from tqdm import tqdm


class TasksDataset(Dataset):
    """Dataset used with tasks being sub-trees in the taxonomy."""

    def __init__(self, indices, relevant_labels, one_hot_labels, cfg):
        """Constructor for TasksDataset class.

        Constructs a dataset specific to be used with models created from PyTorch.

        Args:
            indices (list): A list containing indices of files to load as data.
            relevant_labels (list): A list of tensors containing the column indices that are relevant for each sub-tree. It is applied as "slice" for the labels.
            one_hot_labels (boolean): If the labels need to be transformed to one-hot encoding.
            cfg (dict): Dict config of the framework.

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
        for task_id in tqdm(range(len(indices)), leave=False, desc="Loading tasks"):
            task_files = []
            for idx in indices[task_id]:
                file_path = self.cfg['paths']['data'] / f"{idx}.json"
                with open(file_path, 'r') as f:
                    _, labels = json.load(f)
                    # If sample is in the current task and the selected task, do not add it to the dataset
                    # Except for the selected task where we keep all samples
                    if len(label_idx_in_selected_task.intersection(labels)) != 0 and self.cfg['fewshot_exp'] and task_id != self.cfg['selected_task']:
                        continue
                    else:
                        task_files.append(file_path)
            self.files.append(task_files)

    def __len__(self):
        return len(self.files)

    # idx is the task_id
    def __getitem__(self, idx):
        input_data = []
        labels = []
        for file_path in self.files[idx]:
            with open(file_path, 'rb') as f:
                text_tokenized, label = orjson.loads(f.read())
            input_data.append(torch.tensor(text_tokenized, dtype=torch.int32))
            # Construct one-hot labels or keep integer labels
            if self.one_hot_labels:
                # Cast to tensor since label is a list
                one_hot_label = torch.zeros(self.taxonomy.n_nodes).scatter_(0, torch.tensor(label, dtype=torch.int64), 1.)
                labels.append(one_hot_label)
            else:
                labels.append(label)

        if self.one_hot_labels: labels = torch.vstack(labels)
        input_data = torch.vstack(input_data)

        return input_data, labels, self.relevant_labels[idx], idx


class TasksDatasetHectorTamlec(Dataset):
    """Dataset used with tasks being sub-trees in the taxonomy."""

    def __init__(self, indices, relevant_labels, one_hot_labels, cfg):
        """Constructor for TasksDataset class.

        Constructs a dataset specific to be used with models created from PyTorch.

        Args:
            indices (list): A list containing indices of files to load as data.
            relevant_labels (list): A list of tensors containing the column indices that are relevant for each sub-tree. It is applied as "slice" for the labels.
            one_hot_labels (boolean): If the labels need to be transformed to one-hot encoding.
            cfg (dict): Dict config of the framework.

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
        for task_id in tqdm(range(len(indices)), leave=False, desc="Loading tasks"):
            task_files = []
            task_paths = []
            for idx in indices[task_id]:
                file_path = self.cfg['paths']['data'] / f"{idx}.json"
                #doc_paths = self.cfg['paths']['paths_per_doc'] / f"{idx}.json"
                with open(file_path, 'r') as f:
                    _, labels = json.load(f)
                    # If sample is in the current task and the selected task, do not add it to the dataset
                    # Except for the selected task where we keep all samples
                    if len(label_idx_in_selected_task.intersection(labels)) != 0 and self.cfg['fewshot_exp'] and task_id != self.cfg['selected_task']:
                        continue
                    else:
                        #task_paths.append(doc_paths)
                        task_files.append(file_path)
            self.files.append(task_files)
            #self.paths.append(task_paths)    
    def __len__(self):
        return len(self.files)

    # idx is the task_id
    def __getitem__(self, idx):
        return self.files[idx],  self.relevant_labels[idx], idx

class ResampledTasksDataset(Dataset):
    """Dataset used with tasks being sub-trees in the taxonomy. Tasks are resampled until a specific number of elements, this if for Tamlec and Hector to always have a complete batch during training."""

    def __init__(self, indices, relevant_labels, cfg):
        """Constructor for TasksDataset class.

        Constructs a dataset specific to be used with models created from PyTorch.

        Args:
            indices (list): A list containing indices of files to load as data.
            relevant_labels (list): A list of tensors containing the column indices that are relevant for each sub-tree. It is applied as "slice" for the labels.
            cfg (dict): Dict config of the framework.

        """
        super().__init__()
        self.cfg = cfg
        # Contains paths to file belonging to the dataset
        self.files = []
        self.relevant_labels = relevant_labels
        self.taxonomy = self.cfg['taxonomy']
        self.batch_size = self.cfg['batch_size_train']
        self.rng = np.random.default_rng()
        #self.paths = []
        subroot = self.taxonomy.idx_to_label[self.cfg['task_to_subroot'][self.cfg['selected_task']]]
        label_idx_in_selected_task = set([self.taxonomy.label_to_idx[subroot]] + [self.taxonomy.label_to_idx[node] for node in self.taxonomy.all_children(subroot)])
        for task_id in tqdm(range(len(indices)), leave=False, desc="Loading tasks"):
            task_files = []
            #task_paths = []
            for idx in indices[task_id]:
                file_path = self.cfg['paths']['data'] / f"{idx}.json"
                #doc_paths_file = self.cfg['paths']['paths_per_doc'] / f"{idx}.json"

                with open(file_path, 'r') as f:
                    _, labels = json.load(f)
                    # If sample is in the current task and the selected task, do not add it to the dataset
                    # Except for the selected task where we keep all samples
                    if len(label_idx_in_selected_task.intersection(labels)) != 0 and self.cfg['fewshot_exp'] and task_id != self.cfg['selected_task']:
                        continue
                    else:
                        #task_paths.append(doc_paths_file)
                        task_files.append(file_path)

            new_task_files = copy.deepcopy(task_files)
            #new_task_paths = copy.deepcopy(task_paths)
            # Resampling if a task is too small to have at least batch_size * accum_iter elements
            if len(task_files) < self.batch_size * self.cfg['tamlec_params']['accum_iter']:
                indices_to_add = self.rng.choice(len(task_files), size=self.batch_size*self.cfg['tamlec_params']['accum_iter']-len(task_files), replace=True)
                for idx in indices_to_add:
                    new_task_files.append(task_files[idx])
                    #new_task_paths.append(task_paths[idx])
            #self.paths.append(new_task_paths)
            self.files.append(new_task_files)

    def __len__(self):
        return len(self.files)

    # idx is the task_id
    def __getitem__(self, idx):
        return self.files[idx], self.relevant_labels[idx], idx

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