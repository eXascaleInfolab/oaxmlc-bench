import random
import torch
from torch.utils.data import Sampler
import numpy as np
from tqdm import tqdm

class SubtreeSampler(Sampler):
    """SubtreeSampler Class

    This class is a custom data sampler designed for tasks involving hierarchical taxonomies. It manages the sampling of subtrees, handles the creation of support and query sets for protonet, maml and bdc algorithms, and implements various collation strategies tailored to different algorithms.

    Attributes:
        cfg (dict): Configuration dictionary
        dataset (list): List of subtrees, where each subtree is represented as a tuple with:
            - torch.tensor: tokenized documents.
            - torch.tensor: labels.
            - torch.tensor: Column indices of relevant labels.
        batch_size (int): Number of examples in a batch.
        possible_subtrees (list): List of subtree indices.
        items_idx_per_label (dict): Mapping of subtree indices to dictionaries that map labels to sets of item indices.

    Methods:
        __len__():
            Returns the number of subtrees in the dataset.

        __iter__():
            Iterates through the subtrees, yielding subtree indices.

        collate_standard(seed=None):
            Generates a closure for standard collation, creating mini-batches of inputs and labels.

        collate_fewshot_eval(seed=None):
            Collates data for evaluation on the few-shot methods, where we do not need a separation in support and query sets

        collate_no_batch(input_data):
            Collates data for FastXML, transforming inputs into sparse matrices and grouping labels.

        collate_hector_tamlec(seed=None):
            Generates a closure for Hector/Tamlec collation, creating mini-batches while maintaining taxonomy-specific constraints.

        collate_fewshot(input_data):
            Selects the collation function based on the `sampling_strategy` configuration.

        standard_task(input_data):
            Implements a standard few-shot task sampling strategy, constructing support and query sets.

        min_including(input_data):
            Implements the "minimum including" algorithm for few-shot task sampling, ensuring minimum support examples for each class.
    """

    def __init__(self, dataset, cfg, batch_size):
        super().__init__(data_source=None)
        self.cfg = cfg
        self.taxonomy = self.cfg['taxonomy']
        self.dataset = dataset
        self.batch_size = batch_size
        self.possible_subtrees = list(range(len(self.dataset)))
        if self.cfg['method'] in ['protonet', 'bdc', 'maml']:
            # key: subtree index, values: dict with labels as key and set of items index having this label as values
            self.items_idx_per_label = {}
            for subtree_idx in tqdm(self.possible_subtrees, leave=False, desc="Loading subtree"):
                self.items_idx_per_label[subtree_idx] = {}
                column_indices = self.dataset[subtree_idx][2]
                relevant_labels = self.dataset[subtree_idx][1][:, column_indices]
                # Get all labels from all items
                for i, col_idx in enumerate(column_indices):
                    col_idx = col_idx.item()
                    column = relevant_labels[:,i]
                    indices = torch.nonzero(column, as_tuple=True)[0]
                    for index in indices:
                        if col_idx in self.items_idx_per_label[subtree_idx]:
                            self.items_idx_per_label[subtree_idx][col_idx].add(index.item())
                        else:
                            self.items_idx_per_label[subtree_idx][col_idx] = {index.item()}


    def __len__(self):
        return len(self.possible_subtrees)


    def __iter__(self):
        for subtree_idx in self.possible_subtrees:
            self.subtree_idx = subtree_idx
            self.sampled_labels = self.dataset[subtree_idx][1].tolist()

            yield subtree_idx


    def collate_standard(self, seed=None):
        def _collate_standard(input_data):
            """Collate function for standard classification tasks. The seed can be defined in the closure so that in training we have random batches while in evaluation we have always the same batches.

            Args:
                input_data (list): A list of one tuple containing three elements:
                    - A torch.tensor containing the inputs for the model.
                    - A torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

            Returns:
                tuple: A tuple containing three elements:
                    - A list of torch.tensor containing the inputs for the model.
                    - A list of torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
            """

            document_data = input_data[0][0]
            labels_data = input_data[0][1]
            column_indices = input_data[0][2]

            # Create mini-batches
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            n_batches = int(np.ceil(len(indices)/self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, n_batches)]
            # Remove empty batch if any
            batches_indices = [batch for batch in batches_indices if len(batch) != 0]
            # Create batched input and labels
            batched_input = [document_data[batch] for batch in batches_indices]
            batched_labels = [labels_data[batch] for batch in batches_indices]

            return (
                batched_input,
                batched_labels,
                column_indices,
            )
        return _collate_standard


    def collate_fewshot_eval(self, seed=None):
        def _collate_fewshot_eval(input_data):
            """Collate function for the evaluation of standard classification tasks. The seed can be defined in the closure so that in training we have random batches while in evaluation we have always the same batches. This function ensures the evaluation of the fewshot methods are deterministic as well.

            Args:
                input_data (list): A list of one tuple containing three elements:
                    - A torch.tensor containing the inputs for the model.
                    - A torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

            Returns:
                tuple: A tuple containing three elements:
                    - A list of torch.tensor containing the inputs for the model.
                    - A list of torch.tensor containing the labels.
                    - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
            """

            document_data = input_data[0][0]
            labels_data = input_data[0][1]
            column_indices = input_data[0][2]

            # Create mini-batches
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            self.batch_size = max(1, min(64, int(np.ceil(len(document_data)))))  # ≈20% of N, never >64, never <1
            n_batches = int(np.ceil(len(indices)/self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(0, n_batches)]
            # Remove empty batch if any
            batches_indices = [batch for batch in batches_indices if len(batch) != 0]

            # Create batched support and query sets, so that all documents are going once in the query set
            support_sets = []
            all_support_labels = []
            query_sets = []
            all_query_labels = []

            for i in range(len(batches_indices)):
                random.seed(seed)
                query_idx = batches_indices[i]

                support_idx = set()
                for sampled_label in self.sampled_labels:
                    # Pick n_shot items if possible, otherwise take all possible elements
                    sampled_indices = random.sample(self.items_idx_per_label[self.subtree_idx][sampled_label].difference(query_idx), min(self.cfg['n_shot'], len(self.items_idx_per_label[self.subtree_idx][sampled_label].difference(query_idx))))
                    if len(sampled_indices) == 0:
                        sampled_indices = [list(self.items_idx_per_label[self.subtree_idx][sampled_label])[0]]
                    support_idx = support_idx.union(sampled_indices)
                # Cast to list for next operations
                support_idx = list(support_idx)

                # Aggregate all support items together
                support_items = document_data[support_idx]
                support_labels = labels_data[support_idx]
                support_sets.append(support_items)
                all_support_labels.append(support_labels)

                # Aggregate all query items together
                query_items = document_data[query_idx]
                query_labels = labels_data[query_idx]
                query_sets.append(query_items)
                all_query_labels.append(query_labels)

            return (
                support_sets,
                all_support_labels,
                query_sets,
                all_query_labels,
                column_indices,
            )
        return _collate_fewshot_eval


    def collate_no_batch(self, input_data):
        """Collate function that returns the entire dataset with no batches.

        Args:
            input_data (list): A list of one tuple containing three elements:
                - A torch.tensor containing the inputs for the model.
                - A list of lists containing the labels.
                - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

        Returns:
            tuple: A tuple containing three elements:
                - A torch.tensor containing the inputs for the model.
                - A list of lists containing the labels.
                - A list of lists with relevant column indices, i.e. labels that appear in this subtree.
        """

        document_data = input_data[0][0]
        labels_data = input_data[0][1]
        column_indices = input_data[0][2]

        return (
            document_data,
            labels_data,
            column_indices,
        )
    def collate_hector_eval_flat(self, seed=None):
        def _collate_hector_eval_flat(input_data):
            """
            Eval-time collate for Hector/Tamlec that matches the generic eval_step:
            returns (batched_input_list, flat_labels_list, column_indices)

            - batched_input_list: List[Tensor] (mini-batches); inference_eval will vstack internally
            - flat_labels_list:   List[List[int]] length B (each a 1-D list of label ids)
            - column_indices:     1-D Tensor/list of task-relevant class ids
            """
            document_data  = input_data[0][0]   # Tensor [N, seq_len]
            labels_data    = input_data[0][1]   # List[List[int]] per sample (global for the task)
            column_indices = input_data[0][2]   # 1-D indices of relevant classes for this task
            task_id        = input_data[0][3]

            # --- make mini-batches: same logic as collate_hector_tamlec ---
            indices = np.arange(len(document_data))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)
            n_batches = int(np.ceil(len(indices) / self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batches)]
            batches_indices = [b for b in batches_indices if len(b) != 0]

            batched_input = []
            flat_labels = []   # <--- IMPORTANT: flat list (length B)

            if self.cfg['method'] == 'tamlec':
                # keep subtree root (no global root)
                for batch in batches_indices:
                    batched_input.append(document_data[batch])
                    for idx in batch:
                        labs = [self.cfg['task_to_subroot'][task_id]] + [lab for lab in labels_data[idx] if lab in column_indices]
                        flat_labels.append(labs)
            else:
                # HECTOR: prepend global root 0 + subroot
                for batch in batches_indices:
                    batched_input.append(document_data[batch])
                    for idx in batch:
                        labs = [0, self.cfg['task_to_subroot'][task_id]] + [lab for lab in labels_data[idx] if lab in column_indices]
                        flat_labels.append(labs)

            return (batched_input, flat_labels, column_indices)
        return _collate_hector_eval_flat

    def collate_hector_tamlec(self, seed=None):
        def _collate_hector_tamlec(input_data):
            doc_files, column_indices, task_id = input_data[0]

            indices = np.arange(len(doc_files))
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(indices)

            n_batches = int(np.ceil(len(indices) / self.batch_size))
            batches_indices = [indices[i*self.batch_size:(i+1)*self.batch_size] for i in range(n_batches)]
            batches_indices = [b for b in batches_indices if len(b) != 0]
            
            return doc_files, column_indices, task_id, batches_indices
        return _collate_hector_tamlec


    # Select the fewshot collate function according the selected sampling strategy
    # Only for few-shot method
    def collate_fewshot(self, input_data):
        if self.cfg['sampling_strategy'] == 'standard':
            return self.standard_task(input_data)
        elif self.cfg['sampling_strategy'] == 'min_including':
            return self.min_including(input_data)
        else:
            raise NotImplementedError(f"Please either implement a collate function for {self.cfg['sampling_stategy']} sampling strategy, or add it to the conditions above this line.")


    def standard_task(self, input_data):
        """Collate function for standard few-shot tasks.

        Construct a batch for a specific task. It first constructs the support set by randomly sampling `n_shot` items per each sampled class if possible, or take all possible elements. It then constructs the query set by randomly sampling `n_query` items from the remaining ones that are not already in the support set. It is possible that the size of the query set is smaller than `n_query`.

        Args:
            input_data (list): A list of one tuple containing three elements:
                - A torch.tensor containing the inputs for the model.
                - A torch.tensor containing the labels.
                - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

        Returns:
            tuple: A tuple (support_items, support_labels, query_items, query_items, column_indices) where:
                - support_items and query_items are torch.tensor containing the inputs for the model.
                - support_labels and query_labels are torch.tensor representing the labels.
                - column_indices is a torch.tensor with relevant column indices, i.e. labels that appear in this subtree.
        """

        data_items = input_data[0][0]
        labels_data = input_data[0][1].cpu().numpy()
        column_indices = input_data[0][2]

        support_idx = set()
        for sampled_label in self.sampled_labels:
            # Pick n_shot items if possible, otherwise take all possible elements
            sampled_indices = random.sample(self.items_idx_per_label[self.subtree_idx][sampled_label], min(self.cfg['n_shot'], len(self.items_idx_per_label[self.subtree_idx][sampled_label])))
            support_idx = support_idx.union(sampled_indices)
        # Cast to list for next operations
        support_idx = list(support_idx)

        # Remaining samples that are not in the support set are all put in the query set
        query_idx = set(range(len(labels_data))).difference(support_idx)
        # Take maximum n_query elements
        try:
            query_idx = random.sample(list(query_idx), self.cfg['n_query'])
        # If len(query_idx) < self.cfg['n_query'] then we have a ValueError
        except ValueError:
            query_idx = list(query_idx)

        # Still if the query set is empty, we will sample new labels
        if len(query_idx) == 0: return (None, None, None, None, None)

        # Aggregate all support items together
        support_items = data_items[support_idx]
        support_labels = torch.tensor(labels_data[support_idx])
            
        # Aggregate all query items together
        query_items = data_items[query_idx]
        query_labels = torch.tensor(labels_data[query_idx])

        return (
            support_items,
            support_labels,
            query_items,
            query_labels,
            column_indices,
        )


    def min_including(self, input_data):
        """Collate function implementing Minimum including algorithm.

        Construct a batch for a specific task. The support set is constructed using the `minimum including algorithm`: in a first part, it randomly samples items until there is at least `n_shot` items for all sampled classes. In a second part, it tests for each sampled items if removing it makes the number of occurrences for a class less than `n_shot`, in which case the item is kept in the support set, or is removed otherwise. The query set is constructed by randomly sampling `n_query` items from the remaining ones that are not already in the support set. It is possible that the size of the query set is smaller than `n_query`.

        Args:
            input_data (list): A list of one tuple containing three elements:
                - A torch.tensor containing the inputs for the model.
                - A torch.tensor containing the labels.
                - A torch.tensor with relevant column indices, i.e. labels that appear in this subtree.

        Returns:
            tuple: A tuple (support_items, support_labels, query_items, query_items, column_indices) where:
                - support_items and query_items are torch.tensor containing the inputs for the model.
                - support_labels and query_labels are torch.tensor representing the labels.
                - column_indices is a tensor with relevant column indices, i.e. labels that appear in this subtree.
        """

        dict_data = input_data[0][0]
        labels_data = input_data[0][1].cpu().numpy()
        column_indices = input_data[0][2]

        sampled_idx = set()
        # Init count for each sampled label to 0
        counts = {label: 0 for label in self.sampled_labels}

        # First part, for all sampled labels, sample items until counts[i] >= n_shot
        for sampled_label in self.sampled_labels:
            n_to_sample = self.cfg['n_shot'] - counts[sampled_label]
            if n_to_sample > 0:
                # Sample items having this label that are not already sampled
                sampled_indices = random.sample(list(self.items_idx_per_label[self.subtree_idx][sampled_label].difference(sampled_idx)), n_to_sample)
                # Update the counts for all labels
                for idx in sampled_indices:
                    labels = labels_data[idx]
                    # Take only the sampled labels
                    for label in self.sampled_labels.intersection(labels):
                        counts[label] += 1
                    sampled_idx.add(idx)

        # Second part, for all sampled items, check if we need to keep it, i.e. removing it would make counts[i] < n_shot
        support_idx = set()
        for idx in sampled_idx:
            labels = labels_data[idx]
            # For all labels check if we need to keep the sample
            sample_to_keep = False
            for label in self.sampled_labels.intersection(labels):
                # In this case the sample cannot be removed
                if counts[label] <= self.cfg['n_shot']:
                    sample_to_keep = True
                    break
            if sample_to_keep:
                support_idx.add(idx)
            # If we do not keep the sample, do not add it to support set and update the counts
            else:
                for label in self.sampled_labels.intersection(labels):
                    counts[label] -= 1
        # Cast to list for next operations
        support_idx = list(support_idx)

        # Remaining samples that are not in the support set are all put in the query set
        query_idx = set(range(len(labels_data))).difference(support_idx)
        # Take maximum n_query elements
        try:
            query_idx = random.sample(list(query_idx), self.cfg['n_query'])
        # If len(query_idx) < self.cfg['n_query'] then we have a ValueError
        except ValueError:
            query_idx = list(query_idx)

        # Still if the query set is empty, we will sample new labels
        if len(query_idx) == 0: return (None, None, None, None, None)

        # Aggregate all support items together
        support_items = {}
        for key, values in dict_data.items():
            support_items[key] = values[support_idx]
        support_labels = torch.tensor(labels_data[support_idx])

        # Aggregate all query items together
        query_items = {}
        for key, values in dict_data.items():
            query_items[key] = values[query_idx]
        query_labels = torch.tensor(labels_data[query_idx])

        return (
            support_items,
            support_labels,
            query_items,
            query_labels,
            column_indices,
        )
