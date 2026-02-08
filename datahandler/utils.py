import re
from tqdm import tqdm
import re
import nltk
import copy
import random
import sentencepiece as spm
import numpy as np
import torchtext.vocab as tvoc
from collections import defaultdict
from math import ceil
import torch
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Tools for text pre-processing
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
stop_words = set(nltk.corpus.stopwords.words("english"))
lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer = nltk.stem.snowball.SnowballStemmer('english') # type: ignore


def get_split_datastructures(cfg):
    items_per_label = {}
    labels_per_idx = {}
    taxonomy = cfg['taxonomy']
    for doc_path in cfg['paths']["data"].iterdir():
        with open(doc_path) as f:
            _, labels = json.load(f)
        
        #labels.remove(0)
        idx = doc_path.stem 
        labels = [taxonomy.idx_to_label[label] for label in labels]
        labels_per_idx[idx] = labels
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
    
    level = 1
    subtrees_to_keep = [node for node in taxonomy.level_to_labels[level] if node not in taxonomy.leaves]
    subtrees_to_keep = sorted(subtrees_to_keep)
    return items_per_label, taxonomy, subtrees_to_keep, labels_per_idx
    
            
def text_preprocessing(raw_line, simple=True):
    # Lower characters, remove trailing spaces, remove end of line
    preproc_line = raw_line.lower().strip().replace('\n', '')
    # Remove all special characters, keep only letters, numbers and spaces
    preproc_line = re.sub(r"[^\w\s]", '', preproc_line)
    if simple:
        return preproc_line

    else:
        all_words = nltk.tokenize.word_tokenize(preproc_line)
        new_preproc_line = []
        for word in all_words:
            # Remove stopwords
            if word.casefold() in stop_words: continue
            # Stemming
            # word = stemmer.stem(word)
            # Lemmatizing
            word = lemmatizer.lemmatize(word)
            new_preproc_line.append(word.strip())

        new_preproc_line = ' '.join(new_preproc_line).strip()
        return new_preproc_line

def tokenization_hf(texts, cfg):
    # Tokenization with the *same* tokenizer used by DEXA (HF AutoTokenizer)
    """
    Tokenize a list of texts with HuggingFace AutoTokenizer (DEXA style),
    pad/truncate to cfg['seq_length'], and return:
      - texts_tokenized: List[List[int]] input_ids
      - vocabulary: List[str] tokenizer vocab (tokens)
      - fully_padded_indices: indices whose attention_mask is all zeros

    Expected cfg keys (minimal):
      cfg['seq_length'] : int
      cfg['encoder_name'] or cfg['tokenizer_type'] : str (e.g. "msmarco-distilbert-base-v4")
      optional:
        cfg['tokenizer_kwargs'] : dict (passed to from_pretrained)
    Side-effects kept like your original:
      cfg['tamlec_params']['src_vocab'] set from tokenizer vocab

    Note: for DistilBERT, pad token exists; if not, we force pad_token = eos/sep.
    """
    from transformers import AutoTokenizer

    seq_len = int(cfg['seq_length'])

    # pick model name from cfg
    model_name = (
        cfg.get('encoder_name')
        or cfg.get('tokenizer_type')
        or cfg.get('tokenizer_name')
    )
    if model_name is None:
        raise ValueError("cfg must contain one of: encoder_name / tokenizer_type / tokenizer_name")

    tok_kwargs = dict(cfg.get('tokenizer_kwargs', {}))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **tok_kwargs)

    # Ensure we have a pad token (some models don't define it)
    if tokenizer.pad_token_id is None:
        # prefer eos, else sep, else unk
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token_id is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            tokenizer.pad_token = tokenizer.unk_token

    print("> Tokenizing data...")
    enc = tokenizer(
        texts,
        add_special_tokens=True,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="np",
    )


    input_ids = enc["input_ids"]           # (N, seq_len) np.int64
    attention_mask = enc["attention_mask"] # (N, seq_len) np.int64

    # convert to python lists to match your original output type
    texts_tokenized = input_ids.astype(int).tolist()

    # indices that are fully padded (mask all zeros)
    fully_padded_indices = np.where(attention_mask.sum(axis=1) == 0)[0].astype(int).tolist()

    # tokenizer vocabulary (token strings)
    # AutoTokenizer.get_vocab() returns {token_str: token_id}
    vocab_dict = tokenizer.get_vocab()
    # make list ordered by id
    vocabulary = [None] * (max(vocab_dict.values()) + 1)
    for tok, idx in vocab_dict.items():
        if 0 <= idx < len(vocabulary):
            vocabulary[idx] = tok
    # remove any holes (rare), keep only non-None
    vocabulary = [t for t in vocabulary if t is not None]

    # Build word_to_idx dict exactly like before (token -> id)
    # (No "▁" stripping here; that's SentencePiece-specific and wrong for BERT/DistilBERT.)
    word_to_idx = vocab_dict

    return texts_tokenized, vocabulary, fully_padded_indices, attention_mask


# Tokenization with the sentencepiece library
def tokenization(texts, cfg):
    """Tokenization Function

    Tokenizes a list of texts with SentencePiece and pads to seq_length.
    Returns token ids, vocabulary, and the indices of samples that became
    fully padding (all PAD=0).

    Args:
        texts (list of str): Input texts (already preprocessed upstream).
        cfg (dict): Configuration dictionary.

    Returns:
        tuple:
            - texts_tokenized (list[list[int]]): token ids (padded to seq_length)
            - vocabulary (list[str]): sentencepiece vocab (cleaned for leading ▁)
            - fully_padded_indices (list[int]): indices of samples whose sequence is all PAD=0
    """

    # Train the tokenizer
    # -1 for bos and eos tokens since we do not use them in classification
    spm.SentencePieceTrainer.train( # type: ignore
        sentence_iterator=iter(texts),
        model_prefix=str(cfg['paths']['tokenizer']).split('.')[0],
        vocab_size=cfg['voc_size'],
        character_coverage=1.0,
        model_type=cfg['tokenization_mode'],
        unk_id=1,
        pad_id=0,
        bos_id=-1,
        eos_id=-1
    )

    # Tokenize and get token ids for all data
    print(f"> Tokenizing data...")
    sp = spm.SentencePieceProcessor(model_file=str(cfg['paths']['tokenizer']), out_type=int, num_threads=16) # type: ignore
    texts_encoded = sp.encode(texts) # type: ignore

    PAD_ID = 0
    texts_tokenized = []
    fully_padded_indices = []

    for i, sample in enumerate(texts_encoded):
        # Truncate then pad with PAD=0 up to seq_length
        sample = sample[:cfg['seq_length']]
        if len(sample) < cfg['seq_length']:
            sample = sample + [PAD_ID] * (cfg['seq_length'] - len(sample))

        # Detect fully-padded (all zeros)
        if all(t == PAD_ID for t in sample):
            fully_padded_indices.append(i)

        texts_tokenized.append(sample)

    # Create lookup word->id from the created vocabulary
    word_to_idx = {}
    with open(cfg['paths']['vocabulary']) as f:
        for idx, line in enumerate(f):
            word, _ = line.replace('\n', '').split("\t")
            # SentencePiece adds a meta symbol "▁" at the beginning of tokens
            if word and word[0] == '▁' and len(word) > 1:
                word = word[1:]
            word_to_idx[word] = int(idx)

    vocabulary = list(word_to_idx.keys())

    # Source vocab for hector and tamlec
    cfg['tamlec_params']['src_vocab'] = tvoc.vocab(word_to_idx, min_freq=0)

    return texts_tokenized, vocabulary, fully_padded_indices



# In the taxonomy, make sure each sample has all available ancestor nodes of labels
def complete_labels(label_list, label_to_parents):
    """Complete Labels Function

    Expands a list of labels by including all their ancestor labels until reaching the root.

    Args:
        label_list (list): A list of labels (strings) to be expanded.
        label_to_parents (dict): A dictionary mapping each label (string) to a list of its parent labels (strings).

    Returns:
        list: A list of all labels from `label_list` along with their ancestor labels.
    """
    complete_label = set()
    for lab in label_list:
        complete_label.add(lab)
        # If lab is not in the taxonomy, then we have a key error
        # So simply continue
        try:
            parents = label_to_parents[lab]
        except KeyError:
            continue
        while 'root' not in parents:
            new_parents = set()
            for parent in parents:
                complete_label.add(parent)
                new_parents = new_parents.union(label_to_parents[parent])
            parents = list(new_parents)

    return list(complete_label)

def _compute_label_split_coverage(taxonomy, items_per_label, global_indices):
    """Returns dict[label] -> {'train': n_docs, 'val': n_docs, 'test': n_docs} using global split indices."""
    split_sets = {k: set(v) for k, v in global_indices.items()}
    coverage = {}
    for lab, idxs in items_per_label.items():
        s = set(idxs)
        coverage[lab] = {
            'train': len(s & split_sets['train']),
            'val'  : len(s & split_sets['val']),
            'test' : len(s & split_sets['test']),
        }
    return coverage


def preprocess_docs_and_taxonomy(documents, labels_data, cfg):
    if "seed" not in cfg.keys():
        cfg['seed'] = 42
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    print(f"> Using seed {cfg['seed']} for random operations")    
    print(f"> Get some stats on dataset")
    print(f">> Dataset has {len(documents)} documents")
    taxonomy = cfg['taxonomy']
    # key: label, values: list of items index having this label
    items_per_label = {}
    # key: index, values: list of labels 
    labels_per_idx = {}
    # Get all labels from all items
    for idx, labels in enumerate(labels_data):
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
            labels_per_idx[idx] = labels
    print(f">> Documents per label: {np.mean([len(items) for items in items_per_label.values()])}")
    print(f">> Labels per document: {np.mean([len(labels) for labels in labels_per_idx.values()])}")

    # Recursively remove leaves in the taxonomy that do not appear in the dataset (otherwise we could have a KeyError afterwards)
    print(f"> Removing leaves not appearing in dataset...")
    
    
    leaves_to_remove = [leaf for leaf in taxonomy.leaves if leaf not in [node for node in items_per_label.keys() if node not in taxonomy.label_to_children.keys() or taxonomy.is_leaf(node)]]
    while(len(leaves_to_remove) != 0):
        taxonomy.remove_leaves(leaves_to_remove)
        leaves_to_remove = [leaf for leaf in taxonomy.leaves if leaf not in [node for node in items_per_label.keys() if node not in taxonomy.label_to_children.keys() or taxonomy.is_leaf(node)]]

    # Prune all leaves until they all have at least min_freq samples
    if "min_freq" in cfg.keys() and cfg["min_freq"] is not None:
        min_freq = cfg["min_freq"]
    else:
        min_freq = 50
    n_before = len(taxonomy.all_children('root'))
    while(min([len(items_per_label[leaf]) for leaf in taxonomy.leaves])) < min_freq:
        for leaf in taxonomy.leaves:
            if len(items_per_label[leaf]) < min_freq:
                taxonomy.remove_leaves(leaf)
                del items_per_label[leaf]

    # Remove leaves on level 1
    level = 1
    taxonomy.remove_leaves([node for node in taxonomy.level_to_labels[level] if taxonomy.is_leaf(node)])
    all_labels = set(taxonomy.all_children('root'))
    
    print(f"> Pruned {n_before - len(all_labels)} leaf nodes having less than {min_freq} samples and the leaves on first level")
    print(f">> Updated taxonomy has {taxonomy.n_nodes} nodes and {len(taxonomy.leaves)} leaves and a height of {taxonomy.height}")

    # Take sub-trees only from level 1, if they are not leaves
    # Each sub-tree will be considered as a task
    subtrees_to_keep = [node for node in taxonomy.level_to_labels[level] if node not in taxonomy.leaves]
    # Remove sub-trees that have only one leaf, i.e. it is a straight line from root to leaf
    # subtrees_to_keep = [subtree_root for subtree_root in subtrees_to_keep if len([node for node in taxonomy.all_children(subtree_root) if node in taxonomy.leaves]) > 1]
    # Sorted to make sure sub-trees have always the same order
    subtrees_to_keep = sorted(subtrees_to_keep)
    subtrees_to_keep_set = set(subtrees_to_keep)
    print(f"> Possible sub-trees {len(subtrees_to_keep)} with root on level {level}")

    # Keep samples that have at least one label
    print(f"> Remove documents having no more label in common with taxonomy")
    indices_to_keep = []
    for idx, labels in labels_per_idx.items():
        doc_labels = [lab for lab in labels if lab in all_labels]
        # If no labels in common with taxonomy
        if len(doc_labels) == 0:
            continue
        # If labels are root of a sub-tree
        # If labels are root of a sub-tree
        elif set(doc_labels).issubset(subtrees_to_keep_set):
            continue
        else:
            indices_to_keep.append(idx)
    indices_to_keep = torch.tensor(indices_to_keep, dtype=torch.int64)

    # Update data: remove documents that have no more labels
    # Update labels: remove labels that are not in taxonomy anymore
    n_docs_before = len(documents)
    new_documents = []
    new_labels = []
    for idx in indices_to_keep:
        # Sorted to always keep the same order
        new_documents.append(documents[idx])
        new_labels.append(sorted(all_labels & set(labels_data[idx])))
    print(f">> Documents kept: {len(new_documents)} (out of {n_docs_before})")
    assert len(new_documents) == len(new_labels), f"Error in documents removing"

    # Update the data-structures after pruning and cleaning
    # key: label, values: set of items index having this label
    print(f"> Updating storage and stats...")
    items_per_label = {}
    # key: index, values: set of labels
    labels_per_idx = {} 
    # Get all labels from all items
    for idx, labels in enumerate(new_labels):
        labels_per_idx[idx] = labels
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
            
    print(f">> Now {np.mean([len(items) for items in items_per_label.values()])} documents per label")
    print(f">> Now {np.mean([len(labels) for labels in labels_per_idx.values()])} labels per document")

    
    print(f">> Constructing the labels...")
    final_labels = []
    for labels in new_labels:
        # Add root of taxonomy
        class_list = [0] + [taxonomy.label_to_idx[label] for label in labels]
        final_labels.append(class_list)
    # Save the tokenized documents and their labels into separate files
    # This will be loaded at runtime
    
    for idx, doc in tqdm(enumerate(new_documents), leave=False, desc="Saving processed documents", total=len(new_documents)):
        tokenized_doc = doc
        labels = final_labels[idx]
        file_path = cfg['paths']['data'] / f"{idx}.json"
        with open(file_path, 'w') as f:
            json.dump((tokenized_doc, labels), f)
            
    # Construct specific data for hector and tamlec
    # Target vocab, deepcopy to have another reference in memory
    voc_plus_pad = copy.deepcopy(taxonomy.label_to_idx)
    voc_plus_pad['<pad>'] = len(taxonomy.label_to_idx)
    cfg['tamlec_params']['trg_vocab'] = tvoc.vocab(voc_plus_pad, min_freq=0)
    # Taxonomy for hector is full taxonomy
    taxo_id_root = taxonomy.label_to_idx['root']
    children = {taxo_id_root: [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children['root']]}
    for curr_node in taxonomy.all_children('root'):
        if not taxonomy.is_leaf(curr_node):
            children[taxonomy.label_to_idx[curr_node]] = [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[curr_node]]
    cfg['tamlec_params']['taxos_hector'] = [(taxo_id_root, children)]
    # Taxonomies for tamlec are taxonomies of selected sub-trees
    cfg['tamlec_params']['taxos_tamlec'] = []
    for subtree_root in subtrees_to_keep:
        taxo_id_root = taxonomy.label_to_idx[subtree_root]
        children = {taxo_id_root: [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[subtree_root]]}
        for curr_node in taxonomy.all_children(subtree_root):
            if not taxonomy.is_leaf(curr_node):
                children[taxonomy.label_to_idx[curr_node]] = [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[curr_node]]
        cfg['tamlec_params']['taxos_tamlec'].append((taxo_id_root, children))
    # Abstract dict
    cfg['tamlec_params']['abstract_dict'] = {}
    for label, abstract in taxonomy.label_to_abstract.items():
        cfg['tamlec_params']['abstract_dict'][taxonomy.label_to_idx[label]] = abstract

    return items_per_label, taxonomy, subtrees_to_keep, labels_per_idx



def data_split(items_per_label, taxonomy, subtrees_to_keep, labels_per_idx, cfg):
    """Data Splitting and Pruning Function

    Processes and splits the dataset into training, validation, and test splits, at a global level and for each sub-task.

    This function performs several operations:
      - Computes basic statistics about the input dataset.
      - Prunes the taxonomy tree to remove under-represented or irrelevant nodes.
      - Filters out documents that no longer correspond to the pruned taxonomy.
      - Updates configuration (`cfg`) with vocabulary and taxonomy data for downstream modules.
      - Constructs global and per-task data splits.
      - Saves processed documents and label information as JSON files.

    Args:
        documents (torch.tensor): A tensor containing the tokenized dataset documents.
        labels_data (list of list of str): A list where each element is a list of labels associated with a document.
        cfg (dict): Configuration dictionary

    Returns:
        tuple: A tuple containing:
            - global_indices (dict): Dictionary with keys `'train'`, `'val'`, and `'test'`, each mapping to a sorted list of document indices for that split.
            - global_relevant_labels (torch.Tensor): Tensor containing all label indices used globally in the pruned taxonomy (excluding main root and roots of tasks).
            - all_tasks_indices (dict): Dictionary with keys `'train'`, `'val'`, and `'test'`, each mapping to a list of lists where each sub-list contains indices of documents belonging to a particular task.
            - all_tasks_relevant_labels (list of torch.Tensor): List of tensors where each tensor contains the relevant label indices for a given task.

    Notes:
        - Taxonomy pruning is guided by a `min_freq` threshold.
        - Tasks and labels are saved in a deterministic order for reproducibility.
        - The function modifies `cfg` in-place with vocabulary and task-related metadata for downstream use.
    """
    
    
    # Sort labels by increasing frequency (rarer first), keep only *current* leaf/inner labels that remain in taxonomy
    labels_sorted_by_freq = sorted(
        [lab for lab in items_per_label.keys() if lab in taxonomy.label_to_idx], 
        key=lambda l: len(items_per_label[l])
    )
    n_labels = len(labels_sorted_by_freq)
    n_rare = max(1, int(round(cfg['split']['rare_label_fraction'] * n_labels)))
    rare_labels = set(labels_sorted_by_freq[:n_rare])
    
    # 1) Build per-label holdout pool: take 10% of documents of each rare label (deterministic)
    random.seed(cfg['seed'])
    holdout_per_label = {}
    for lab in rare_labels:
        idxs = sorted(set(items_per_label[lab]))
        k = max(1, int(round(cfg['split']['rare_holdout_ratio'] * len(idxs)))) if len(idxs) > 0 else 0
        random.shuffle(idxs)
        holdout = set(idxs[:k]) if k > 0 else set()
        holdout_per_label[lab] = holdout

    # Remove holdout docs from items_per_label for initial splitting (we'll re-inject later if needed)
    for lab, ho in holdout_per_label.items():
        if ho:
            items_per_label[lab] = [i for i in items_per_label[lab] if i not in ho]
    

    # Various data
    cfg['tasks_size'] = []
    cfg['task_to_subroot'] = {}
    # Root belongs to no task
    cfg['label_to_tasks'] = {taxonomy.label_to_idx['root']: set()}
    
    # Dict of document indices per split
    global_indices = {'train': set(), 'val': set(), 'test': set()}
    # Dict of lists (by split) of lists (by tasks) of indices (integer)
    all_tasks_indices = {'train': [], 'val': [], 'test': []}
    # List of lists of relevant labels for each task
    all_tasks_relevant_labels = []
    # Set of all possible labels (root and sub-roots excluded)
    global_relevant_labels = set()
    
    for key in labels_per_idx.keys():
        doc_labels = []
        for lab in labels_per_idx[key]:
            doc_labels.append(taxonomy.label_to_idx[lab])
        labels_per_idx[key] = doc_labels
    
    print(f">> Train-val-test split...")
    doc_to_subtrees = defaultdict(set)
    for subtree_idx, subtree_root in enumerate(tqdm(subtrees_to_keep, leave=False)):
        
        # Dict of indices for this task
        task_indices = {'train': set(), 'val': set(), 'test': set()}
        all_subtree_indices = set()
        relevant_labels_task = []
        # Set idx of the node that is the root of this sub-tree
        cfg['task_to_subroot'][subtree_idx] = taxonomy.label_to_idx[subtree_root]
        try:
            cfg['label_to_tasks'][taxonomy.label_to_idx[subtree_root]].add(subtree_idx)
        except KeyError:
            cfg['label_to_tasks'][taxonomy.label_to_idx[subtree_root]] = set([subtree_idx])
            
        # Make the split for all other nodes in the sub-tree
        # process labels in rarity order (rarest first)
        subtree_nodes = set(taxonomy.all_children(subtree_root))
        nodes_in_order = [lab for lab in labels_sorted_by_freq if lab in subtree_nodes]
        for subnode in nodes_in_order:
            # Subnode to the task
            try:
                cfg['label_to_tasks'][taxonomy.label_to_idx[subnode]].add(subtree_idx)
            except KeyError:
                cfg['label_to_tasks'][taxonomy.label_to_idx[subnode]] = set([subtree_idx])
            # Add node to the relevant indices of this task and to the global
            relevant_labels_task.append(taxonomy.label_to_idx[subnode])
            global_relevant_labels.add(taxonomy.label_to_idx[subnode])
            # exclude holdout docs for this label (if any) from the initial splits
            indices_in_subnode = set(items_per_label.get(subnode, []))
            # Complete document indices of this task
            all_subtree_indices = all_subtree_indices.union(indices_in_subnode)
            # Update task indices if already in global indices
            task_indices['test'] = task_indices['test'].union(global_indices['test'].intersection(indices_in_subnode))
            task_indices['val'] = task_indices['val'].union(global_indices['val'].intersection(indices_in_subnode))
            task_indices['train'] = task_indices['train'].union(global_indices['train'].intersection(indices_in_subnode))
            # Sorted to make sure it has always the same order
            documents_not_seen = sorted(list(indices_in_subnode.difference(task_indices['test']).difference(task_indices['val']).difference(task_indices['train'])))
            if len(documents_not_seen) == 0:
                continue
            random.seed(cfg['seed'])
            random.shuffle(documents_not_seen)
            # Train 70% - validation 15% - test 15%
            # Make sure the sets are mutually exclusive
            train_idx = set(documents_not_seen[:int(0.7*len(documents_not_seen))])
            val_idx = set(documents_not_seen[int(0.7*len(documents_not_seen)):int(0.85*len(documents_not_seen))])
            test_idx = set(documents_not_seen[int(0.85*len(documents_not_seen)):])
            # Update task and global indices
            task_indices['train'] = task_indices['train'].union(train_idx)
            global_indices['train'] = global_indices['train'].union(train_idx)
            task_indices['val'] = task_indices['val'].union(val_idx)
            global_indices['val'] = global_indices['val'].union(val_idx)
            task_indices['test'] = task_indices['test'].union(test_idx)
            global_indices['test'] = global_indices['test'].union(test_idx)

        # Make sure the indices are mutually exclusive within the task
        assert (not task_indices['train'].intersection(task_indices['val'])) and (not task_indices['train'].intersection(task_indices['test'])) and (not task_indices['val'].intersection(task_indices['test'])), f"Tasks sets are not mutually exclusive"
        assert (not global_indices['train'].intersection(global_indices['val'])) and (not global_indices['train'].intersection(global_indices['test'])) and (not global_indices['val'].intersection(global_indices['test'])), f"Global sets are not mutually exclusive"
        # Register membership of docs to this subtree/task (for later correction)
        for _doc in all_subtree_indices:
            doc_to_subtrees[_doc].add(subtree_idx)
        # Make sure all samples were taken
        assert len(task_indices['train']) + len(task_indices['val']) + len(task_indices['test']) == len(all_subtree_indices)

        # Get indices of this task
        for split, set_indices in task_indices.items():
            all_tasks_indices[split].append(sorted(list(set_indices)))
        relevant_labels_task = torch.tensor(relevant_labels_task, dtype=torch.int64)
        all_tasks_relevant_labels.append(relevant_labels_task)

        # Number of elements in each task
        cfg['tasks_size'].append(len(all_subtree_indices))

    # Coverage snapshot BEFORE correction
    global_indices_before = {
        'train': sorted(list(global_indices['train'])),
        'val'  : sorted(list(global_indices['val'])),
        'test' : sorted(list(global_indices['test'])),
    }
    print("")
    print(f">> Unique documents: {len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test'])}")
    print(f">> Training set: {len(global_indices['train'])}, validation set: {len(global_indices['val'])}, test set: {len(global_indices['test'])}")

    # Decide where to save stats
    stats_dir = cfg['paths'].get('dataset_stats', cfg['paths']['dataset'] / 'dataset_stats')
    stats_dir.mkdir(parents=True, exist_ok=True)

    coverage = _compute_label_split_coverage(taxonomy, items_per_label, global_indices)
    # Coverage correction using holdout pool ---
    task_root_names = set(subtrees_to_keep)
    # For each label that misses a split, draw from its holdout pool (if available)
    for lab, counts in coverage.items():
        if lab == 'root' or lab in task_root_names:
            continue
        # For each missing split, try to add holdout docs for this label
        for split in ('train','val','test'):
            if counts[split] > 0:
                continue
            ho_candidates = holdout_per_label.get(lab, set())
            if not ho_candidates:
                continue  # nothing to add

            # We can only add documents that *actually* contain this label in our dataset
            # (ho_candidates already satisfy that), and they are currently in no split.
            # Choose as many as needed (typically at least 1).
            # We aim to add up to ceil(rare_holdout_ratio * freq) but 1 is generally enough to fix coverage.
            need = 1
            pick = []
            for d in sorted(ho_candidates):
                # ensure this doc isn't already assigned elsewhere
                if (d not in global_indices['train']) and (d not in global_indices['val']) and (d not in global_indices['test']):
                    pick.append(d)
                    if len(pick) >= need:
                        break
            if not pick:
                continue

            # Assign picked docs to the missing split globally
            global_indices[split].update(pick)

            # Also add to every task (subtree) that contains these docs
            # NOTE: all_tasks_indices[split] is a list aligned by subtree_idx
            for d in pick:
                for subtree_idx in doc_to_subtrees.get(d, []):
                    # Append to that subtree's split list
                    # (convert inner list set-like by appending then we'll sort later)
                    all_tasks_indices[split][subtree_idx].append(d)

            # Consume from holdout so we don't reuse
            holdout_per_label[lab].difference_update(pick)

            # Update local coverage accounting so we don't try to "fix" again
            counts[split] += len(pick)
    
    # Re-sort and sanity checks after correction ---
    for split in ('train','val','test'):
        global_indices[split] = sorted(list(set(global_indices[split]))) # type: ignore

    # Ensure tasks lists remain sorted and unique
    for split in ('train','val','test'):
        for t in range(len(all_tasks_indices[split])):
            all_tasks_indices[split][t] = sorted(list(set(all_tasks_indices[split][t])))

    # Mutual exclusivity (global)
    assert (not set(global_indices['train']).intersection(global_indices['val'])) \
        and (not set(global_indices['train']).intersection(global_indices['test'])) \
        and (not set(global_indices['val']).intersection(global_indices['test'])), "Sets are not mutually exclusive after correction"
   
    
    # Reintegrate any unused holdout documents so nothing is lost ---
    # 1) Gather all holdout docs and find the ones still unassigned to any split
    all_holdout_docs = set().union(*[s for s in holdout_per_label.values()]) if holdout_per_label else set()
    already_assigned = set(global_indices['train']) | set(global_indices['val']) | set(global_indices['test'])
    leftovers = sorted(list(all_holdout_docs - already_assigned))

    if leftovers:
        # 2) Deterministic allocation of leftovers in 70/15/15 (train/val/test)
        random.seed(cfg['seed'])
        random.shuffle(leftovers)

        n = len(leftovers)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        # rest to test
        n_test  = n - n_train - n_val

        assign = {
            'train': leftovers[:n_train],
            'val'  : leftovers[n_train:n_train+n_val],
            'test' : leftovers[n_train+n_val:]
        }

        # 3) Add to global splits and to the corresponding tasks (using doc_to_subtrees)
        for split, docs in assign.items():
            if not docs:
                continue
            global_indices[split].extend(docs) # type: ignore
            for d in docs:
                for subtree_idx in doc_to_subtrees.get(d, []):
                    all_tasks_indices[split][subtree_idx].append(d)

        # 4) Re-sort & uniqueness again + exclusivity checks
        for split in ('train','val','test'):
            global_indices[split] = sorted(list(set(global_indices[split]))) # type: ignore
            for t in range(len(all_tasks_indices[split])):
                all_tasks_indices[split][t] = sorted(list(set(all_tasks_indices[split][t])))

        assert (not set(global_indices['train']).intersection(global_indices['val'])) \
            and (not set(global_indices['train']).intersection(global_indices['test'])) \
            and (not set(global_indices['val']).intersection(global_indices['test'])), \
            "Sets are not mutually exclusive after reintegrating leftovers"
    
     
    print(f">> Unique documents: {len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test'])}")
    print(f">> Training set: {len(global_indices['train'])}, validation set: {len(global_indices['val'])}, test set: {len(global_indices['test'])}")

    
    
    print(f"> Constructing global sets...")
    global_relevant_labels = torch.tensor(list(global_relevant_labels), dtype=torch.int64)
    # Save global indices, sorted so it is deterministic
    for split in global_indices.keys():
        global_indices[split] = sorted(list(global_indices[split])) # type: ignore
    
    assert len(list(cfg['paths']['data'].iterdir())) == len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test']), "Why mismatch?"
    
    return labels_per_idx, global_indices, global_relevant_labels, all_tasks_indices, all_tasks_relevant_labels




def data_split_hf(documents, labels_data, cfg):
    """Data Splitting and Pruning Function

    Processes and splits the dataset into training, validation, and test splits, at a global level and for each sub-task.

    This function performs several operations:
      - Computes basic statistics about the input dataset.
      - Prunes the taxonomy tree to remove under-represented or irrelevant nodes.
      - Filters out documents that no longer correspond to the pruned taxonomy.
      - Updates configuration (`cfg`) with vocabulary and taxonomy data for downstream modules.
      - Constructs global and per-task data splits.
      - Saves processed documents and label information as JSON files.

    Args:
        documents (torch.tensor): A tensor containing the tokenized dataset documents.
        labels_data (list of list of str): A list where each element is a list of labels associated with a document.
        cfg (dict): Configuration dictionary

    Returns:
        tuple: A tuple containing:
            - global_indices (dict): Dictionary with keys `'train'`, `'val'`, and `'test'`, each mapping to a sorted list of document indices for that split.
            - global_relevant_labels (torch.Tensor): Tensor containing all label indices used globally in the pruned taxonomy (excluding main root and roots of tasks).
            - all_tasks_indices (dict): Dictionary with keys `'train'`, `'val'`, and `'test'`, each mapping to a list of lists where each sub-list contains indices of documents belonging to a particular task.
            - all_tasks_relevant_labels (list of torch.Tensor): List of tensors where each tensor contains the relevant label indices for a given task.

    Notes:
        - Taxonomy pruning is guided by a `min_freq` threshold.
        - Tasks and labels are saved in a deterministic order for reproducibility.
        - The function modifies `cfg` in-place with vocabulary and task-related metadata for downstream use.
    """
    if "seed" not in cfg.keys():
        cfg['seed'] = 42
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    print(f"> Using seed {cfg['seed']} for random operations")    
    print(f"> Get some stats on dataset")
    print(f">> Dataset has {len(documents)} documents")
    taxonomy = cfg['taxonomy']
    # key: label, values: list of items index having this label
    items_per_label = {}
    # key: index, values: list of labels 
    labels_per_idx = {}
    # Get all labels from all items
    for idx, labels in enumerate(labels_data):
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
            labels_per_idx[idx] = labels
    print(f">> Documents per label: {np.mean([len(items) for items in items_per_label.values()])}")
    print(f">> Labels per document: {np.mean([len(labels) for labels in labels_per_idx.values()])}")

    # Recursively remove leaves in the taxonomy that do not appear in the dataset (otherwise we could have a KeyError afterwards)
    print(f"> Removing leaves not appearing in dataset...")
    
    
    leaves_to_remove = [leaf for leaf in taxonomy.leaves if leaf not in [node for node in items_per_label.keys() if node not in taxonomy.label_to_children.keys() or taxonomy.is_leaf(node)]]
    while(len(leaves_to_remove) != 0):
        taxonomy.remove_leaves(leaves_to_remove)
        leaves_to_remove = [leaf for leaf in taxonomy.leaves if leaf not in [node for node in items_per_label.keys() if node not in taxonomy.label_to_children.keys() or taxonomy.is_leaf(node)]]

    # Prune all leaves until they all have at least min_freq samples
    if "min_freq" in cfg.keys() and cfg["min_freq"] is not None:
        min_freq = cfg["min_freq"]
    else:
        min_freq = 50
    n_before = len(taxonomy.all_children('root'))
    while(min([len(items_per_label[leaf]) for leaf in taxonomy.leaves])) < min_freq:
        for leaf in taxonomy.leaves:
            if len(items_per_label[leaf]) < min_freq:
                taxonomy.remove_leaves(leaf)
                del items_per_label[leaf]

    # Remove leaves on level 1
    level = 1
    taxonomy.remove_leaves([node for node in taxonomy.level_to_labels[level] if taxonomy.is_leaf(node)])
    all_labels = set(taxonomy.all_children('root'))
    
    print(f"> Pruned {n_before - len(all_labels)} leaf nodes having less than {min_freq} samples and the leaves on first level")
    print(f">> Updated taxonomy has {taxonomy.n_nodes} nodes and {len(taxonomy.leaves)} leaves and a height of {taxonomy.height}")

    # Take sub-trees only from level 1, if they are not leaves
    # Each sub-tree will be considered as a task
    subtrees_to_keep = [node for node in taxonomy.level_to_labels[level] if node not in taxonomy.leaves]
    # Remove sub-trees that have only one leaf, i.e. it is a straight line from root to leaf
    # subtrees_to_keep = [subtree_root for subtree_root in subtrees_to_keep if len([node for node in taxonomy.all_children(subtree_root) if node in taxonomy.leaves]) > 1]
    # Sorted to make sure sub-trees have always the same order
    subtrees_to_keep = sorted(subtrees_to_keep)
    subtrees_to_keep_set = set(subtrees_to_keep)
    print(f"> Possible sub-trees {len(subtrees_to_keep)} with root on level {level}")

    # Keep samples that have at least one label
    print(f"> Remove documents having no more label in common with taxonomy")
    indices_to_keep = []
    for idx, labels in labels_per_idx.items():
        doc_labels = [lab for lab in labels if lab in all_labels]
        # If no labels in common with taxonomy
        if len(doc_labels) == 0:
            continue
        # If single label is a root of a sub-tree
        elif (len(doc_labels) == 1) and (doc_labels[-1] in subtrees_to_keep_set):
            continue
        else:
            indices_to_keep.append(idx)
    indices_to_keep = torch.tensor(indices_to_keep, dtype=torch.int64)

    # Update data: remove documents that have no more labels
    # Update labels: remove labels that are not in taxonomy anymore
    n_docs_before = len(documents)
    new_documents = []
    new_labels = []
    for idx in indices_to_keep:
        # Sorted to always keep the same order
        new_documents.append(documents[idx])
        new_labels.append(sorted(all_labels & set(labels_data[idx])))
    print(f">> Documents kept: {len(new_documents)} (out of {n_docs_before})")
    assert len(new_documents) == len(new_labels), f"Error in documents removing"

    # Update the data-structures after pruning and cleaning
    # key: label, values: set of items index having this label
    print(f"> Updating storage and stats...")
    items_per_label = {}
    # key: index, values: set of labels
    labels_per_idx = {} 
    # Get all labels from all items
    for idx, labels in enumerate(new_labels):
        labels_per_idx[idx] = labels
        # Each single label for an item
        for label in labels:
            if label in items_per_label:
                items_per_label[label].append(idx)
            else:
                items_per_label[label] = [idx]
            
    print(f">> Now {np.mean([len(items) for items in items_per_label.values()])} documents per label")
    print(f">> Now {np.mean([len(labels) for labels in labels_per_idx.values()])} labels per document")

    # Sort labels by increasing frequency (rarer first), keep only *current* leaf/inner labels that remain in taxonomy
    labels_sorted_by_freq = sorted(
        [lab for lab in items_per_label.keys() if lab in taxonomy.label_to_idx], 
        key=lambda l: len(items_per_label[l])
    )
    n_labels = len(labels_sorted_by_freq)
    n_rare = max(1, int(round(cfg['split']['rare_label_fraction'] * n_labels)))
    rare_labels = set(labels_sorted_by_freq[:n_rare])

    # Build per-label holdout pool: take 10% of documents of each rare label (deterministic)
    random.seed(cfg['seed'])
    holdout_per_label = {}
    for lab in rare_labels:
        idxs = sorted(set(items_per_label[lab]))
        k = max(1, int(round(cfg['split']['rare_holdout_ratio'] * len(idxs)))) if len(idxs) > 0 else 0
        random.shuffle(idxs)
        holdout = set(idxs[:k]) if k > 0 else set()
        holdout_per_label[lab] = holdout

    # Remove holdout docs from items_per_label for initial splitting (we'll re-inject later if needed)
    for lab, ho in holdout_per_label.items():
        if ho:
            items_per_label[lab] = [i for i in items_per_label[lab] if i not in ho]
    
    # Construct specific data for hector and tamlec
    # Target vocab, deepcopy to have another reference in memory
    voc_plus_pad = copy.deepcopy(taxonomy.label_to_idx)
    voc_plus_pad['<pad>'] = len(taxonomy.label_to_idx)
    cfg['tamlec_params']['trg_vocab'] = tvoc.vocab(voc_plus_pad, min_freq=0)
    # Taxonomy for hector is full taxonomy
    taxo_id_root = taxonomy.label_to_idx['root']
    children = {taxo_id_root: [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children['root']]}
    for curr_node in taxonomy.all_children('root'):
        if not taxonomy.is_leaf(curr_node):
            children[taxonomy.label_to_idx[curr_node]] = [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[curr_node]]
    cfg['tamlec_params']['taxos_hector'] = [(taxo_id_root, children)]
    # Taxonomies for tamlec are taxonomies of selected sub-trees
    cfg['tamlec_params']['taxos_tamlec'] = []
    for subtree_root in subtrees_to_keep:
        taxo_id_root = taxonomy.label_to_idx[subtree_root]
        children = {taxo_id_root: [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[subtree_root]]}
        for curr_node in taxonomy.all_children(subtree_root):
            if not taxonomy.is_leaf(curr_node):
                children[taxonomy.label_to_idx[curr_node]] = [taxonomy.label_to_idx[node] for node in taxonomy.label_to_children[curr_node]]
        cfg['tamlec_params']['taxos_tamlec'].append((taxo_id_root, children))
    # Abstract dict
    cfg['tamlec_params']['abstract_dict'] = {}
    for label, abstract in taxonomy.label_to_abstract.items():
        cfg['tamlec_params']['abstract_dict'][taxonomy.label_to_idx[label]] = abstract

    print(f">> Constructing the labels...")
    final_labels = []
    for labels in new_labels:
        # Add root of taxonomy
        class_list = [0] + [taxonomy.label_to_idx[label] for label in labels]
        final_labels.append(class_list)

    # Various data
    cfg['tasks_size'] = []
    cfg['task_to_subroot'] = {}
    # Root belongs to no task
    cfg['label_to_tasks'] = {taxonomy.label_to_idx['root']: set()}
    # Dict of document indices per split
    global_indices = {'train': set(), 'val': set(), 'test': set()}
    # Dict of lists (by split) of lists (by tasks) of indices (integer)
    all_tasks_indices = {'train': [], 'val': [], 'test': []}
    # List of lists of relevant labels for each task
    all_tasks_relevant_labels = []
    # Set of all possible labels (root and sub-roots excluded)
    global_relevant_labels = set()
    
    for key in labels_per_idx.keys():
        doc_labels = []
        for lab in labels_per_idx[key]:
            doc_labels.append(taxonomy.label_to_idx[lab])
        labels_per_idx[key] = doc_labels
    
    print(f">> Train-val-test split...")
    doc_to_subtrees = defaultdict(set)
    for subtree_idx, subtree_root in enumerate(tqdm(subtrees_to_keep, leave=False)):
        
        # Dict of indices for this task
        task_indices = {'train': set(), 'val': set(), 'test': set()}
        all_subtree_indices = set()
        relevant_labels_task = []
        # Set idx of the node that is the root of this sub-tree
        cfg['task_to_subroot'][subtree_idx] = taxonomy.label_to_idx[subtree_root]
        try:
            cfg['label_to_tasks'][taxonomy.label_to_idx[subtree_root]].add(subtree_idx)
        except KeyError:
            cfg['label_to_tasks'][taxonomy.label_to_idx[subtree_root]] = set([subtree_idx])
            
        # Make the split for all other nodes in the sub-tree
        # process labels in rarity order (rarest first)
        subtree_nodes = set(taxonomy.all_children(subtree_root))
        nodes_in_order = [lab for lab in labels_sorted_by_freq if lab in subtree_nodes]
        for subnode in nodes_in_order:
            # Subnode to the task
            try:
                cfg['label_to_tasks'][taxonomy.label_to_idx[subnode]].add(subtree_idx)
            except KeyError:
                cfg['label_to_tasks'][taxonomy.label_to_idx[subnode]] = set([subtree_idx])
            # Add node to the relevant indices of this task and to the global
            relevant_labels_task.append(taxonomy.label_to_idx[subnode])
            global_relevant_labels.add(taxonomy.label_to_idx[subnode])
            # exclude holdout docs for this label (if any) from the initial splits
            indices_in_subnode = set(items_per_label.get(subnode, []))
            # Complete document indices of this task
            all_subtree_indices = all_subtree_indices.union(indices_in_subnode)
            # Update task indices if already in global indices
            task_indices['test'] = task_indices['test'].union(global_indices['test'].intersection(indices_in_subnode))
            task_indices['val'] = task_indices['val'].union(global_indices['val'].intersection(indices_in_subnode))
            task_indices['train'] = task_indices['train'].union(global_indices['train'].intersection(indices_in_subnode))
            # Sorted to make sure it has always the same order
            documents_not_seen = sorted(list(indices_in_subnode.difference(task_indices['test']).difference(task_indices['val']).difference(task_indices['train'])))
            if len(documents_not_seen) == 0:
                continue
            random.seed(cfg['seed'])
            random.shuffle(documents_not_seen)
            # Train 70% - validation 15% - test 15%
            # Make sure the sets are mutually exclusive
            train_idx = set(documents_not_seen[:int(0.7*len(documents_not_seen))])
            val_idx = set(documents_not_seen[int(0.7*len(documents_not_seen)):int(0.85*len(documents_not_seen))])
            test_idx = set(documents_not_seen[int(0.85*len(documents_not_seen)):])
            # Update task and global indices
            task_indices['train'] = task_indices['train'].union(train_idx)
            global_indices['train'] = global_indices['train'].union(train_idx)
            task_indices['val'] = task_indices['val'].union(val_idx)
            global_indices['val'] = global_indices['val'].union(val_idx)
            task_indices['test'] = task_indices['test'].union(test_idx)
            global_indices['test'] = global_indices['test'].union(test_idx)

        # Make sure the indices are mutually exclusive within the task
        assert (not task_indices['train'].intersection(task_indices['val'])) and (not task_indices['train'].intersection(task_indices['test'])) and (not task_indices['val'].intersection(task_indices['test'])), f"Tasks sets are not mutually exclusive"
        assert (not global_indices['train'].intersection(global_indices['val'])) and (not global_indices['train'].intersection(global_indices['test'])) and (not global_indices['val'].intersection(global_indices['test'])), f"Global sets are not mutually exclusive"
        # Register membership of docs to this subtree/task (for later correction)
        for _doc in all_subtree_indices:
            doc_to_subtrees[_doc].add(subtree_idx)
        # Make sure all samples were taken
        assert len(task_indices['train']) + len(task_indices['val']) + len(task_indices['test']) == len(all_subtree_indices)

        # Get indices of this task
        for split, set_indices in task_indices.items():
            all_tasks_indices[split].append(sorted(list(set_indices)))
        relevant_labels_task = torch.tensor(relevant_labels_task, dtype=torch.int64)
        all_tasks_relevant_labels.append(relevant_labels_task)

        # Number of elements in each task
        cfg['tasks_size'].append(len(all_subtree_indices))

    # Coverage snapshot BEFORE correction
    global_indices_before = {
        'train': sorted(list(global_indices['train'])),
        'val'  : sorted(list(global_indices['val'])),
        'test' : sorted(list(global_indices['test'])),
    }
    print("")
    print(f">> Unique documents: {len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test'])}")
    print(f">> Training set: {len(global_indices['train'])}, validation set: {len(global_indices['val'])}, test set: {len(global_indices['test'])}")
    cfg['n_global_train'] = len(global_indices['train'])
    cfg['n_global_val'] = len(global_indices['val'])
    cfg['n_global_test'] = len(global_indices['test'])
    # Decide where to save stats
    stats_dir = cfg['paths'].get('dataset_stats', cfg['paths']['dataset'] / 'dataset_stats')
    stats_dir.mkdir(parents=True, exist_ok=True)

    coverage = _compute_label_split_coverage(taxonomy, items_per_label, global_indices)
    # Coverage correction using holdout pool ---
    task_root_names = set(subtrees_to_keep)
    # For each label that misses a split, draw from its holdout pool (if available)
    for lab, counts in coverage.items():
        if lab == 'root' or lab in task_root_names:
            continue
        # For each missing split, try to add holdout docs for this label
        for split in ('train','val','test'):
            if counts[split] > 0:
                continue
            ho_candidates = holdout_per_label.get(lab, set())
            if not ho_candidates:
                continue  # nothing to add

            # We can only add documents that *actually* contain this label in our dataset
            # (ho_candidates already satisfy that), and they are currently in no split.
            # Choose as many as needed (typically at least 1).
            # We aim to add up to ceil(rare_holdout_ratio * freq) but 1 is generally enough to fix coverage.
            need = 1
            pick = []
            for d in sorted(ho_candidates):
                # ensure this doc isn't already assigned elsewhere
                if (d not in global_indices['train']) and (d not in global_indices['val']) and (d not in global_indices['test']):
                    pick.append(d)
                    if len(pick) >= need:
                        break
            if not pick:
                continue

            # Assign picked docs to the missing split globally
            global_indices[split].update(pick)

            # Also add to every task (subtree) that contains these docs
            # NOTE: all_tasks_indices[split] is a list aligned by subtree_idx
            for d in pick:
                for subtree_idx in doc_to_subtrees.get(d, []):
                    # Append to that subtree's split list
                    # (convert inner list set-like by appending then we'll sort later)
                    all_tasks_indices[split][subtree_idx].append(d)

            # Consume from holdout so we don't reuse
            holdout_per_label[lab].difference_update(pick)

            # Update local coverage accounting so we don't try to "fix" again
            counts[split] += len(pick)
    
    # Re-sort and sanity checks after correction ---
    for split in ('train','val','test'):
        global_indices[split] = sorted(list(set(global_indices[split]))) # type: ignore

    # Ensure tasks lists remain sorted and unique
    for split in ('train','val','test'):
        for t in range(len(all_tasks_indices[split])):
            all_tasks_indices[split][t] = sorted(list(set(all_tasks_indices[split][t])))

    # Mutual exclusivity (global)
    assert (not set(global_indices['train']).intersection(global_indices['val'])) \
        and (not set(global_indices['train']).intersection(global_indices['test'])) \
        and (not set(global_indices['val']).intersection(global_indices['test'])), "Sets are not mutually exclusive after correction"
   
    
    # Reintegrate any unused holdout documents so nothing is lost ---
    # 1) Gather all holdout docs and find the ones still unassigned to any split
    all_holdout_docs = set().union(*[s for s in holdout_per_label.values()]) if holdout_per_label else set()
    already_assigned = set(global_indices['train']) | set(global_indices['val']) | set(global_indices['test'])
    leftovers = sorted(list(all_holdout_docs - already_assigned))

    if leftovers:
        # 2) Deterministic allocation of leftovers in 70/15/15 (train/val/test)
        random.seed(cfg['seed'])
        random.shuffle(leftovers)

        n = len(leftovers)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        # rest to test
        n_test  = n - n_train - n_val

        assign = {
            'train': leftovers[:n_train],
            'val'  : leftovers[n_train:n_train+n_val],
            'test' : leftovers[n_train+n_val:]
        }

        # 3) Add to global splits and to the corresponding tasks (using doc_to_subtrees)
        for split, docs in assign.items():
            if not docs:
                continue
            global_indices[split].extend(docs) # type: ignore
            for d in docs:
                for subtree_idx in doc_to_subtrees.get(d, []):
                    all_tasks_indices[split][subtree_idx].append(d)

        # 4) Re-sort & uniqueness again + exclusivity checks
        for split in ('train','val','test'):
            global_indices[split] = sorted(list(set(global_indices[split]))) # type: ignore
            for t in range(len(all_tasks_indices[split])):
                all_tasks_indices[split][t] = sorted(list(set(all_tasks_indices[split][t])))

        assert (not set(global_indices['train']).intersection(global_indices['val'])) \
            and (not set(global_indices['train']).intersection(global_indices['test'])) \
            and (not set(global_indices['val']).intersection(global_indices['test'])), \
            "Sets are not mutually exclusive after reintegrating leftovers"
    
     
    print(f">> Unique documents: {len(global_indices['train']) + len(global_indices['val']) + len(global_indices['test'])}")
    print(f">> Training set: {len(global_indices['train'])}, validation set: {len(global_indices['val'])}, test set: {len(global_indices['test'])}")

    
    print(f"> Constructing global sets...")
    global_relevant_labels = torch.tensor(list(global_relevant_labels), dtype=torch.int64)
    # Save global indices, sorted so it is deterministic
    for split in global_indices.keys():
        global_indices[split] = sorted(list(global_indices[split])) # type: ignore

    # Save the tokenized documents and their labels into separate files
    prepare_data_dexa(global_indices, taxonomy, new_documents, final_labels, cfg['paths']['dataset']/f'preprocessed_data/seed_{cfg["seed"]}', cfg)
    
def prepare_data_dexa(global_datasets, taxo, new_documents, labels, dexa_data_path, cfg):
    #global_datasets = torch.load(global_datasets_path)
    #taxo = self.taxonomy
    taxo_lbl_idxs = sorted(taxo.idx_to_label.keys())
    # The idea is to:
    # 
    # 1) load training tokenized documents in trnX
    #    build the XY_train.txt file where for each line there's the list of labels
    # 2) Use idx_to_label dictionary to, for each key, find the label text in the ontology,
    #    tokenize it and put it in lbl_input_ids.npy
    #
    # Tokenization should create 32 long arrays of numbers padded with zeros and attention_masks should
    # be arrays of ones and zeros indicating which are the non_padding tokens with ones.
    
    train_rows_pos = []  # list of list[int], compact label ids
    trn_input_ids = []
    trn_attention_mask = []
    with open(dexa_data_path / "trn_X_Y.txt", "w") as x_y_f:
        x_y_f.write(f"{len(global_datasets['train'])} {len(taxo_lbl_idxs) - 1 }\n")
        for idx in global_datasets['train']:
            doc_tokens, doc_labels = new_documents[idx], labels[idx]    
            trn_input_ids.append(
                [ doc_tokens[i] if i<len(doc_tokens) else 0 for i in range(32) ]
            )
            trn_attention_mask.append(
                [ 1 if i<len(doc_tokens) else 0 for i in range(32)]    
            )
            pos = []  
            x_y_line = ""
            for lab in sorted(doc_labels):
                if lab == 0: continue
                compact = lab - 1
                pos.append(compact)
                x_y_line += str(lab - 1) + ":1.0 "
            train_rows_pos.append(pos)
            x_y_f.write(x_y_line + '\n')
    test_rows_pos = []    
    tst_input_ids = []
    tst_attention_mask = []
    with open(dexa_data_path / "tst_X_Y.txt", "w") as x_y_f:
        x_y_f.write(f"{len(global_datasets['test'])} {len(taxo_lbl_idxs) - 1}\n")
        for idx in global_datasets['test']:
            doc_tokens, doc_labels = new_documents[idx], labels[idx]    
            tst_input_ids.append(
                [ doc_tokens[i] if i<len(doc_tokens) else 0 for i in range(32) ]
            )
            tst_attention_mask.append(
                [ 1 if i<len(doc_tokens) else 0 for i in range(32)]    
            )
            pos = []
            
            x_y_line = ""
            for lab in sorted(doc_labels):
                if lab == 0: continue
                compact = lab - 1
                pos.append(compact)
                x_y_line += str(lab - 1) + ":1.0 "
            test_rows_pos.append(pos)
            x_y_f.write(x_y_line + '\n')
            
    
    # Iterate over the keys of taxo.idx_to_label in increasing order and use the idx_to_title to get
    # tokenize the label descriptions and tokenize them, then save them in lbl_input_ids and lbl_attention_mask
    
    lbl_txts = []
    for lbl_id in taxo_lbl_idxs:
        if lbl_id == 0: continue
        label = taxo.idx_to_label[lbl_id]
        label_txt = taxo.label_to_title[label]
        lbl_txts.append(label_txt)
    lbl_input_ids, _, _, lbl_attention_masks = tokenization_hf(lbl_txts, cfg)
    np.save(dexa_data_path/"trn_doc_input_ids.npy", trn_input_ids)
    np.save(dexa_data_path/"trn_doc_attention_mask.npy", trn_attention_mask)
    
    
    np.save(dexa_data_path/"tst_doc_input_ids.npy", tst_input_ids)
    np.save(dexa_data_path/"tst_doc_attention_mask.npy", tst_attention_mask)
    
    
    np.save(dexa_data_path/"lbl_input_ids.npy",lbl_input_ids )
    np.save(dexa_data_path/"lbl_attention_mask.npy",lbl_attention_masks )
    
    _write_filter_pairs(dexa_data_path / "filter_labels_train.txt", train_rows_pos)
    _write_filter_pairs(dexa_data_path / "filter_labels_test.txt",  test_rows_pos)

    n_doc_train , n_doc_test = len(global_datasets['train']), len(global_datasets['test'])
    return n_doc_train, n_doc_test


def _write_filter_pairs(out_path, rows_labels):
        """
        rows_labels: list where rows_labels[i] is a list of compact label ids (0..L-1)
        Writes lines: "<row_id> <label_id>"
        """
        with open(out_path, "w") as f:
            f.write("")