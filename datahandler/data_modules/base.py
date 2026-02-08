from torch.utils.data import DataLoader
import json
import torch
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datahandler.datasets.global_datasets import GlobalDataset
from datahandler.datasets.tasks_datasets import TasksDataset
from datahandler.samplers.tasks_sampler import SubtreeSampler
from datahandler.samplers.global_sampler import collate_global_int_labels, collate_global_int_labels_hector_tamlec
from datahandler.taxonomy import Taxonomy
from datahandler.tree import Tree
from datahandler.embeddings import EmbeddingHandler
from datahandler.utils import  preprocess_docs_and_taxonomy,\
                                    text_preprocessing, tokenization,\
                                    complete_labels,\
                                    data_split,\
                                    get_split_datastructures
from misc import utils
import tqdm

_PARENTS = None
_DATASET = None

def _init_worker(parents, dataset):
    global _PARENTS, _DATASET
    _PARENTS = parents
    _DATASET = dataset

def _worker(lines):
    parents = _PARENTS
    dataset = _DATASET
    is_topics = 'topics' in dataset
    is_concepts = 'concepts' in dataset

    docs, labs = [], []
    for line in lines:
        d = json.loads(line)

        tp = d.get('text_processed')
        if tp is not None:
            raw, simple = tp, True
        else:
            raw, simple = (d.get("title") or "") + " " + (d.get("abstract") or ""), False

        text = text_preprocessing(raw, simple=simple)

        if is_topics:
            y = list(d['topics_labels'])
        elif is_concepts:
            y = list(d['concepts_labels'])
        else:
            y = list(d['label'])

        y = complete_labels(y, parents)
        if y:
            docs.append(text)
            labs.append(y)

    return docs, labs

def _chunks(f, n):
    buf = []
    for line in f:
        buf.append(line)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

class DataModuleBase:
    
    def __init__(self, cfg:dict, one_hot_labels, verbose=True):
        """
        Initialize the base data module and ensure preprocessing artifacts exist.

        This loads or creates taxonomy, embeddings, dataset splits, and builds
        task/global datasets along with their samplers.

        :param cfg: Experiment configuration dictionary.
        :param one_hot_labels: Whether labels are represented as one-hot vectors.
        :param verbose: If True, print preprocessing/loading messages.
        :return: None.
        """
        self.verbose = verbose
        # Add sensible default values for splitting parameters
        cfg.setdefault('split', {})
        cfg['split'].setdefault('rare_label_fraction', 0.10)   # bottom 10% of labels considered "rarest"
        cfg['split'].setdefault('rare_holdout_ratio', 0.10)    # hold out 10% of docs per rare label

        #Check if data is already pre-processed
        condition1 = cfg['paths']['taxos_hector'].is_file()
        condition2 = cfg['paths']['taxos_tamlec'].is_file()
        condition3 = cfg['paths']['data'].is_dir()
        condition_split = not (cfg['paths']['preprocessed_data'] / f"seed_{cfg['seed']}" ).is_dir()
        self.cfg = cfg
        
        self.one_hot_labels = one_hot_labels
        
        # if data not already pre-processed, preprocess else load it
        if not (condition1 and condition2 and condition3):
            self.preprocess()
        elif condition_split:
            
            self.preprocess(split_only = True)
            
        else:
            if self.verbose: print(f"> Found pre-processed data, load it...")
            self.cfg['paths_per_doc'] = self.cfg['paths']['paths_per_doc']
            with open(self.cfg['paths']['global_datasets'], 'rb') as f:
                self.global_indices, self.global_relevant_labels = torch.load(f)
            with open(self.cfg['paths']['tasks_datasets'], 'rb') as f:
                self.tasks_indices, self.tasks_relevant_labels = torch.load(f)
            with open(self.cfg['paths']['taxonomy'], 'rb') as f:
                self.cfg['taxonomy'] = torch.load(f)
            with open(self.cfg['paths']['embeddings'], 'rb') as f:
                self.embeddings = torch.load(f)
            with open(self.cfg['paths']['task_to_subroot'], 'rb') as f:
                self.cfg['task_to_subroot'] = torch.load(f)
            with open(self.cfg['paths']['tasks_size'], 'rb') as f:
                self.cfg['tasks_size'] = torch.load(f)
            with open(self.cfg['paths']['label_to_tasks'], 'rb') as f:
                self.cfg['label_to_tasks'] = torch.load(f)
        # Get the tasks datasets and dataloader
        self.tasks_train_set = TasksDataset(self.tasks_indices['train'], self.tasks_relevant_labels, one_hot_labels, cfg)
        self.tasks_val_set = TasksDataset(self.tasks_indices['val'], self.tasks_relevant_labels, one_hot_labels, cfg)
        self.tasks_test_set = TasksDataset(self.tasks_indices['test'], self.tasks_relevant_labels, one_hot_labels, cfg)

        # Construct samplers
        self.train_sampler = SubtreeSampler(self.tasks_train_set, cfg=cfg, batch_size=cfg['batch_size_train'])
        self.val_sampler = SubtreeSampler(self.tasks_val_set, cfg=cfg, batch_size=cfg['batch_size_eval'])
        self.test_sampler = SubtreeSampler(self.tasks_test_set, cfg=cfg, batch_size=cfg['batch_size_eval'])

    def preprocess(self, split_only = False):
        '''
        Preprocess raw data or run split-only preprocessing.

        When ``split_only`` is False, this loads the taxonomy, processes raw
        documents, tokenizes, builds embeddings, and saves artifacts. When
        True, it reuses cached artifacts and only recomputes data splits.

        :param split_only: If True, reuse cached artifacts and only split data.
        :return: None.
        '''
        (self.cfg['paths']['preprocessed_data'] / f"seed_{self.cfg['seed']}").mkdir(exist_ok = True)
        if split_only:
            # Use the pruned taxonomy and saved artifacts from the initial preprocessing
            if self.cfg['paths']['taxonomy'].is_file():
                with open(self.cfg['paths']['taxonomy'], 'rb') as f:
                    self.cfg['taxonomy'] = torch.load(f)
            else:
                self.cfg['taxonomy'] = Taxonomy()
                self.cfg['taxonomy'].load_taxonomy(self.cfg)

            # Load stored data needed by downstream steps (paths, embeddings, etc.)
            if self.cfg['paths']['embeddings'].is_file():
                with open(self.cfg['paths']['embeddings'], 'rb') as f:
                    self.embeddings = torch.load(f)
            if self.cfg['paths']['src_vocab'].is_file():
                with open(self.cfg['paths']['src_vocab'], 'rb') as f:
                    self.cfg['tamlec_params']['src_vocab'] = torch.load(f)
            if self.cfg['paths']['trg_vocab'].is_file():
                with open(self.cfg['paths']['trg_vocab'], 'rb') as f:
                    self.cfg['tamlec_params']['trg_vocab'] = torch.load(f)
            if self.cfg['paths']['abstract_dict'].is_file():
                with open(self.cfg['paths']['abstract_dict'], 'rb') as f:
                    self.cfg['tamlec_params']['abstract_dict'] = torch.load(f)
            if self.cfg['paths']['taxos_hector'].is_file():
                with open(self.cfg['paths']['taxos_hector'], 'rb') as f:
                    self.cfg['tamlec_params']['taxos_hector'] = torch.load(f)
            if self.cfg['paths']['taxos_tamlec'].is_file():
                with open(self.cfg['paths']['taxos_tamlec'], 'rb') as f:
                    self.cfg['tamlec_params']['taxos_tamlec'] = torch.load(f)
        else:
            self.cfg['taxonomy'] = Taxonomy()
            self.cfg['taxonomy'].load_taxonomy(self.cfg)
            
        if self.verbose:
            print(f"> Loading and pre-processing of the data...")

        

        parents = self.cfg['taxonomy'].label_to_parents
        dataset = self.cfg['dataset']

        if not split_only:
            
            documents, labels = [], []
            chunk_size = 2000

            with ProcessPoolExecutor(initializer=_init_worker, initargs=(parents, dataset)) as ex:
                max_inflight = (ex._max_workers or 1) * 3
                done_chunks = 0
                queued_chunks = 0

                for file_name in ['documents.json', 'train.json', 'test.json']:
                    p = self.cfg['paths']['dataset'] / file_name
                    if not p.exists():
                        continue

                    with open(p) as f:
                        it = iter(_chunks(f, chunk_size))
                        inflight = set()

                        # prime
                        for _ in range(max_inflight):
                            try:
                                inflight.add(ex.submit(_worker, next(it)))
                                queued_chunks += 1
                            except StopIteration:
                                break

                        print(f"\n{file_name}: queued {queued_chunks} chunks...", flush=True)

                        while inflight:
                            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)

                            for fut in done:
                                d, l = fut.result()
                                documents.extend(d)
                                labels.extend(l)
                                done_chunks += 1

                            # refill
                            for _ in range(len(done)):
                                try:
                                    inflight.add(ex.submit(_worker, next(it)))
                                    queued_chunks += 1
                                except StopIteration:
                                    break

                            # progress (prints even if no chunk finished yet? -> now you see queue moving)
                            if (done_chunks % 10) == 0 or (queued_chunks % 50) == 0:
                                print(f"\rqueued: {queued_chunks} | done: {done_chunks}", end="", flush=True)

            print()
            assert documents and len(documents) == len(labels)
            # Tokenize all documents
            documents_tokenized, vocabulary, fully_padded_indices = tokenization(documents, self.cfg)
            
            # Skip docs that turned into fully PAD (all zeros) after tokenization,
            

            if fully_padded_indices:
                print(f"> Skipping {len(fully_padded_indices)} documents that became fully padding after tokenization.")
                keep = [i for i in range(len(documents)) if i not in set(fully_padded_indices)]
                # Filter raw docs (only used for reporting/debug), tokenized docs, and labels
                documents = [documents[i] for i in keep]
                documents_tokenized = [documents_tokenized[i] for i in keep]
                labels = [labels[i] for i in keep]
            
            
            items_per_label, taxonomy, subtrees_to_keep, labels_per_idx = preprocess_docs_and_taxonomy(documents_tokenized, labels, self.cfg)
            # Get the embeddings
            emb_handler = EmbeddingHandler(self.cfg)
            # special_embs=True includes a <PAD> and a <UNK> token
            self.embeddings = emb_handler.get_glove_embeddings(vocabulary, special_embs=True)
            # Save data
            # Data for hector and tamlec
            torch.save(self.cfg['tamlec_params']['src_vocab'], self.cfg['paths']['src_vocab'])
            torch.save(self.cfg['tamlec_params']['trg_vocab'], self.cfg['paths']['trg_vocab'])
            torch.save(self.cfg['tamlec_params']['abstract_dict'], self.cfg['paths']['abstract_dict'])
            torch.save(self.cfg['tamlec_params']['taxos_hector'], self.cfg['paths']['taxos_hector'])
            torch.save(self.cfg['tamlec_params']['taxos_tamlec'], self.cfg['paths']['taxos_tamlec'])
            # Data for all algorithms
            torch.save(self.cfg['taxonomy'], self.cfg['paths']['taxonomy'])
            torch.save(self.embeddings, self.cfg['paths']['embeddings'])
                  
           
        else:
            items_per_label, taxonomy, subtrees_to_keep, labels_per_idx = get_split_datastructures(self.cfg)
        
        # Split data
        (self.labels_per_idx,
        self.global_indices,
        self.global_relevant_labels,
        self.tasks_indices,
        self.tasks_relevant_labels) = data_split(items_per_label, taxonomy, subtrees_to_keep, labels_per_idx, self.cfg)
        
        paths_dir = self.cfg["paths"]["paths_per_doc"]   # make this a directory Path
        
        # Save local/task indices
        torch.save((self.tasks_indices, self.tasks_relevant_labels), self.cfg['paths']['tasks_datasets'])
        # Save global indices
        torch.save((self.global_indices, self.global_relevant_labels), self.cfg['paths']['global_datasets'])
        torch.save(self.cfg['tasks_size'], self.cfg['paths']['tasks_size'])
        torch.save(self.cfg['task_to_subroot'], self.cfg['paths']['task_to_subroot'])
        torch.save(self.cfg['label_to_tasks'], self.cfg['paths']['label_to_tasks'])
    
    def get_dataloaders(self):
        '''
        Build dataloaders for global and task datasets.

        :return: Dict with train/validation/test loaders plus embeddings.
        '''
        tasks_train_loader = DataLoader(self.tasks_train_set, sampler=self.train_sampler, collate_fn=self.train_sampler.collate_standard(seed=None))
        tasks_val_loader = DataLoader(self.tasks_val_set, sampler=self.val_sampler, collate_fn=self.val_sampler.collate_standard(seed=16))
        tasks_test_loader = DataLoader(self.tasks_test_set, sampler=self.test_sampler, collate_fn=self.test_sampler.collate_standard(seed=16))
        global_train_set = GlobalDataset(self.global_indices['train'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=True)
        global_val_set = GlobalDataset(self.global_indices['val'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=False)
        global_test_set = GlobalDataset(self.global_indices['test'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=False)
        print(f"> Datasets size: train {len(global_train_set)} | val {len(global_val_set)} | test {len(global_test_set)}")
        if not self.one_hot_labels:
            # Do not shuffle in validation and test sets so we have always the same batches, which is not the case for training
            global_train_loader = DataLoader(global_train_set, batch_size=self.cfg['batch_size_train'], shuffle=True, drop_last=False, collate_fn=collate_global_int_labels)
            global_val_loader = DataLoader(global_val_set, batch_size=self.cfg['batch_size_eval'], shuffle=False, drop_last=False, collate_fn=collate_global_int_labels)
            global_test_loader = DataLoader(global_test_set, batch_size=self.cfg['batch_size_eval'], shuffle=False, drop_last=False, collate_fn=collate_global_int_labels)
        else:
            global_train_loader = DataLoader(global_train_set, batch_size=self.cfg['batch_size_train'], shuffle=True, drop_last=False)
            global_val_loader = DataLoader(global_val_set, batch_size=self.cfg['batch_size_eval'], shuffle=False, drop_last=False)
            global_test_loader = DataLoader(global_test_set, batch_size=self.cfg['batch_size_eval'], shuffle=False, drop_last=False)

        return {
            'global_train': global_train_loader,
            'global_validation': global_val_loader,
            'global_test': global_test_loader,
            'tasks_train': tasks_train_loader,
            'tasks_validation': tasks_val_loader,
            'tasks_test': tasks_test_loader,
            'embeddings': self.embeddings,
        }
        
    
