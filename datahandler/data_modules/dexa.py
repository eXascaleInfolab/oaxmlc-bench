import torch
from pathlib import Path       
from datahandler.taxonomy import Taxonomy

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datahandler.data_modules.base import _init_worker, _chunks, _worker
from datahandler.utils import tokenization_hf, data_split_hf


class DataModuleDexa:
    def __init__(self, cfg:dict, one_hot_labels, verbose=True):
        '''
        Initialize the DEXA data module and ensure preprocessing artifacts exist.

        :param cfg: Experiment configuration dictionary.
        :type cfg: dict
        :param one_hot_labels: Unused for DEXA data module.
        :param verbose: If True, print preprocessing/loading messages.
        :return: None.
        '''
        self.verbose = verbose
        # Add sensible default values for splitting parameters
        cfg.setdefault('split', {})
        cfg['split'].setdefault('rare_label_fraction', 0.10)   # bottom 10% of labels considered "rarest"
        cfg['split'].setdefault('rare_holdout_ratio', 0.10)    # hold out 10% of docs per rare label
        seed = cfg['seed']
        preprocessed_path = Path(cfg['paths']['dataset']) / f'preprocessed_data/seed_{seed}'
        #Check if data is already pre-processed
        condition1 = (preprocessed_path / 'trn_X_Y.txt').is_file()
        condition2 = (preprocessed_path / 'tst_X_Y.txt').is_file()
        condition3 = (preprocessed_path / 'filter_labels_test.txt').is_file()
        condition4 = (preprocessed_path / 'filter_labels_train.txt').is_file()
        self.cfg = cfg
        
        self.one_hot_labels = one_hot_labels
        
        # if data not already pre-processed, preprocess else load it
        if not (condition1 and condition2 and condition3 and condition4):
            self.preprocess()
        else:
            with open(cfg['paths']['taxonomy_dexa'], 'rb') as f:
                self.cfg['taxonomy'] = torch.load(f)
            with open(preprocessed_path/'trn_X_Y.txt', 'r') as f:
                self.cfg['n_global_train'] = int(f.readline().split(' ')[0])
            with open(preprocessed_path/'tst_X_Y.txt', 'r') as f:
                self.cfg['n_global_test'] = int(f.readline().split(' ')[0])
            with open(preprocessed_path/'taxos_dexa.pt', 'rb') as f:
                self.cfg["tamlec_params"]['taxos_tamlec'] = torch.load(f)
            with open(preprocessed_path/'task_to_subroot_dexa.pt', 'rb') as f:
                self.cfg['task_to_subroot'] = torch.load(f)
            
    def preprocess(self):
        '''
        Preprocess raw data for the DEXA pipeline.

        This loads taxonomy, tokenizes documents using HF tokenization, splits
        data, and saves the required artifacts to disk.

        :return: None.
        '''
        (self.cfg['paths']['preprocessed_data'] / f"seed_{self.cfg['seed']}").mkdir(exist_ok = True)
        self.cfg['taxonomy'] = Taxonomy()
        self.cfg['taxonomy'].load_taxonomy(self.cfg)
        if self.verbose:
            print(f"> Loading and pre-processing of the data...")

        

        parents = self.cfg['taxonomy'].label_to_parents
        dataset = self.cfg['dataset']

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
        documents_tokenized, vocabulary, fully_padded_indices, _ = tokenization_hf(documents, self.cfg)
        

        if fully_padded_indices:
            print(f"> Skipping {len(fully_padded_indices)} documents that became fully padding after tokenization.")
            keep = [i for i in range(len(documents)) if i not in set(fully_padded_indices)]
            # Filter raw docs (only used for reporting/debug), tokenized docs, and labels
            documents = [documents[i] for i in keep]
            documents_tokenized = [documents_tokenized[i] for i in keep]
            labels = [labels[i] for i in keep]
        
        
            
        # Split data
        data_split_hf(documents_tokenized, labels, self.cfg)
        torch.save(self.cfg['taxonomy'], self.cfg['paths']['taxonomy_dexa'])
        torch.save(self.cfg['tamlec_params']['taxos_tamlec'], self.cfg['paths']['taxos_dexa'])
        torch.save(self.cfg['task_to_subroot'], self.cfg['paths']['task_to_subroot_dexa'])
        
    def get_dataloaders(self):
        '''
        Return placeholder dataloaders for DEXA (handled externally).

        :return: Dict with None entries for loaders/embeddings.
        '''
        return {
            'global_train': None,
            'global_validation': None,
            'global_test': None,
            'tasks_train': None,
            'tasks_validation': None,
            'tasks_test': None,
            'embeddings': None,
        }
        
