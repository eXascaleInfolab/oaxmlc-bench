import torch
from torch.utils.data import DataLoader
from datahandler.datasets import TasksDatasetHectorTamlec, ResampledTasksDataset
from datahandler.datasets import GlobalDatasetHectorTamlec

from datahandler.samplers.tasks_sampler import SubtreeSampler
from datahandler.samplers.global_sampler import collate_global_int_labels_hector_tamlec
from datahandler.data_modules.base import DataModuleBase

class DataModuleHectorTamlec(DataModuleBase):
    '''
    Data module for Hector/Tamlec models with specialized datasets/collation.
    '''
    def __init__(self, cfg, one_hot_labels,verbose=True):
        '''
        Initialize Hector/Tamlec data module and load extra vocab assets.

        :param cfg: Experiment configuration dictionary.
        :param one_hot_labels: Ignored; Hector/Tamlec uses integer labels.
        :param verbose: If True, print preprocessing/loading messages.
        :return: None.
        '''
        super().__init__(cfg, one_hot_labels=False, verbose=verbose)
        #Check if data is already pre-processed
        condition1 = self.cfg['paths']['taxos_hector'].is_file()
        condition2 = self.cfg['paths']['taxos_tamlec'].is_file()
        condition3 = self.cfg['paths']['global_datasets'].is_file()
        
        # Additional preprocessed data to be loaded for Hector and Tamlec
        if (condition1 and condition2 and condition3):
            with open(cfg['paths']['src_vocab'], 'rb') as f:
                cfg['tamlec_params']['src_vocab'] = torch.load(f)
            with open(cfg['paths']['trg_vocab'], 'rb') as f:
                cfg['tamlec_params']['trg_vocab'] = torch.load(f)
            with open(cfg['paths']['abstract_dict'], 'rb') as f:
                cfg['tamlec_params']['abstract_dict'] = torch.load(f)
            with open(cfg['paths'][f"taxos_{cfg['method']}"], 'rb') as f:
                cfg['tamlec_params'][f"taxos_{cfg['method']}"] = torch.load(f)
        if self.cfg['tamlec_params']['tasks_size']:
            self.cfg['tamlec_params']['tasks_size'] = self.cfg['tasks_size']
        else:
            self.cfg['tamlec_params']['tasks_size'] = None
        self.tasks_val_set = TasksDatasetHectorTamlec(self.tasks_indices['val'], self.tasks_relevant_labels, one_hot_labels, cfg)
        self.tasks_test_set = TasksDatasetHectorTamlec(self.tasks_indices['test'], self.tasks_relevant_labels, one_hot_labels, cfg)

        # Construct samplers
        self.val_sampler = SubtreeSampler(self.tasks_val_set, cfg=cfg, batch_size=cfg['batch_size_eval'])
        self.test_sampler = SubtreeSampler(self.tasks_test_set, cfg=cfg, batch_size=cfg['batch_size_eval'])
        
    def get_dataloaders(self):
        '''
        Build dataloaders with Hector/Tamlec-specific collation.

        :return: Dict with train/validation/test loaders plus embeddings.
        '''
        resampled_train_set = ResampledTasksDataset(self.tasks_indices['train'], self.tasks_relevant_labels, self.cfg)
        new_sampler = SubtreeSampler(resampled_train_set, cfg=self.cfg, batch_size=self.cfg['batch_size_train'])
        tasks_train_loader = DataLoader(resampled_train_set, sampler=new_sampler, collate_fn=new_sampler.collate_hector_tamlec(seed=None))
        tasks_val_loader  = DataLoader(self.tasks_val_set,  sampler=self.val_sampler, collate_fn=self.val_sampler.collate_hector_tamlec(seed=16))
        tasks_test_loader = DataLoader(self.tasks_test_set, sampler=self.test_sampler, collate_fn=self.test_sampler.collate_hector_tamlec(seed=16))
        global_train_set = GlobalDatasetHectorTamlec(self.global_indices['train'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=True)
        global_val_set = GlobalDatasetHectorTamlec(self.global_indices['val'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=False)
        global_test_set = GlobalDatasetHectorTamlec(self.global_indices['test'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=False)
        print(f"> Datasets size: train {len(global_train_set)} | val {len(global_val_set)} | test {len(global_test_set)}")
        if not self.one_hot_labels:
            # Do not shuffle in validation and test sets so we have always the same batches, which is not the case for training
            global_train_loader = DataLoader(global_train_set, batch_size=self.cfg['batch_size_train'], shuffle=True, drop_last=False, collate_fn=collate_global_int_labels_hector_tamlec)
            global_val_loader = DataLoader(global_val_set, batch_size=self.cfg['batch_size_eval'], shuffle=False, drop_last=False, collate_fn=collate_global_int_labels_hector_tamlec)
            global_test_loader = DataLoader(global_test_set, batch_size=self.cfg['batch_size_eval'], shuffle=False, drop_last=False, collate_fn=collate_global_int_labels_hector_tamlec)
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
        
     
 
