from base import DataModuleBase
from datahandler.datasets.global_datasets import GlobalDataset
from datahandler.samplers.global_sampler import collate_global_int_labels
from torch.utils.data import DataLoader

class DataModuleFastxmlParabel(DataModuleBase):
    '''
    Data module for FastXML/Parabel baselines with integer labels.
    '''
    def __init__(self, cfg, one_hot_labels, verbose=True):
        '''
        Initialize FastXML/Parabel data module.

        :param cfg: Experiment configuration dictionary.
        :param one_hot_labels: Ignored; FastXML/Parabel uses integer labels.
        :param verbose: If True, print preprocessing/loading messages.
        :return: None.
        '''
        super().__init__(cfg, one_hot_labels=False, verbose = verbose)
        
    def get_dataloaders(self):
        '''
        Build dataloaders for FastXML/Parabel workflows.

        :return: Dict with train/validation/test loaders plus embeddings.
        '''
        tasks_train_loader = DataLoader(self.tasks_train_set, sampler=self.train_sampler, collate_fn=self.train_sampler.collate_no_batch)
        tasks_val_loader = DataLoader(self.tasks_val_set, sampler=self.val_sampler, collate_fn=self.val_sampler.collate_no_batch)
        tasks_test_loader = DataLoader(self.tasks_test_set, sampler=self.test_sampler, collate_fn=self.test_sampler.collate_no_batch)
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
        
