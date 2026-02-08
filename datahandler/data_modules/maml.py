from torch.utils.data import DataLoader
from datahandler.datasets.global_datasets import GlobalDataset
from datahandler.data_modules.base import DataModuleBase

       
class DataModuleMaml(DataModuleBase):
    '''
    Data module for MAML-style few-shot training.
    '''
    def __init__(self, cfg, verbose=True):
        super().__init__(cfg, one_hot_labels=True, verbose=verbose)
        
    def get_dataloaders(self):
        '''
        Build dataloaders for MAML training and evaluation.

        :return: Dict with train/validation/test loaders plus embeddings.
        '''
        tasks_train_loader = DataLoader(self.tasks_train_set, sampler=self.train_sampler, collate_fn=self.train_sampler.collate_fewshot)
        tasks_val_loader = DataLoader(self.tasks_val_set, sampler=self.val_sampler, collate_fn=self.val_sampler.collate_standard(seed=16))
        tasks_test_loader = DataLoader(self.tasks_test_set, sampler=self.test_sampler, collate_fn=self.test_sampler.collate_standard(seed=16))
        global_train_set = GlobalDataset(self.global_indices['train'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=True)
        global_val_set = GlobalDataset(self.global_indices['val'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=False)
        global_test_set = GlobalDataset(self.global_indices['test'], self.global_relevant_labels, self.one_hot_labels, self.cfg, train_dataset=False)
        print(f"> Datasets size: train {len(global_train_set)} | val {len(global_val_set)} | test {len(global_test_set)}")
        
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
