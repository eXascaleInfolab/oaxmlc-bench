from torch.utils.data import DataLoader
from datahandler.data_modules.base import DataModuleBase

class DataModuleProtoNetBDC(DataModuleBase):
    '''
    Data module for ProtoNet/BDC few-shot experiments.
    '''
    def __init__(self, cfg, one_hot_labels, verbose=True):
        '''
        Initialize ProtoNet/BDC data module.

        :param cfg: Experiment configuration dictionary.
        :param one_hot_labels: Ignored; ProtoNet/BDC uses one-hot labels.
        :param verbose: If True, print preprocessing/loading messages.
        :return: None.
        '''
        super().__init__(cfg, one_hot_labels=True, verbose=verbose)
    
    def get_dataloaders(self):
        '''
        Build few-shot task dataloaders for train/validation/test.

        :return: Dict with task loaders plus embeddings.
        '''
        # For protonet and bdc we need the fewshot sampler for all sets
        tasks_train_loader = DataLoader(self.tasks_train_set, sampler=self.train_sampler, collate_fn=self.train_sampler.collate_fewshot)
        # The seed is fixed for validation and test sets to have always the same batches, which is not the case for training
        tasks_val_loader = DataLoader(self.tasks_val_set, sampler=self.val_sampler, collate_fn=self.val_sampler.collate_fewshot_eval(seed=16))
        tasks_test_loader = DataLoader(self.tasks_test_set, sampler=self.test_sampler, collate_fn=self.test_sampler.collate_fewshot_eval(seed=16))
        
        return {
            'tasks_train': tasks_train_loader,
            'tasks_validation': tasks_val_loader,
            'tasks_test': tasks_test_loader,
            'embeddings': self.embeddings,
        }
