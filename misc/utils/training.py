

from pathlib import Path
import torch
import copy
import numpy as np


class LRHandler:
    """Handles learning rate scheduling with warm-up, cosine decay, and a minimum learning rate threshold.

    This class adjusts the learning rate of an optimizer dynamically over the course of training. 
    It supports a warm-up phase at the start of training, followed by a cosine decay schedule, 
    and maintains a minimum learning rate after the decay phase.

    Args:
        cfg (dict): Configuration dictionary.
        optimizer (torch.optim.Optimizer): The optimizer for which learning rate will be updated.

    Attributes:
        epochs (int): Total number of epochs.
        max_lr (float): Maximum learning rate.
        warm_up_epochs (int): Number of warm-up epochs (20% of total epochs).
        decay_epochs (int): Number of decay epochs (80% of total epochs).
        min_lr (float): Minimum learning rate (10% of the maximum learning rate).
        optimizer (torch.optim.Optimizer): The optimizer for which learning rate will be updated.
        curr_lr (float): Current learning rate being used.

    Methods:
        update_lr(curr_epoch):
            Updates the learning rate based on the current epoch.
    """

    def __init__(self, cfg, optimizer):
        self.epochs = cfg['epochs']
        self.max_lr = cfg['learning_rate']
        self.warm_up_epochs = int(0.2 * self.epochs)
        self.decay_epochs = int(0.8 * self.epochs)
        self.min_lr = 0.1 * self.max_lr
        self.optimizer = optimizer

    def update_lr(self, curr_epoch):
        # Update the lr according the curr_epoch
        # Warm-up phase
        if curr_epoch <= self.warm_up_epochs:
            self.curr_lr = curr_epoch * self.max_lr / self.warm_up_epochs

        # After the decay, keep to the min_lr
        elif curr_epoch > self.decay_epochs:
            self.curr_lr = self.min_lr

        # Cosine decay
        else:
            decay_ratio = (curr_epoch - self.warm_up_epochs) / (self.decay_epochs - self.warm_up_epochs)
            self.curr_lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * decay_ratio))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.curr_lr


class EarlyStopping:
    """
    Implements early stopping to monitor training and stop the process when the performance metric does not improve within a specified patience.

    Attributes:
        cfg (dict): Configuration dictionary.
        finetuning (bool): Indicates whether the training is for fine-tuning a model (during few-shot experiment).
        best_metric (float): Best metric value achieved during training.
        max_patience (int): Maximum number of epochs to wait for improvement before stopping.
        current_patience (int): Number of epochs since the last improvement.
        metrics_dict (dict): Stores the metrics of the best model so far.
        best_epoch (int): Epoch at which the best metric was achieved.
        method_to_min_epoch (dict): Maps algorithms to their minimum number of epoch before starting the monitoring.
        min_epoch (int): Minimum number of epochs required before early stopping is applied.

    Methods:
        checkpoint(new_model, metrics_dict, epoch): Evaluates the current epoch and decides whether to stop or continue.
        _save_model(new_model): Saves the current model to the configured path.
    """

    def __init__(
        self, 
        method:str,
        max_patience:int,
        state_dict_path:str,
        state_dict_finetuned_path:str,
        model_path:Path,
        model_finetuned_path:str,
        min_improvement: float = .0, # Minimum improvement for restarting patience
        finetuning=False
        ):
        
        self.min_improvement = min_improvement
        self.model_path = model_path
        self.model_finetuned_path = model_finetuned_path
        self.state_dict_path = state_dict_path
        self.state_dict_finetuned_path = state_dict_finetuned_path
        self.method = method
        self.finetuning = finetuning
        self.best_metric = float('inf')
        self.max_patience = max_patience
        self.current_patience = 0
        self.metrics_dict = {}
        self.best_epoch = 0
        self.method_to_min_epoch = {
            'protonet': 30,
            'maml': 20,
            'match': 5,
            'xmlcnn': 5,
            'attentionxml': 5,
            'hector': 5,
            'tamlec': 0,
            'siamese': 15,
            'bdc': 15,
            'lightxml': 5,
            'cascadexml': 5,
            'parabel': 5,
        }
        self.min_epoch = self.method_to_min_epoch[self.method]

        

    
    def reset(self):
        self.best_metric = float('inf')
        self.current_patience = 0        

    # Check if we need to continue the training
    # It also saves the model occasionally
    def checkpoint(self, new_model, metrics_dict, epoch,saving=True):
        # Minimum number of epoch on which we do not check
        if epoch < self.min_epoch and not self.finetuning:
            # Anyway still save the model in training for the first epochs
            if saving:
                self._save_model(new_model)
            return True

        # Take last loss and check if improving or not
        new_metric = metrics_dict['loss']
        if new_metric < self.best_metric * (1. - self.min_improvement): # at least self.min_improvement% decrease
            self.metrics_dict = metrics_dict
            self.best_epoch = epoch
            self.best_metric = new_metric
            self.current_patience = 0
            # Save the model
            if saving:
                self._save_model(new_model)
        else:
            self.current_patience += 1

        if self.current_patience == self.max_patience:
            return False
        return True

    def _save_model(self, new_model):
        #print(f">>> Saving model at epoch {self.best_epoch} with loss {self.best_metric:.4f}")
        # Save the entire model for those methods
        if self.method in ['hector', 'tamlec', 'attentionxml']:
            if self.finetuning:
                torch.save(new_model, self.model_finetuned_path)
            else:
                torch.save(new_model, self.model_path)
        # Here save the state dict of a Pytorch model than can be loaded afterwards
        else:
            if self.finetuning:
                torch.save(copy.deepcopy(new_model.state_dict()), self.state_dict_finetuned_path)
            else:
                torch.save(copy.deepcopy(new_model.state_dict()), self.state_dict_path)


class AdaptivePatience:
    """
    Manages adaptive patience for multi-task learning, allowing independent early stopping
    for individual tasks as well as a global criterion. Used in TAMLEC

    Attributes:
        cfg (dict): Configuration dictionary.
        max_patience (int): Maximum patience for all tasks and the global criterion.
        n_tasks (int): Total number of tasks being trained.
        tasks_to_complete (list): List of task IDs that are yet to complete training.
        patiences (dict): Dictionary of `EarlyStopping` objects for each task and global criterion.
        tasks_completed (list): List of task IDs that have completed the training.
        metrics_dicts (dict): Stores the best metrics for the completed tasks.
        best_epochs (dict): Tracks the epoch at which each task achieved its best performance.
    
    Methods:
        global_checkpoint(new_model, metrics_dict, epoch): Checks global and task-specific criteria for early stopping.
        tasks_checkpoint(new_model, metrics_dict, epoch): Checks task-specific criteria for early stopping.
        mark_task_done(task_id): Marks a task as completed manually.
    """

    def __init__(
        self,
        method:str,
        max_patience:int,
        state_dict_path:str,
        state_dict_finetuned_path:str,
        model_path:Path,
        model_finetuned_path:str,
        n_tasks:int,
        all_tasks_key,
        finetuning=False,
        min_improvement = .0
        ):
        
        self.max_patience = max_patience
        self.n_tasks = n_tasks
        self.state_dict_path = state_dict_path
        self.state_dict_finetuned_path = state_dict_finetuned_path
        self.model_path = model_path
        self.model_finetuned_path = model_finetuned_path
        self.method = method
        self.finetuning = finetuning
        self.all_tasks_key = all_tasks_key
        self.tasks_to_complete = list(range(n_tasks))
        
        self.patiences = {
            idx: EarlyStopping(
                method = self.method, 
                state_dict_path = self.state_dict_path,
                state_dict_finetuned_path = self.state_dict_finetuned_path,
                model_path = self.model_path,
                model_finetuned_path = self.model_finetuned_path,
                finetuning = self.finetuning,
                max_patience = self.max_patience,
                min_improvement = min_improvement
                )
            for idx in self.tasks_to_complete
            }
        
        self.patiences[self.all_tasks_key] = EarlyStopping(
                method = self.method, 
                state_dict_path = self.state_dict_path,
                state_dict_finetuned_path = self.state_dict_finetuned_path,
                model_path = self.model_path,
                model_finetuned_path = self.model_finetuned_path,
                finetuning = self.finetuning,
                max_patience = self.max_patience,
                min_improvement = min_improvement
        )
        
        self.tasks_completed = []
        self.metrics_dicts = {}
        self.best_epochs = {}
    def reset(self):
        for task_id in self.tasks_to_complete:
            self.patiences[task_id].best_metric = float('inf')
            self.patiences[task_id].current_patience = 0 
        self.patiences[self.all_tasks_key].best_metric = float('inf')
        self.patiences[self.all_tasks_key].current_patience = 0
        
    def global_checkpoint(self, new_model, metrics_dict, epoch, freezing):
        # Check if any task has converged
        should_save=False
        if freezing :
            for task_id in self.tasks_to_complete:
                if not self.patiences[task_id].checkpoint(new_model, {"loss": metrics_dict[task_id]}, epoch,saving=False):
                    should_save=True
                    self.tasks_completed.append(task_id)
                    self.tasks_to_complete.remove(task_id)
                    self.metrics_dicts[task_id] = metrics_dict[task_id]
                    self.best_epochs[task_id] = epoch
                    print(f">>> Task {task_id} completed at epoch {self.best_epochs[task_id]}, N still to complete: {len(self.tasks_to_complete)}")
            if should_save:
                self.patiences[self.all_tasks_key]._save_model(new_model)


        # If converged on the global loss then stop, otherwise continue training
        return self.patiences[self.all_tasks_key].checkpoint(new_model, {"loss":np.mean(metrics_dict)}, epoch)

    def tasks_checkpoint(self, new_model, metrics_dict, epoch):
        # Check if any task has converged, and stop it if this is the case
        for task_id in self.tasks_to_complete:
            if not self.patiences[task_id].checkpoint(new_model, metrics_dict[task_id], epoch):
                self.tasks_completed.append(task_id)
                self.tasks_to_complete.remove(task_id)
                self.metrics_dicts = metrics_dict[task_id]
                self.best_epochs[task_id] = epoch
                print(f">>> Task {task_id} completed at epoch {self.best_epochs[task_id]}, still to complete: {self.tasks_to_complete}")
        return len(self.tasks_to_complete) != 0

    def mark_task_done(self, task_id):
        if task_id not in self.tasks_completed:
            self.tasks_to_complete.remove(task_id)
            self.tasks_completed.append(task_id)
