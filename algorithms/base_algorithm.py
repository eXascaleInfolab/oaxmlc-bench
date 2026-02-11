from __future__ import annotations
from tqdm import tqdm
import torch
import numpy as np
import time
from pathlib import Path
from misc.utils.time import format_time_diff
from misc.utils.training import EarlyStopping
import copy
import random
import models


from abc import ABC, abstractmethod
from typing import  Dict, Iterable, Tuple, Optional, Any
from misc.checkpointing import CheckpointManager

class AbstractAlgorithm(ABC):
    """
    Minimal common interface for training/evaluation algorithms.

    Children must implement:
      - load_model(): prepare self.model on the correct device
      - run_init(): any extra one-off setup before training (tokenizers, caches, etc.)
      - inference_eval(batch): forward pass used during evaluation (no grad)
      - optimization_loop(inputs, labels): one optimization step and bookkeeping
      - train(dataloaders): full training loop (can be a no-op for non-trainable methods)

    Conventions:
      - `device` attribute holds the active torch.device or string (e.g., "cuda:0").
      - `method` is a short identifier for logging/registry.
      - `model` should be a torch.nn.Module moved to `device`.
    """

    @property
    def method_name(self) -> str:
        """Convenience accessor for logging/registry."""
        return str(getattr(self, "method", "unknown"))

    # ---- Lifecycle / setup ----
    @abstractmethod
    def load_model(self) -> None:
        """Instantiate and/or load weights into self.model, move to device."""
        raise NotImplementedError
    
    @abstractmethod    
    def save_model(self, path: Optional[str] = None) -> None:
        """Optional: override to save model/optimizer state."""
        raise NotImplementedError("save_model is optional and not implemented.")

    @abstractmethod
    def run_init(self) -> None:
        """Optional extra initialization hook (tokenizers, caches, etc.)."""
        raise NotImplementedError

    # ---- Core algorithm steps ----
    @abstractmethod
    def inference_eval(self, input_data: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass used for evaluation (should be under no-grad by the caller).
        Returns a tensor of predictions/scores shaped [B, n_labels] (or method-specific).
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, dataloaders: Dict[str, Iterable]) -> None:
        """
        Orchestrate training over dataloaders (e.g., keys like 'global_train', ...).
        Implementers may ignore unused keys.
        """
        raise NotImplementedError
    
    @abstractmethod
    def finetune(self, dataloaders):
        raise NotImplementedError       

    
    

# Base algorithm class that initializes the model, trains and evaluates it
# This base class needs to be used as parent class for the algorithms
class BaseAlg(AbstractAlgorithm):
    """Common training/evaluation utilities shared by neural algorithms."""
    def __init__(
        self,
        metrics_handler,
        loss_function: torch.nn.Module, # Any torch loss function (e.g. torch.nn.BCELoss(),
        optimizer: Optional[torch.optim.Optimizer] = None, # Optimizer class from torch.optim
        method: str = "", # Identifier of the model/method (e.g., "lightxml").
        device: str = 'cpu', # Device to use (cpu, cuda:0, ...)
        state_dict_path: str = "", # Path to a state dict to load the model from
        state_dict_finetuned_path: str = "", # Path to a state dict to load the finetuned model from
        model_path: Path = Path(), # Path to a model to load the model from
        model_finetuned_path: str = "", # Path to a model to load the finetuned model from
        verbose=True,
        patience: int = 10, # Maximum number of epochs without improvement before early stopping
        all_tasks_key: str = "", # Key used to store global (all-tasks) metrics
        selected_task: int = 0, # Task index to fine-tune if few-shot mode is enabled
        checkpoint_manager: Optional[CheckpointManager] = None ,
        resume: bool = True,
        min_improvement: float = .0,
        **kwargs
        ):
        """
        Initializes the BaseAlg class with the provided configuration and metrics handler.

        Args:
            cfg (dict): Configuration dictionary containing all necessary experiment parameters.
                Required keys:
                    - device (str | int | torch.device): CUDA device spec (e.g., "cuda:0") or "cpu".
                    - taxonomy (object): Taxonomy structure used for per-level metric aggregation.
                    - patience (int): Maximum number of epochs without improvement before early stopping.
                    - k_list (List[int]): Cutoffs for final evaluation (e.g., [1, 3, 5]).
                    - k_list_eval_perf (List[int]): Cutoffs used during validation evaluations.
                    - loss_function (str): Name of the loss function (e.g., "bce", "softmax").
                    - threshold (float): Threshold for score binarization in metric computation.
                    - method (str): Identifier of the model/method (e.g., "LightXML").
                    - all_tasks_key (str | int): Key used to store global (all-tasks) metrics.
                    - paths (dict): Dictionary of file paths for saving predictions, labels, and metrics.
                        Must include:
                            * "<split>_predictions", "<split>_labels", "<split>_relevant_labels"
                            * "<split>_level_metrics" for each split (train/validation/test)
                            * (optional for fine-tuning)
                              "<split>_predictions_finetuned", "<split>_labels_finetuned"
                    - fewshot_exp (bool): Whether to perform a fine-tuning phase on a selected task.
                    - selected_task (int): Task index to fine-tune if few-shot mode is enabled.
        
        metrics_handler (dict): Dictionary of MetricsHandler objects to track results across
                different phases, typically with keys such as:
                    - "training"
                    - "eval_validation"
                    - "eval_test"

        verbose (bool, optional): If True, prints device information and progress logs. Defaults to True.
        """
        self.optimizer = optimizer
        
        self.checkpoint_manager = checkpoint_manager
        self.resume_enabled = resume
        self._current_stage = "train"
        self._resumed = False
        self._resume_epoch = 0
        
        self.epoch = None
        self.model = None
        self.scheduler = None
        self.selected_task = selected_task
        self.all_tasks_key = all_tasks_key
        self.model_path = model_path
        self.model_finetuned_path = model_finetuned_path
        self.state_dict_path = state_dict_path
        self.state_dict_finetuned_path = state_dict_finetuned_path
        self.loss_function = loss_function
        self.device = device
        self.verbose = verbose
        self.method = method
        self.patience = patience
        # Select correct torch device
        dev = str(self.device)

        if dev.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(dev)
            idx = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(idx)
            if self.verbose: print(f"> Using CUDA device: {self.device}")

        elif dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            if self.verbose: print("> Using MPS")

        else:
            self.device = torch.device("cpu")
            if self.verbose: print("> Using CPU")
        
        # Monitor performance and stop after convergence
        self.early_stopping = EarlyStopping(
            method = self.method,
            max_patience=self.patience,
            state_dict_path=self.state_dict_path,
            state_dict_finetuned_path=self.state_dict_finetuned_path,
            model_path=self.model_path,
            model_finetuned_path=self.model_finetuned_path,
            finetuning=False,
            min_improvement = min_improvement 
            )
        # Save the metrics
        self.metrics_handler = metrics_handler
        
    def _attempt_resume(self):
        """Attempt to resume training from the latest checkpoint."""
        if self.checkpoint_manager is not None:
            record = self.checkpoint_manager.latest()
        if record is None:
            return
        epoch = record["epoch"]
        model_path = record["model_path"]
        state = record["state"]

        self.load_checkpoint(model_path)
        self.restore_training_state(state)
        self._resume_epoch = epoch
        self.epoch = epoch
        self._resumed = True
        if self.verbose:
            print(f"> Resumed from checkpoint @ epoch {epoch}")



    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a checkpoint payload into the current model."""
        payload = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(payload, torch.nn.Module):
            self.model = payload.to(self.device)
        elif isinstance(payload, dict):
            if isinstance(self.model, torch.nn.Module):
                self.model.load_state_dict(payload)
                self.model.to(self.device)
        elif isinstance(payload, object):
            self.model = payload
        else:
            raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")

    def capture_training_state(self) -> Dict[str, Any]:
        """Capture RNG/optimizer state for checkpointing."""
        state: Dict[str, Any] = {}

        if getattr(self, "optimizer", None) is not None and isinstance(self.optimizer, torch.optim.Optimizer):
            state["optimizer"] = self.optimizer.state_dict()
        if getattr(self, "scheduler", None) is not None and isinstance(self.scheduler, torch.optim.lr_scheduler):
            state["scheduler"] = self.scheduler.state_dict() # type: ignore

        state["rng_state"] = torch.get_rng_state()
        state["numpy_state"] = np.random.get_state()
        state["python_state"] = random.getstate()
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        state["epoch"] = self.epoch or 0
        state["stage"] = self._current_stage
        return state

    def restore_training_state(self, state: Dict[str, Any]) -> None:
        """Restore RNG/optimizer state from a checkpoint payload."""
        if getattr(self, "optimizer", None) is not None and "optimizer" in state and isinstance(self.optimizer, torch.optim.Optimizer):
            self.optimizer.load_state_dict(state["optimizer"])
        if getattr(self, "scheduler", None) is not None and "scheduler" in state and isinstance(self.scheduler, torch.optim.lr_scheduler):
            self.scheduler.load_state_dict(state["scheduler"]) # type: ignore

        if "rng_state" in state:
            torch.set_rng_state(state["rng_state"])
        if "numpy_state" in state:
            np.random.set_state(state["numpy_state"])
        if "python_state" in state:
            random.setstate(state["python_state"])
        if "cuda_rng_state_all" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])

        self._resume_epoch = int(state.get("epoch", 0))
        self._current_stage = state.get("stage", self._current_stage)

    def _save_checkpoint(self, *, epoch: int, metrics: Dict[str, float]) -> None:
        """Persist a checkpoint using the configured checkpoint manager."""
        if not self.checkpoint_manager:
            return

        state = self.capture_training_state()
        state["metrics"] = metrics
        self.checkpoint_manager.save(
            epoch=epoch,
            model_saver=lambda path: self.save_model(path),
            state=state,
        )

    # Returns nothing
    def save_model(self, path: Optional[Path] = None, *, is_finetune: bool = False) -> None:
        """Save model weights or full model depending on method."""
        if self.method in {"attentionxml"}:
            target = Path(path) if path is not None else (
                Path(self.model_finetuned_path) if is_finetune else Path(self.model_path)
            )
            torch.save(self.model, target)
        else:
            target = Path(path) if path is not None else (
                Path(self.state_dict_finetuned_path) if is_finetune else Path(self.state_dict_path)
            )
            torch.save(copy.deepcopy(self.model.state_dict()), target) # type: ignore

    
    
    # Returns nothing
    def load_model(self):
        """Stub: subclasses should implement loading behavior."""
        print(f">> load_model method should be overridden in the child class")
        import sys; sys.exit()
        
    def load_model_finetuned(self):
        """Stub: subclasses should implement finetuned loading behavior."""
        print(f">> load_model method should be overridden in the child class")
        import sys; sys.exit()
        
    # Returns the predictions
    def inference_eval(self, input_data, **kwargs):
        """Stub: subclasses should implement inference."""
        print(f">> inference_eval method should be overridden in the child class")
        import sys; sys.exit()
        # return predictions

    # Returns the train loss, and the gradients norms and their number
    def optimization_loop(self, input_data, labels, **kwargs) -> Iterable:
        """Stub: subclasses should implement a training step."""
        print(f">> optimization_loop method should be overridden in the child class")
        import sys; sys.exit()
        # return train_loss, (gradients_norms, gradient_lengths)

    # Returns nothing
    def run_init(self):
        """Stub: subclasses may implement extra initialization."""
        print(f">> run_init method should be overridden in the child class")
        import sys; sys.exit()

    
    @torch.no_grad()
    def get_valloss(self, val_loader):
        """
        Computes the mean validation loss without using XML metrics.

        Iterates over the validation split, obtains predictions using
        `self.inference_eval()`, and computes the average loss across batches
        using the loss function stored in `self.cfg['loss_function']`.

        Returns:
            float: Mean validation loss over the entire validation set.
        """
        self.model.eval() # type: ignore
        total_loss = 0.0
        total_samples = 0

        for input_data, labels, _ in val_loader:
            # Move tensors to the correct device
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            predictions = self.inference_eval(input_data).to(self.device)

            # Compute loss for the batch
            batch_loss = self.loss_function(predictions, labels).item()

            # Accumulate weighted average
            total_loss += batch_loss * len(input_data)
            total_samples += len(input_data)

        mean_val_loss = total_loss / total_samples if total_samples > 0 else float("nan")
        return mean_val_loss
    
    def train(self, dataloaders):
        """
        Training loop.
        Args:
            dataloaders (dict): Dictionary with keys: 
                'tasks_train': torch.utils.data.DataLoader loading task nodes (subtrees of the taxonomy) - Train split,
                'tasks_validation': torch.utils.data.DataLoader loading task nodes (subtrees of the taxonomy) - Validation split,
                'tasks_test': torch.utils.data.DataLoader loading task nodes (subtrees of the taxonomy) - Test split,
                'embeddings': torch tensor of per token embeddings (e.g. <pad> -> all zeroes, 'dog' -> pretrained embedding vector),
        """
        
        if self.checkpoint_manager and self.resume_enabled:
            self._attempt_resume()
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        training = True
        self.epoch = self._resume_epoch
        split = 'train'
        finetuning = False
        start_train = time.perf_counter()
        

        
        while training:
            self.epoch += 1
            # Storages for the training loss and gradients norms
            train_losses = []
            n_docs_per_batch = []
            grad_norms = []
            grad_lengths = []
            # Train on the batches
            for input_data, labels, _ in tqdm(dataloaders[f"global_{split}"], leave=False, desc=f"Training on {split}"):
                # Send everything on device
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                # Optimization loop
                train_loss, gradients_and_lengths = self.optimization_loop(input_data, labels)
                # Aggregate train loss and gradient norms
                train_losses.append(train_loss)
                n_docs_per_batch.append(len(input_data))
                grad_norms.append(gradients_and_lengths[0])
                grad_lengths.append(gradients_and_lengths[1])

            # Compute train loss for the whole epoch
            mean_train_loss = np.average(train_losses, weights=n_docs_per_batch)
            new_row_dict = {
                'epoch': self.epoch,
                'split': split,
                'model': self.method,
                'task': self.all_tasks_key ,
                'finetuning': finetuning,
                'metric': 'loss',
                'value': mean_train_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)

            # Get Validation loss
            val_loss = self.get_valloss(dataloaders["global_validation"])
            new_row_dict = {
                'epoch': self.epoch,
                'split': 'validation',
                'model': self.method,
                'task': self.all_tasks_key,
                'finetuning': finetuning,
                'metric': 'loss',
                'value': val_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)
            print(f">> Epoch {self.epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_loss:.3f} | Gradients norm -> {np.average(grad_norms, weights=grad_lengths):.6f}")
            
            metrics_payload = {"loss": float(val_loss)}
            self._save_checkpoint(epoch=self.epoch, metrics=metrics_payload)
            # Check if we stop training or not
            training = self.early_stopping.checkpoint(self.model, {'loss': val_loss}, self.epoch)

        stop_train = time.perf_counter()
        print(f"> End training at epoch {self.epoch} after {format_time_diff(start_train, stop_train)}")
        
        

    def finetune(self, dataloaders):
        """
        Fine-tuning loop on a selected task.
        """
        
        print(f"\n> Fine-tuning on task {self.selected_task}")
        self._current_stage = "finetune"

        if self.checkpoint_manager and self.resume_enabled:
            self._attempt_resume()
        else:
            self.load_model()
        assert self.epoch is not None, "You need to train the model first"
        training = True
        epoch_finetuning = max(self._resume_epoch, self.epoch)
        split = 'train'
        finetuning = True
        # Monitor the progression during the fine-tuning
        finetune_patience = EarlyStopping(
            method = self.method,
            max_patience=self.patience,
            state_dict_path=self.state_dict_path,
            state_dict_finetuned_path=self.state_dict_finetuned_path,
            model_path=self.model_path,
            model_finetuned_path=self.model_finetuned_path,
            finetuning=True
            )
        start_finetune = time.perf_counter()

        while training:
            epoch_finetuning += 1
            # Storages for the training loss and gradients norms
            train_losses = []
            n_docs_per_task = []
            grad_norms = []
            grad_lengths = []
            for task_id, (batched_input, batched_labels, _) in enumerate(tqdm(dataloaders[f"tasks_{split}"], leave=False, desc=f"Finetuning on {split} tasks")):
                # Train only on the selected task
                if task_id != self.selected_task: continue
                batch_loss = []
                n_docs_per_batch = []

                # Train on batches
                for input_data, labels in zip(batched_input, batched_labels):
                    # Send everything on device
                    input_data = input_data.to(self.device)
                    labels = labels.to(self.device)
                    # Optimization loop
                    train_loss, gradients_and_lengths = self.optimization_loop(input_data, labels)
                    # Aggregate train loss and gradient norms
                    batch_loss.append(train_loss)
                    n_docs_per_batch.append(len(input_data))
                    grad_norms.append(gradients_and_lengths[0])
                    grad_lengths.append(gradients_and_lengths[1])

                # Save the loss of the task
                train_loss = np.average(batch_loss, weights=n_docs_per_batch)
                train_losses.append(train_loss)
                n_docs_per_task.append(np.sum(n_docs_per_batch))
                new_row_dict = {
                    'epoch': epoch_finetuning,
                    'split': split,
                    'model': self.method,
                    'task': task_id,
                    'finetuning': finetuning,
                    'metric': 'loss',
                    'value': train_loss,
                }
                self.metrics_handler['training'].add_row(new_row_dict)

            # Compute train loss for the whole epoch
            mean_train_loss = np.average(train_losses, weights=n_docs_per_task)
            new_row_dict = {
                'epoch': epoch_finetuning,
                'split': split,
                'model': self.method,
                'task': self.all_tasks_key ,
                'finetuning': finetuning,
                'metric': 'loss',
                'value': mean_train_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)

            # Evaluate the performance on the validation set
            val_loss = self.get_valloss(dataloaders["global_validation"])
            new_row_dict = {
                'epoch': epoch_finetuning,
                'split': 'validation',
                'model': self.method,
                'task': self.all_tasks_key,
                'finetuning': finetuning,
                'metric': 'loss',
                'value': val_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)
            print(f">> Epoch {epoch_finetuning} | Train loss -> {mean_train_loss:.5f} | Validation loss -> {val_loss:.5f} | Gradients norm -> {np.average(grad_norms, weights=grad_lengths):.6f}")
            
            # Check if we stop fine_tuning or not
            training = finetune_patience.checkpoint(self.model, {"loss":val_loss}, epoch_finetuning)

        stop_finetune = time.perf_counter()
        print(f"> End fine-tuning at epoch {epoch_finetuning} after {format_time_diff(start_finetune, stop_finetune)}")
