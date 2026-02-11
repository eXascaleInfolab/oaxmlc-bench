import random
import torch
import numpy as np
import nltk
from tqdm import tqdm
import time
import orjson

from algorithms.base_algorithm import AbstractAlgorithm
from models.Hector.hector import Hector
from models.Hector.prediction_handler import CustomXMLHolder, CustomXMLHolderCompletion
from pathlib import Path
import gc
from misc.utils.time import format_time_diff
from misc.utils.training import EarlyStopping
from misc.checkpointing import CheckpointManager

from typing import  Dict, Iterable, Optional, Any


def _free_cuda_model(m):
    """Release CUDA resources held by a Tamlec model instance."""
    # if your Tamlec stores an optimizer on self (e.g., self.optimizer),
    # drop it explicitly to free its CUDA tensors
    if hasattr(m, 'optimizer'):
        m.optimizer = None
    # move off GPU to decouple CUDA tensors
    try:
        m.to('cpu')
    except Exception:
        pass
    # remove references and flush
    del m
    gc.collect()
    torch.cuda.empty_cache()

class HectorAlg(AbstractAlgorithm):
    """Hector algorithm wrapper with custom prediction handlers."""
    def __init__(
        self,  
        metrics_handler,
        src_vocab,
        trg_vocab,
        taxonomy,
        abstract_dict,
        taxos_hector,
        with_bias,
        accum_iter,
        loss_smoothing,
        seq_length,
        k_list_eval_perf,
        batch_size_eval,
        beam_parameter,
        proba_operator,
        loss_function: torch.nn.Module, # Any torch loss function (e.g. torch.nn.BCELoss()
        method: str = "", # Identifier of the model/method (e.g., "lightxml").
        state_dict_path: str = "", # Path to a state dict to load the model from
        state_dict_finetuned_path: str = "", # Path to a state dict to load the finetuned model from
        model_path: Path = Path(), # Path to a model to load the model from
        model_finetuned_path: str = "", # Path to a model to load the finetuned model from
        patience: int = 10, # Maximum number of epochs without improvement before early stopping
        all_tasks_key: str = "", # Key used to store global (all-tasks
        selected_task: int = 0, # Task index to fine-tune if few-shot mode is enabled
        learning_rate: float = 5e-5, # Learning rate for the optimizer
        device: str = ' cpu', # Device to use (cpu, cuda:0,
        verbose=True,
        checkpoint_manager: Optional[CheckpointManager] = None,
        resume: bool = True,
        min_improvement: float = .0,
        **kwargs        
        ): 
        """Initialize Hector model and training configuration."""
        self.checkpoint_manager = checkpoint_manager
        self.resume_enabled = resume
        self._current_stage = "train"
        self._resumed = False
        self._resume_epoch = 0
        
        self.verbose = verbose
        self.epoch = None
        self.device = torch.device(device)
        self.k_list_eval_perf = k_list_eval_perf
        self.method = method
        self.all_tasks_key = all_tasks_key
        self.model_path = model_path
        self.model_finetuned_path = model_finetuned_path
        self.state_dict_path = state_dict_path
        self.state_dict_finetuned_path = state_dict_finetuned_path
        self.selected_task = selected_task
        self.patience = patience
        self.batch_size_eval = batch_size_eval
        self.loss_function = loss_function
        self.beam_parameter = beam_parameter
        self.proba_operator = proba_operator
        # Download from nltk
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        # Device selection (Hector expects a cuda index string like "cuda:0")
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)
        self.taxonomy = taxonomy
        
        print(f"> Loading model...")
        self.model = Hector(
            src_vocab=src_vocab,
            tgt_vocab=trg_vocab,
            #path_to_glove=".vector_cache/glove.840B.300d.gensim",
            abstract_dict=abstract_dict,
            taxonomies=taxos_hector,
            with_bias=with_bias,
            Number_src_blocs=6,
            Number_tgt_blocs=6,
            dim_src_embedding=300,
            dim_tgt_embedding=600,
            dim_feed_forward=2048,
            number_of_heads=12,
            dropout=0.1,
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.99,
            epsilon=1e-8,
            weight_decay=0.01,
            gamma=.99998,
            accum_iter=accum_iter,
            # 0.0 < x <= 0.1
            loss_smoothing=loss_smoothing,
            max_padding_document=seq_length,
            max_number_of_labels=20,
            device=self.device,
            **kwargs
        )
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
        self.metrics_handler = metrics_handler

    def run_init(self) -> None:
        """
        Any initialization required before training or evaluation.
        """
        pass

    def _attempt_resume(self):
        """Attempt to resume training from the latest checkpoint."""
        record = None
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
    def _set_model_device(self, model):
        """Ensure Hector model tensors are on the configured device."""
        model.device = self.device
        model._gpu_target = None if self.device.type == "cpu" else self.device
        if hasattr(model, "_model"):
            model._model = model._model.to(self.device)
        if hasattr(model, "criterion"):
            model.criterion = model.criterion.to(self.device)
        return model

    def load_model(self):
        """Load a saved Hector model from disk."""
        with open(self.model_path, "rb") as f:
            model = torch.load(f, map_location=self.device)
        self.model = self._set_model_device(model)
    
    def load_model_finetuned(self):
        """Load a finetuned Hector model from disk."""
        with open(self.model_finetuned_path, "rb") as f:
            model = torch.load(f, map_location=self.device)
        self.model = self._set_model_device(model)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a checkpoint payload and bind it to the active device."""
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model = self._set_model_device(payload)
        
    def capture_training_state(self) -> Dict[str, Any]:
        """Capture RNG and epoch state for resuming training."""
        state: Dict[str, Any] = {}

        state["rng_state"] = torch.get_rng_state()
        state["numpy_state"] = np.random.get_state()
        state["python_state"] = random.getstate()
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        state["epoch"] = self.epoch or 0
        state["stage"] = self._current_stage
        return state

    def restore_training_state(self, state: Dict[str, Any]) -> None:
        """Restore RNG and epoch state from a checkpoint payload."""
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
    
    def save_model(self, path: Optional[Path] = None, *, is_finetune: bool = False) -> None:
        """Save the Hector model to disk."""
        target = Path(path) if path is not None else (
                Path(self.model_finetuned_path) if is_finetune else Path(self.model_path)
            )
        torch.save(self.model, target)
    
    @torch.no_grad()
    def get_valloss(self, val_loader):
        """
        Mean validation loss on the *global* loader.
        Expects `val_loader` batches shaped like collate_global_int_labels:
        (input_data, labels_per_sample, relevant_classes)
        Uses self.inference_eval(...) -> [B, L] and self.loss_function.
        """
        self.model.eval()

        total_loss = 0.0
        total_docs = 0
        total_precision = 0.0
        L = self.taxonomy.n_nodes
        loss_fn = self.loss_function

        for input_data, labels, _ in val_loader:
            # Predictions: [B, L] on model device
            preds = self.inference_eval(input_data)      # device-safe inside
            B = preds.size(0)

            # Dense targets on same device
            target = torch.zeros(B, L, dtype=preds.dtype, device=preds.device)

            # Densify labels safely (handle scalars, lists, tensors, PAD/out-of-range)
            for i, idx in enumerate(labels):
                idx = torch.as_tensor(idx, dtype=torch.long, device=preds.device)
                if idx.dim() == 0:
                    idx = idx.unsqueeze(0)
                if idx.numel() == 0:
                    continue
                valid = (idx >= 0) & (idx < L)
                if valid.any():
                    idx = idx[valid].unique()
                    # use src matching idx shape (robust across torch versions)
                    src = torch.ones(idx.shape, dtype=target.dtype, device=target.device)
                    target[i].scatter_(0, idx, src)

            # Compute batch loss
            # NOTE: If your loss is BCEWithLogitsLoss and preds are probabilities,
            # either switch to BCE loss or pass logits here. Keeping as-is to match your eval_step usage.
            batch_loss = loss_fn(preds, target).item()

            total_loss += batch_loss * B
            total_docs += B

        return total_loss / max(total_docs, 1)
    

    def inference_eval_completion(self, input_data, paths_and_children):
        """
        This method is called when running the completion experiment, Hector will exploit the first level label to 
        infer the next labels in the taxonomy.
        
        :param input_data: batch tensor of tokenizes documents
        :param paths_and_children: batch of paths corresponding to the documents provided in input_data
        """
        self.model.eval()
        predictions, ground_truth = self.model.completion_batch(documents_tokens=input_data, paths_and_children=paths_and_children)
        return predictions, ground_truth

    
    def train(self,dataloaders):
        """Train Hector on the global dataset."""
        
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
            train_losses = []
            n_docs_per_batch = []
            # Model in training mode, e.g. activate dropout if specified

            self.model.train() 
            for input_data, labs, _ in tqdm(dataloaders[f"global_{split}"], leave=False):
                # Train on batch
                # For hector always set task_id=0 as the whole taxonomy is taken into account
                loss = self.model.train_on_batch(documents_tokens=input_data, labels=labs)
                train_losses.append(loss.cpu().item())
                n_docs_per_batch.append(len(input_data))

            # Compute train loss for the whole epoch
            mean_train_loss = np.average(train_losses, weights=n_docs_per_batch)
            new_row_dict = {
                'epoch': self.epoch,
                'split': split,
                'model': self.method,
                'task': self.all_tasks_key,
                'finetuning': finetuning,
                'metric': 'loss',
                'value': mean_train_loss,
            }
            self.metrics_handler['training'].add_row(new_row_dict)

            # Evaluate the performance on the validation set
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
            print(f">> Epoch {self.epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_loss:.3f}")    
            
            metrics_payload = {"loss": float(val_loss)}
            self._save_checkpoint(epoch=self.epoch, metrics=metrics_payload)
            # Check if we stop training or not
            training = self.early_stopping.checkpoint(self.model, {"loss":val_loss}, self.epoch)

        stop_train = time.perf_counter()
        print(f"> End training at epoch {self.epoch} after {format_time_diff(start_train, stop_train)}")

    def finetune(self, dataloaders):
        """Fine-tune Hector on the selected task."""
        assert self.epoch is not None, "You should train the model first"
        
        print(f"\n> Fine-tuning on task {self.selected_task}")
        self._current_stage = "finetune"
        
        if self.checkpoint_manager and self.resume_enabled:
            self._attempt_resume()
        else:
            self.load_model()
        
        
        # Reload best model
        with open(self.model_path, "rb") as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.get_optimizer()

        training = True
        epoch_finetuning = max(self._resume_epoch, self.epoch)
        split = 'train'
        finetuning = True
        finetune_patience = EarlyStopping(
            method = self.method,
            max_patience = self.patience,
            state_dict_path = self.state_dict_path,
            state_dict_finetuned_path= self.state_dict_finetuned_path,
            model_path = self.model_path,
            model_finetuned_path = self.model_finetuned_path,
            finetuning=True
            )
        start_finetune = time.perf_counter()

        while training:
            epoch_finetuning += 1
            # Loss for all tasks, and number of documents per task for weighted average
            train_losses = []
            n_docs_per_task = []
            # Model in training mode, e.g. activate dropout if specified
            self.model.train()
            for task_id, (batched_input, batched_labels, _,  batched_paths_and_children) in enumerate(tqdm(dataloaders[f"tasks_{split}"], leave=False)):
                # Train only on the selected task
                if task_id != self.selected_task: continue
                batch_loss = []
                n_docs_per_batch = []

                # Train on batches
                for input_data, labels, paths_and_children in zip(batched_input, batched_labels, batched_paths_and_children):
                    loss = self.model.train_on_batch(documents_tokens=input_data, paths_and_children=paths_and_children)
                    batch_loss.append(loss.cpu().item())
                    n_docs_per_batch.append(len(input_data))

                # Save the loss
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
                'task': self.all_tasks_key,
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
            
            print(f">> Epoch {epoch_finetuning} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_loss:.3f}")

            # Check if we stop fine_tuning or not
            training = finetune_patience.checkpoint(self.model, {"loss":val_loss}, epoch_finetuning)

        stop_finetune = time.perf_counter()
        print(f"> End fine-tuning at epoch {epoch_finetuning} after {format_time_diff(start_finetune, stop_finetune)}")


    @torch.no_grad()
    def inference_eval(self, input_data, **kwargs):
        """
        This method is called during evaluation, it performs a forward pass over the documents and returns the
        output scores.
        
        :param input_data: tensor batch of tokenized input documents
        """
        self.model.eval() 
        device = self.device

        # normalize input to [B, seq_len] long on model device
        if isinstance(input_data, (list, tuple)):
            input_data = torch.vstack([
                t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
                for t in input_data
            ])
        elif isinstance(input_data, torch.Tensor) and input_data.dim() == 3:
            input_data = input_data.reshape(-1, input_data.size(-1))
        elif not isinstance(input_data, torch.Tensor):
            input_data = torch.as_tensor(input_data)
        if input_data.dtype != torch.long:
            input_data = input_data.to(torch.long)
        input_data = input_data.to(device, non_blocking=True)

        beam = self.beam_parameter
        proba_operator = self.proba_operator
        batch_size = getattr(self, "batch_size_eval", 256)

        xml = CustomXMLHolder(
            text_batch=input_data,
            beam_parameter=beam,
            hector=self.model,
            proba_operator=proba_operator,
        )
        pred_lists, score_lists = xml.run_predictions(batch_size=batch_size)

        B = len(pred_lists)
        L = self.taxonomy.n_nodes
        out = torch.zeros(B, L, dtype=torch.float32, device="cpu")

        for i, (pred_ids, scores) in enumerate(zip(pred_lists, score_lists)):
            if not pred_ids:
                continue
            idx = torch.as_tensor(pred_ids, dtype=torch.long, device="cpu")
            val = torch.as_tensor(scores, dtype=torch.float32, device="cpu")
            mask = (idx >= 0) & (idx < L)
            if mask.any():
                out[i].scatter_(0, idx[mask], val[mask])

        return out.to(device)
