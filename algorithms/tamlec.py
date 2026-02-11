import torch
import numpy as np
import nltk
from tqdm import tqdm
import time
import orjson
from models.Tamlec.tamlec import Tamlec
from models.Tamlec.prediction_handler import CustomXMLHolder, CustomXMLHolderGlobal, CustomXMLHolderCompletion
from algorithms.base_algorithm import AbstractAlgorithm
from misc.utils.training import AdaptivePatience
from misc.utils.time import format_time_diff
from pathlib import Path
import gc
from typing import  Dict, Optional, Any
import random
from misc.utils.training import EarlyStopping

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


class TamlecAlg(AbstractAlgorithm):
    """Tamlec algorithm wrapper with task-aware training and completion mode."""
    def __init__(
        self,
        metrics_handler,
        src_vocab,
        trg_vocab,
        taxonomy,
        abstract_dict,
        taxos_tamlec,
        width_adaptive,
        decoder_adaptative,
        tasks_size,
        with_bias,
        accum_iter,
        loss_smoothing,
        seq_length,
        freeze,
        k_list_eval_perf,
        label_to_task,
        len_tasks_validation,
        fewshot_exp,
        task_to_subroot,
        method: str = "", # Identifier of the model/method (e.g., "lightxml"). 
        state_dict_path: str = "", # Path to a state dict to load the model from 
        state_dict_finetuned_path: str = "", # Path to a state dict to load the finetuned model from 
        model_path: Path = Path(), # Path to a model to load the model from 
        model_finetuned_path: str = "", # Path to a model to load the finetuned model from 
        patience: int = 10, # Maximum number of epochs without improvement before early stopping 
        all_tasks_key: str = "", # Key used to store global (all-tasks 
        selected_task: int = 0, # Task index to fine-tune if few-shot mode is enabled 
        learning_rate: float = 5e-5, # Learning rate for the optimizer
        device: str = "cpu", # Device to use (cpu, cuda:0, ...
        verbose = True,
        checkpoint_manager = None,
        resume: bool = True,
        min_improvement: float = .0,
        max_freeze_epochs: int = 50,
        **kwargs
        ): 
        """Initialize Tamlec model, training configuration, and checkpoints."""
        self.verbose = verbose
        self.checkpoint_manager = checkpoint_manager
        self.resume_enabled = resume
        self._current_stage = "train"
        self._resumed = False
        self._resume_epoch = 0
        self.max_freeze_epochs = max_freeze_epochs
        self.k_list_eval_perf = k_list_eval_perf
        self.epoch = None
        self.task_to_subroot = task_to_subroot
        # Download from nltk
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger_eng')
        self.freeze = freeze
        self.taxonomy = taxonomy
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = torch.device(f"cuda:{device}") if torch.cuda.is_available() else torch.device("cpu")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device.index)
        self.method = method
        self.patience = patience
        self.state_dict_path = state_dict_path
        self.state_dict_finetuned_path = state_dict_finetuned_path
        self.model_path = model_path
        self.model_finetuned_path = model_finetuned_path
        self.all_tasks_key = all_tasks_key
        self.fewshot_exp = fewshot_exp
        self.selected_task = selected_task
        self.label_to_task = label_to_task
        print(f"> Loading model...")
        self.model = Tamlec(
            src_vocab=src_vocab,
            tgt_vocab=trg_vocab,
            #path_to_glove=".vector_cache/glove.840B.300d.gensim",
            abstract_dict=abstract_dict,
            taxonomies=taxos_tamlec,
            width_adaptive=width_adaptive,
            decoder_adaptative=decoder_adaptative,
            tasks_size=tasks_size,
            device=self.device,
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
        )
        self.len_tasks_validation = len_tasks_validation
        if freeze:
            self.early_stopping = AdaptivePatience(
                method = self.method,
                max_patience=self.patience,
                state_dict_path= self.state_dict_path,
                state_dict_finetuned_path = self.state_dict_finetuned_path,
                model_path= self.model_path,
                model_finetuned_path= self.model_finetuned_path,                
                n_tasks=self.len_tasks_validation,
                all_tasks_key=self.all_tasks_key,
                min_improvement = min_improvement
                )
        else:
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

    def run_init(self):
        """No-op initialization hook for Tamlec."""
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
        """Ensure Tamlec model tensors are on the configured device."""
        model.device = self.device
        model._gpu_target = None if self.device.type == "cpu" else self.device
        if hasattr(model, "_model"):
            model._model = model._model.to(self.device)
        if hasattr(model, "criterion"):
            model.criterion = model.criterion.to(self.device)
        return model

    def load_model(self):
        """Load a saved Tamlec model from disk."""
        with open(self.model_path, "rb") as f:
            model = torch.load(f, map_location=self.device)
        self.model = self._set_model_device(model)
    
    def load_model_finetuned(self):
        """Load a finetuned Tamlec model from disk."""
        with open(self.model_finetuned_path, "rb") as f:
            model = torch.load(f, map_location=self.device)
        self.model = self._set_model_device(model)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a checkpoint payload and bind it to the active device."""
        payload = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(payload, dict):
            core = self.model._model if hasattr(self.model, "_model") else self.model
            core.load_state_dict(payload, strict=True)
            self._set_model_device(self.model)
            return

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


    def on_early_stopping_save(self, model, epoch: int, metrics: Dict[str, float], finetuning: bool = False) -> None:
        """Save the best model for early-stopping if supported."""
        if self.checkpoint_manager and hasattr(self.checkpoint_manager, "save_best"):
            stage = "finetune" if finetuning else "train"
            self.checkpoint_manager.save_best(
                epoch=epoch,
                stage=stage,
                metrics=metrics,
                model_saver=lambda path: self.save_model(path, is_finetune=finetuning),
            )
        else:
            self.save_model(is_finetune=finetuning)
    
    def save_model(self, path: Optional[Path] = None, *, is_finetune: bool = False) -> None:
        """Save Tamlec model weights to disk."""
        target = Path(path) if path is not None else (
            Path(self.state_dict_finetuned_path) if is_finetune else Path(self.state_dict_path)
        )

        core = self.model._model if hasattr(self.model, "_model") else self.model
        torch.save({k: v.detach().cpu() for k, v in core.state_dict().items()}, target)

        
    
    # To save memory, do no compute the gradients since we do not need them here
    @torch.no_grad()
    def get_valloss(self, dataloaders):
        """Compute validation loss across tasks without gradient tracking."""
        self.model.eval()

        loss_overall = []
        precision_overall = []
        n_docs_per_task = []

        for _enum_task_id, (doc_files, column_indices, task_id2, batches_indices) in enumerate(
            tqdm(dataloaders["tasks_validation"], leave=False)
        ):
            
            losses = []
            precisions = []
            n_docs_per_batch = []

            for batch in batches_indices:
                toks, label_batch = [], []
                for j in batch:
                    with open(doc_files[j], 'rb') as f:
                        text_tokenized, label = orjson.loads(f.read())
                    toks.append(torch.tensor(text_tokenized, dtype=torch.int32))
                    label_batch.append(torch.tensor(label, dtype=torch.int32))

                input_data = torch.vstack(toks)
                input_labels = label_batch
                loss, prec = self.model.eval_batch(
                    documents_tokens=input_data,
                    labels=input_labels,
                    task_id=task_id2,
                )
                try:
                    loss = loss.cpu().item()
                except:
                    pass

                losses.append(loss)
                precisions.append(prec)
                n_docs_per_batch.append(len(input_data))

            loss_overall.append(np.average(losses, weights=n_docs_per_batch))
            precision_overall.append(np.average(precisions, weights=n_docs_per_batch))
            n_docs_per_task.append(np.sum(n_docs_per_batch))

        print("Precision @1 For Validation : ", np.average(precision_overall, weights=n_docs_per_task))
        

        return loss_overall, np.average(loss_overall, weights=n_docs_per_task)
       
    def inference_eval(self, input_data, labels, task_id = None):
        """Run evaluation inference, optionally constrained to a task."""
        batched_labels = labels
        if task_id is None:
            labels = []
            task_id_list = []
            for document_labels in batched_labels:
                labels.append(document_labels)
                # All tasks in which the document appears
                tasks_set = set()
                for lab in document_labels:
                    
                    # Main root does not belong to any task, skip it
                    if lab == 0: continue
                    tasks_set = tasks_set.union(self.label_to_task[lab])
                task_id_list.append(list(tasks_set))
            xml = CustomXMLHolderGlobal(text_batch=input_data, task_id_list=task_id_list, beam_parameter=10, tamlec=self.model, proba_operator="MAX_PROBA")
            predictions, scores = xml.run_predictions(batch_size=256)
        else:
            # Compute predictions
            xml = CustomXMLHolder(text_batch=input_data, task_id=task_id, beam_parameter=10, tamlec=self.model, proba_operator="MAX_PROBA")
            predictions, scores = xml.run_predictions(batch_size=256)
            
        # Transform predictions into one-hot encoding
        all_predictions = []
        all_complete_labels = []
        for pred, score in zip(tqdm(predictions), scores):
            assert len(pred) == len(score)
            pred = torch.tensor(pred)
            score = torch.tensor(score)
            # +1 as we have a <PAD> token for the labels
            pred_sample = torch.zeros(self.taxonomy.n_nodes+1, dtype=torch.float32)
            pred_sample.scatter_(dim=0, index=pred, src=score.to(torch.float32))
            assert pred_sample[-1] == 0.
            all_predictions.append(pred_sample)
            # +1 as we have a <PAD> token for the labels
        
        # Aggregate predictions 
        all_predictions = torch.stack(all_predictions, dim=0).to(torch.float32).to(self.device) 
        return all_predictions[:,1:]
    

    def inference_eval_completion(self, input_data, labels, task_id):
        """Run completion-mode inference for a given task."""
        self.model.eval()
        predictions, ground_truth = self.model.completion_batch(documents_tokens=input_data, labels=labels, task_id=task_id)
        
        return predictions, ground_truth
        
    
    
    def train(self,dataloaders):
        """Train Tamlec over tasks, with optional freeze stage."""
        if self.checkpoint_manager and self.resume_enabled:
            # Here the current_stage and epoch are also restored
            self._attempt_resume()
            
        
        # 1. Train the model on the training set
        print(f"\n> Training the model")
        training = True
        self.epoch = self._resume_epoch
        split = 'train'
        finetuning = False
        start_train = time.perf_counter()
        val_loss_per_task, val_loss = self.get_valloss(dataloaders) # TO REMOVE AFTER DEBUG
        #self._current_stage = "freeze"
        if self._current_stage == "train": # Check if we are resuming from training stage, othwerwise skip to task finetuning
            while training :  
                self.epoch += 1
                train_losses = []
                n_docs_per_task = []
                # Model in training mode, e.g. activate dropout if specified
                self.model.train() 
                print(f"\n>> Epoch {self.epoch} -- Training Stage")
                for task_id, (doc_files,  column_indices, task_id2, batches_indices) in enumerate(tqdm(dataloaders[f"tasks_{split}"], leave=False)):
                    if self.fewshot_exp and task_id == self.selected_task:
                        continue

                    
                    

                    batch_loss = []
                    n_docs_per_batch = []

                    for batch in batches_indices:
                        toks = []
                        label_batch = []

                        #tqdm.write("loading batch documents...")
                        for j in batch:
                            with open(doc_files[j], 'rb') as f:
                                text_tokenized, label = orjson.loads(f.read())
                            toks.append(torch.tensor(text_tokenized, dtype=torch.int32))
                            label_batch.append(torch.tensor(label, dtype=torch.int32))

                        input_data = torch.vstack(toks)
                        input_labels = label_batch

                        loss = self.model.train_on_batch(
                            documents_tokens=input_data,
                            labels=input_labels,
                            task_id=task_id2,
                        )
                        batch_loss.append(loss.cpu().item())
                        n_docs_per_batch.append(len(input_data))

                    train_loss = np.average(batch_loss, weights=n_docs_per_batch)

                    train_losses.append(train_loss)
                    n_docs_per_task.append(np.sum(n_docs_per_batch))
                    new_row_dict = {
                        'epoch': self.epoch,
                        'split': split,
                        'model': self.method,
                        'task': task_id2,
                        'finetuning': finetuning,
                        'metric': 'loss',
                        'value': train_loss,
                    }
                    self.metrics_handler['training'].add_row(new_row_dict)

                # Compute train loss for the whole epoch
                mean_train_loss = np.average(train_losses, weights=n_docs_per_task)
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

                print(f"\n>> Epoch {self.epoch} -- Validation Stage")
                val_loss_per_task, val_loss = self.get_valloss(dataloaders)
                # Evaluate the performance on the validation set
                print(f">> Epoch {self.epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_loss:.3f}")

                metrics_payload = {"loss": float(val_loss)}
                self._save_checkpoint(epoch=self.epoch, metrics=metrics_payload)
                
                # Check if we stop training or not
                if self.freeze:
                    training = self.early_stopping.global_checkpoint(self.model, val_loss_per_task, self.epoch, freezing=False)  # type: ignore
                else:
                    training = self.early_stopping.checkpoint(self.model, {"loss":val_loss} , self.epoch)  # type: ignore
        print(f"\n> Finished training stage, switching to freeze mode")
        self._current_stage = "freeze"
        # If in freeze mode, second part is to train all tasks that have not converged yet
        if self.freeze:
            # In fewshot experiment anyway mark that the selected task has converged
            if self.fewshot_exp: self.early_stopping.mark_task_done(self.selected_task)  # type: ignore
            if len(self.early_stopping.tasks_to_complete) == 0:  # type: ignore
                print(f"\n All tasks already converged")
                training = False
            else:
                print(f"\n> Freeze shared parameters and continue training on {len(self.early_stopping.tasks_to_complete)} tasks")  # type: ignore
                training = True
                
            self.early_stopping.reset()  
            # Freeze parameters shared in all tasks
            self.model.freeze()  

            self.max_freeze_epochs = self.max_freeze_epochs + self.epoch
            
            while training:
                if self.epoch >= self.max_freeze_epochs:
                    print(f"> Reached maximum number of epochs in freeze mode ({self.max_freeze_epochs}). Stopping training.")
                    break
                self.epoch += 1
                train_losses = []
                n_docs_per_task = []
                # Model in training mode, e.g. activate dropout if specified
                self.model.train() 
                for task_id, (doc_files,  column_indices, task_id2, batches_indices) in enumerate(tqdm(dataloaders[f"tasks_{split}"], leave=False)):
                    if self.fewshot_exp and task_id == self.selected_task:
                        continue

                    assert task_id2==task_id, "Task id mismatch {task_id2} vs {task_id}"

                    if task_id  in self.early_stopping.tasks_completed:
                        continue  # type: ignore


                    batch_loss = []
                    n_docs_per_batch = []

                    for batch in batches_indices:
                        toks = []
                        label_batch = []
                        

                        #tqdm.write("loading batch documents...")
                        for j in batch:
                            
                            with open(doc_files[j], 'rb') as f:
                                text_tokenized, label = orjson.loads(f.read())
                            toks.append(torch.tensor(text_tokenized, dtype=torch.int32))
                            label_batch.append(torch.tensor(label, dtype=torch.int32))

                        input_data = torch.vstack(toks)
                        input_labels = label_batch

                        loss = self.model.train_on_batch(
                            documents_tokens=input_data,
                            labels=input_labels,
                            task_id=task_id2,
                        )
                        batch_loss.append(loss.cpu().item())
                        n_docs_per_batch.append(len(input_data))

                    train_loss = np.average(batch_loss, weights=n_docs_per_batch)
                    train_losses.append(train_loss)
                    n_docs_per_task.append(np.sum(n_docs_per_batch))
                    new_row_dict = {
                        'epoch': self.epoch,
                        'split': split,
                        'model': self.method,
                        'task': task_id2,
                        'finetuning': finetuning,
                        'metric': 'loss',
                        'value': train_loss,
                    }
                    self.metrics_handler['training'].add_row(new_row_dict)

                # Compute train loss for the whole epoch
                mean_train_loss = np.average(train_losses, weights=n_docs_per_task)
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
                val_loss_per_task, val_loss = self.get_valloss(dataloaders)
                print(f">> Epoch {self.epoch} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_loss:.3f}")
                
                metrics_payload = {"loss": float(val_loss)}
                self._save_checkpoint(epoch=self.epoch, metrics=metrics_payload)
                
                # Check if we stop training or not
                self.early_stopping.global_checkpoint(self.model, val_loss_per_task, self.epoch, freezing=True)  # type: ignore
                # Anyway save the model since only specific weights are updated

                if len(self.early_stopping.tasks_to_complete) == 0:  # type: ignore
                    print(f"\n All tasks already converged")
                    training = False
                else:
                    print(f"\n> Freeze shared parameters and continue training on {len(self.early_stopping.tasks_to_complete)} tasks")  # type: ignore
                    training = True

                if finetuning:
                    torch.save(self.model, self.model_finetuned_path)
                else:
                    torch.save(self.model, self.model_path)
               

        stop_train = time.perf_counter()
        print(f"> End training at epoch {self.epoch} after {format_time_diff(start_train, stop_train)}")
        
        _free_cuda_model(self.model)
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        
    def finetune(self, dataloaders):
        """Fine-tune Tamlec on the selected task."""
        # 2. In few-shot experiment, fine-tune the task that has not been trained
        print(f"\n> Fine-tuning on task {self.selected_task}")
        self._current_stage = "finetune"

        if self.checkpoint_manager and self.resume_enabled:
            self._attempt_resume()
        else:
            self.load_model()
        assert self.epoch is not None, "The mdoel needs to be trained first"
        assert self.fewshot_exp, "Model needs to be trained with fewshot_exp = True first and keep fewshot_exp = True"

        training = True
        epoch_finetuning = self.epoch
        split = 'train'
        finetuning = True
        finetune_patience = EarlyStopping(
            method = self.method,
            max_patience=self.patience,
            state_dict_path= self.state_dict_path,
            state_dict_finetuned_path= self.state_dict_finetuned_path,
            model_path = self.model_path,
            model_finetuned_path= self.model_finetuned_path,
            finetuning=True,
            min_improvement = .01
            )
        start_finetune = time.perf_counter()

        while training:
            epoch_finetuning += 1
            # Loss for all tasks, and number of documents per task for weighted average
            train_losses = []
            n_docs_per_task = []
            # Model in training mode, e.g. activate dropout if specified
            self.model.train() 
            for _enum_task_id, (doc_files,  column_indices, task_id2, batches_indices) in enumerate(
                tqdm(dataloaders[f"tasks_{split}"], leave=False)
            ):
                if task_id2 != self.selected_task:
                    continue

                batch_loss = []
                n_docs_per_batch = []

                for batch in batches_indices:
                    toks, labels = [], []
                    for j in batch:
                        
                        with open(doc_files[j], 'rb') as f:
                            text_tokenized, _label = orjson.loads(f.read())
                        toks.append(torch.tensor(text_tokenized, dtype=torch.int32))
                        labels.append(_label)

                    input_data = torch.vstack(toks)

                    loss = self.model.train_on_batch(
                        documents_tokens=input_data,
                        labels=labels,
                        task_id=task_id2,
                    )
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
                    'task': task_id2,
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
            val_loss_per_task, val_loss = self.get_valloss(dataloaders)
            print(f">> Epoch {epoch_finetuning} | Train loss -> {mean_train_loss:.3f} | Validation loss -> {val_loss:.3f}")

            # Check if we stop fine_tuning or not
            training = finetune_patience.checkpoint(self.model, {"loss":val_loss}, epoch_finetuning)

        stop_finetune = time.perf_counter()
        print(f"> End fine-tuning at epoch {epoch_finetuning} after {format_time_diff(start_finetune, stop_finetune)}")
