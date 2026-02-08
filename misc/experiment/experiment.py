import time
import torch
from pathlib import Path
from misc.utils.time import print_time, format_time_diff, print_config
from misc.utils.metrics_handler import MetricsHandler
from misc.cfg_factory import build_cfg_from_factory, _make_dirs, _clean_up_assert
from misc.checkpointing import CheckpointManager
from misc.metrics.npz import (
    stream_metrics_from_npz_and_xy,
    stream_completion_metrics_per_level_from_npz_and_xy,
    stream_completion_metrics_from_npz_and_xy,
    stream_metrics_per_level_from_npz_and_xy,
    stream_metrics_per_task_from_taxos_tamlec
)
from misc.experiment.registry import SUPPORTED_METHODS, get_method_spec, build_base_alg_args
from misc.experiment.eval import (
    eval_step as _eval_step_impl,
    eval_step_completion as _eval_step_completion_impl,
)
import os
from datetime import datetime
import sys

def in_ipython():
    try:
        from IPython.core.getipython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

    
if __name__ == '__main__':
    print("You should not call this directly, see inside the `configs` folder.")
    import sys
    
    sys.exit()

# Set max number of threads used for one experiment
n_threads = 4
torch.set_num_threads(min(torch.get_num_threads(), n_threads))
torch.set_num_interop_threads(min(torch.get_num_interop_threads(), n_threads))



class Experiment:
    '''
    Orchestrates training, evaluation, and logging for an experiment run.

    Builds configuration, prepares data/algorithm components, and provides
    entry points for training, finetuning, and evaluation workflows.
    '''
    
    def __init__(self, cfg):  
        '''
        Initialize experiment state, logging, and metrics handlers.

        :param cfg: Base configuration dictionary. Will be expanded and
            normalized via the config factory.
        :return: None.
        '''  
        self.cfg = cfg
        assert self.cfg['method'] in SUPPORTED_METHODS, (
            f"{self.cfg['method']} not available, choose in {sorted(SUPPORTED_METHODS)}"
        )
        
        
        self.cfg = build_cfg_from_factory(self.cfg)
        _make_dirs(self.cfg)
        _clean_up_assert(self.cfg)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if not in_ipython():
            from brutelogger import BruteLogger
            base_log = self.cfg.get("console_log_fname", "all_console.log")
            stem, ext = os.path.splitext(base_log)
            ext = ext or ".log"
            
            log_name = f"{stem}_{timestamp}{ext}"
            log_name = self._unique_path_if_needed(Path(self.cfg["paths"]["output"]), log_name)
            BruteLogger.save_stdout_to_file(
                path=self.cfg['paths']['output'],
                fname=log_name
            )
        
        enc = getattr(sys.stdout, "encoding", None)
        if not isinstance(enc, str):
            try:
                setattr(sys.stdout, "encoding", "utf-8")
            except Exception:
                pass
        try:
            import sympy  # noqa: F401
        except Exception as e:
            print(f"Warning: SymPy import failed during setup: {e}")
        
        stem, ext = os.path.splitext(self.cfg['paths']['metrics'])
        metrics_path = f"{stem}_{timestamp}{ext}"
        self.cfg['paths']['metrics'] = metrics_path
        
        stem, ext = os.path.splitext(self.cfg['paths']['validation_metrics'])
        val_metrics_path = f"{stem}_{timestamp}{ext}"
        self.cfg['paths']['validation_metrics'] = val_metrics_path
        
        stem, ext = os.path.splitext(self.cfg['paths']['test_metrics'])
        test_metrics_path = f"{stem}_{timestamp}{ext}"
        self.cfg['paths']['test_metrics'] = test_metrics_path
        
        self.metrics_handler = {
                # Metrics during the training/fine-tuning part
                'training': MetricsHandler(
                    columns=['epoch', 'split', 'model', 'task', 'finetuning', 'metric', 'value'], 
                    output_path=self.cfg['paths']['metrics']
                ),
                # Evaluation on validation and test sets
                'eval_validation': MetricsHandler(
                    columns=['model', 'task', 'finetuning', 'metric', 'value'],
                    output_path=self.cfg['paths']['validation_metrics']
                ),
                'eval_test': MetricsHandler(
                    columns=['model', 'task', 'finetuning', 'metric', 'value'],
                    output_path=self.cfg['paths']['test_metrics']
                ),
                # Evaluation on training set
                'eval_train': MetricsHandler(
                    columns=['model', 'task', 'finetuning', 'metric', 'value'],
                    output_path=self.cfg['paths']['train_metrics']
                ),
            }
        
        
        
        self._setup_method()

    def _setup_method(self):
        '''
        Resolve method spec and initialize algorithm/data module factories.

        :return: None.
        '''
        spec = get_method_spec(self.cfg["method"])
        if spec.pre_setup:
            spec.pre_setup(self.cfg)
        if spec.init_message:
            print(spec.init_message)

        self.algclass = spec.algclass_factory()
        self.dataclass = spec.dataclass_factory()
        self.one_hot_labels = spec.one_hot_labels
        self.dmodule = self.dataclass(self.cfg, self.one_hot_labels, verbose=True)

        base_args = build_base_alg_args(self.cfg, self.metrics_handler)
        self.alg_args = spec.build_args(self.cfg, base_args)

    def _build_algorithm(self, dataloaders, run_init=False):
        '''
        Build the algorithm instance and attach runtime dependencies.

        :param dataloaders: Dataloader dict used to pass embeddings/metadata.
        :param run_init: If True, call algorithm.run_init() after creation.
        :return: Instantiated algorithm.
        '''
        self.alg_args["taxonomy"] = self.cfg["taxonomy"]
        if self.cfg["method"] == "tamlec":
            self.alg_args["len_tasks_validation"] = len(dataloaders["tasks_validation"])
        else:
            self.alg_args["embeddings_loader"] = dataloaders["embeddings"]

        checkpoint_mgr = CheckpointManager(
            self.cfg["paths"]["checkpoints"],
            self.cfg["checkpoint_keep_last"],
        )
        self.alg_args["checkpoint_manager"] = checkpoint_mgr
        print("selected_task", self.alg_args["selected_task"])

        algorithm = self.algclass(**self.alg_args)
        if run_init:
            algorithm.run_init()
        return algorithm

    def _eval_step(
        self,
        algorithm,
        dataloaders,
        split,
        metrics_handler,
        verbose=False,
        save_pred=False,
        eval_finetuning=False,
    ) -> dict:
        '''
        Evaluate one split and log metrics via the shared eval implementation.

        :param algorithm: Algorithm instance providing inference methods.
        :param dataloaders: Dataloader dict for the split.
        :param split: Dataset split name (e.g., "validation", "test").
        :param metrics_handler: Key into ``self.metrics_handler`` to log rows.
        :param verbose: If True, print metrics to stdout.
        :param save_pred: If True, save prediction/label batches.
        :param eval_finetuning: If True, log metrics as finetuned evaluation.
        :return: Dict mapping task identifiers to metrics dicts.
        :rtype: dict[Any, Any]
        '''
        return _eval_step_impl(
            self,
            algorithm,
            dataloaders,
            split,
            metrics_handler,
            verbose=verbose,
            save_pred=save_pred,
            eval_finetuning=eval_finetuning,
        )

    def _eval_step_completion(self, algorithm, dataloaders, split, verbose=False, save_pred=False):
        '''
        Evaluate completion metrics for a split using shared implementation.

        :param algorithm: Algorithm instance providing completion inference.
        :param dataloaders: Dataloader dict for the split.
        :param split: Dataset split name (e.g., "validation", "test").
        :param verbose: If True, print metrics to stdout.
        :param save_pred: If True, save prediction/label batches.
        :return: Metrics dict for the global completion evaluation (or {}).
        '''
        return _eval_step_completion_impl(
            self,
            algorithm,
            dataloaders,
            split,
            verbose=verbose,
            save_pred=save_pred,
        )

    def main_train_dexa(self, only_finetune = False):
        '''
        Train/evaluate DEXA-style workflows using streamed NPZ metrics.

        This path trains (unless only_finetune is True), evaluates test
        predictions from exported NPZ files, and optionally finetunes.

        :param only_finetune: If True, skip base training and only finetune.
        :return: None.
        '''
        start_time = time.perf_counter()
        print_time(f"Starting {self.cfg['method']}")
        
        # Initialize algorithm    
        algorithm = self.algclass(**self.alg_args)
        split = "test"
        mh = self.metrics_handler[f"eval_{split}"]
        # -------- per-level (needs its own handler because schema has "level") --------
        level_mh = MetricsHandler(
            columns=["model", "level", "finetuning", "metric", "value"],
            output_path=self.cfg["paths"][f"{split}_level_metrics"]  
        )
        seed = self.cfg['seed']
        print("selected_task", self.alg_args['selected_task'])
        if not only_finetune:
            algorithm.train()
            
            
            finetuning = False
            preds_path = self.cfg["paths"]["output"]/"extreme/tst_predictions_clf.npz"
            xy_path = self.cfg["paths"]["dataset"]/f"preprocessed_data/seed_{seed}/tst_X_Y.txt"
            taxonomy = self.cfg["taxonomy"]
            k_list = self.cfg['k_list_eval_perf']
            batch_size = self.cfg['batch_size_eval']
            # Compute metrics on test set
            metrics_global, n_docs = stream_metrics_from_npz_and_xy(
                pred_npz_path=preds_path,
                xy_path=xy_path,
                k_list=self.cfg['k_list_eval_perf'],
                batch_size=self.cfg['batch_size_eval'],
                threshold=0.5,
                as_probs="sigmoid",  
                device="cpu",
            )
            for metric_name, metric_value in metrics_global.items():
                mh.add_row({
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'finetuning': finetuning,
                    'metric': metric_name,
                    'value': metric_value
                })
                
            # -------- per-level --------
            
            
            metrics_per_level = stream_metrics_per_level_from_npz_and_xy(
                pred_npz_path=preds_path,
                xy_path=xy_path,
                taxonomy=taxonomy,
                k_list=k_list,
                batch_size=batch_size,
                threshold=0.5,
                as_probs="sigmoid",
                device="cpu",
            )
            for level, md in metrics_per_level.items():
                for metric_name, metric_value in md.items():
                    level_mh.add_row({
                        "model": self.cfg["method"],
                        "level": int(level),
                        "finetuning": finetuning,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })

            # -------- per-task --------
            metrics_per_task = stream_metrics_per_task_from_taxos_tamlec(
                pred_npz_path=preds_path,
                xy_path=xy_path,
                taxos_tamlec=self.cfg['tamlec_params']['taxos_tamlec'],
                k_list=self.cfg['k_list_eval_perf'],
                batch_size=self.cfg['batch_size_eval'],
                threshold=0.5,
                as_probs="sigmoid",
                device="cpu",
            )   
            for task_id, md in metrics_per_task.items():
                for metric_name, metric_value in md.items():
                    mh.add_row({
                        "model": self.cfg["method"],
                        "task": task_id,
                        "finetuning": finetuning,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })
        if self.cfg["fewshot_exp"] or only_finetune:
            assert "selected_task" in self.cfg.keys()
            algorithm.finetune()
            
            finetuning = True            
            
            preds_path = self.cfg["paths"]["output"]/"finetune/extreme/tst_predictions_clf.npz"
            xy_path = self.cfg["paths"]["dataset"]/f"preprocessed_data/seed_{seed}/tst_X_Y.txt"
            taxonomy = self.cfg["taxonomy"]
            k_list = self.cfg['k_list_eval_perf']
            batch_size = self.cfg['batch_size_eval']
            # Compute metrics on test set
            metrics_global, n_docs =stream_metrics_from_npz_and_xy(
                pred_npz_path=preds_path,
                xy_path=xy_path,
                k_list=self.cfg['k_list_eval_perf'],
                batch_size=self.cfg['batch_size_eval'],
                threshold=0.5,
                as_probs="sigmoid",  
                device="cpu",
            )
            for metric_name, metric_value in metrics_global.items():
                mh.add_row({
                    'model': self.cfg['method'],
                    'task': self.cfg['all_tasks_key'],
                    'finetuning': finetuning,
                    'metric': metric_name,
                    'value': metric_value
                })
                
            # -------- per-level (needs its own handler because schema has "level") --------
            
            
            
            metrics_per_level = stream_metrics_per_level_from_npz_and_xy(
                pred_npz_path=preds_path,
                xy_path=xy_path,
                taxonomy=taxonomy,
                k_list=k_list,
                batch_size=batch_size,
                threshold=0.5,
                as_probs="sigmoid",
                device="cpu",
            )
            for level, md in metrics_per_level.items():
                for metric_name, metric_value in md.items():
                    level_mh.add_row({
                        "model": self.cfg["method"],
                        "level": int(level),
                        "finetuning": finetuning,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })

            # -------- per-task --------
            metrics_per_task = stream_metrics_per_task_from_taxos_tamlec(
                pred_npz_path=preds_path,
                xy_path=xy_path,
                taxos_tamlec=self.cfg['tamlec_params']['taxos_tamlec'],
                k_list=self.cfg['k_list_eval_perf'],
                batch_size=self.cfg['batch_size_eval'],
                threshold=0.5,
                as_probs="sigmoid",
                device="cpu",
            )   
            for task_id, md in metrics_per_task.items():
                for metric_name, metric_value in md.items():
                    mh.add_row({
                        "model": self.cfg["method"],
                        "task": task_id,
                        "finetuning": finetuning,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })
    
    def main_train(self):
        '''
        Train the model and evaluate on validation/test splits.

        If few-shot is enabled, also finetunes and re-evaluates.

        :return: None.
        '''
        start_time = time.perf_counter()
        print_time(f"Starting {self.cfg['method']}")
        
        # Prepare data
        dataloaders = self.dmodule.get_dataloaders()
        algorithm = self._build_algorithm(dataloaders, run_init=True)
                
        algorithm.train(dataloaders)
        # Print the config so we have it in case of logging or in the terminal
        print_config(self.cfg)
        # Reload best model and evaluate it on validation and test splits
        algorithm.load_model()
        for split in ('validation', 'test'):
            self._eval_step(
                algorithm,
                dataloaders,
                split,
                verbose=True,
                metrics_handler=f"eval_{split}",
                save_pred= (split == 'test'),
                eval_finetuning=False
            )

        if self.cfg['fewshot_exp']:
            assert "selected_task" in self.cfg.keys()
            algorithm.finetune(dataloaders)            
            # Reload best model
            algorithm.load_model_finetuned()
            # Reload and evaluate the model
            for split in ('validation', 'test'):
                self._eval_step(
                    algorithm,
                    dataloaders,
                    split,
                    verbose=True,
                    metrics_handler=f"eval_{split}",
                    save_pred= (split == 'test'),
                    eval_finetuning=True
                )
        
        stop_time = time.perf_counter()
        print_time(f"Experiment ended in {format_time_diff(start_time, stop_time)}")
    
    def main_finetune(self, splits: tuple = ("train", "validation", "test"), save_preds: set = {"test"}):
        '''
        Run finetuning and evaluation only (no base training).

        :param splits: Splits to evaluate.
        :type splits: tuple
        :param save_preds: Splits for which to save prediction batches.
        :type save_preds: set
        :return: None.
        '''
        start_time = time.perf_counter()
        print_time(f"Starting {self.cfg['method']} finetuning experiment")
        
        # Prepare data
        dataloaders = self.dmodule.get_dataloaders()
        algorithm = self._build_algorithm(dataloaders, run_init=False)
        if self.cfg['fewshot_exp']:
            assert "selected_task" in self.cfg.keys()
            algorithm.finetune(dataloaders)            
            # Reload best model
            algorithm.load_model_finetuned()
            # Reload and evaluate the model
            for split in splits:
                self._eval_step(
                    algorithm,
                    dataloaders,
                    split,
                    verbose=True,
                    metrics_handler=f"eval_{split}",
                    save_pred= split in save_preds,
                    eval_finetuning=True
                )
        
        stop_time = time.perf_counter()
        print_time(f"Experiment ended in {format_time_diff(start_time, stop_time)}")
           
                
                
    # Run only the evaluation and logging of metrics for a trained model
    @torch.no_grad()
    def main_evaluate(self, splits: tuple = ("train", "validation", "test"), save_preds: set = {"test"}):
        '''
        Evaluate a trained model on the requested splits.

        :param splits: Splits to evaluate.
        :type splits: tuple
        :param save_preds: Splits for which to save prediction batches.
        :type save_preds: set
        :return: None.
        '''
        save_preds = set(save_preds)
        start_time = time.perf_counter()
        print_time(f"Starting {self.cfg['method']}")
        
        # Prepare data
        dataloaders = self.dmodule.get_dataloaders()
        algorithm = self._build_algorithm(dataloaders, run_init=True)
        
        # Print the config so we have it in case of logging or in the terminal
        print_config(self.cfg)
        # Reload best model and evaluate it on validation and test splits
        algorithm.load_model()
        for split in splits:
            self._eval_step(
                algorithm,
                dataloaders,
                split,
                verbose=True,
                metrics_handler=f"eval_{split}",
                save_pred= split in save_preds,
                eval_finetuning=False
            )

        if self.cfg['fewshot_exp']:
            assert "selected_task" in self.cfg.keys()
            # Reload finetuned model
            algorithm.load_model_finetuned()
            # Reload and evaluate the model
            for split in splits:
                self._eval_step(
                    algorithm,
                    dataloaders,
                    split,
                    verbose=True,
                    metrics_handler=f"eval_{split}",
                    save_pred= split in save_preds,
                    eval_finetuning=True
                )
        
        stop_time = time.perf_counter()
        print_time(f"Evaluation ended in {format_time_diff(start_time, stop_time)}")
        
    @torch.no_grad()
    def main_evaluate_completion(self, splits: tuple = ("train", "validation", "test"), save_preds: set = {"test"}):
        '''
        Evaluate completion metrics on the requested splits.

        :param splits: Splits to evaluate.
        :type splits: tuple
        :param save_preds: Splits for which to save prediction batches.
        :type save_preds: set
        :return: None.
        '''
        save_preds = set(save_preds)
        start_time = time.perf_counter()
        print_time(f"Starting {self.cfg['method']}")
        print()
        # Prepare data
        dataloaders = self.dmodule.get_dataloaders()
        algorithm = self._build_algorithm(dataloaders, run_init=True)
        
        # Print the config so we have it in case of logging or in the terminal
        print_config(self.cfg)
        # Reload best model and evaluate it on validation and test splits
        algorithm.load_model()
        for split in splits:
            self._eval_step_completion(
                algorithm,
                dataloaders,
                split,
                verbose=True,
                save_pred= split in save_preds,
            )
        
        stop_time = time.perf_counter()
        print_time(f"Evaluation ended in {format_time_diff(start_time, stop_time)}")
        
        
    def _unique_path_if_needed(self, dirpath: Path, fname: str) -> str:
        """If a file with the same name already exists (e.g., same second),
        add a numeric suffix -001, -002, ... to avoid overwrite."""
        p = dirpath / fname
        if not p.exists():
            return fname
        stem, ext = os.path.splitext(fname)
        i = 1
        while (dirpath / f"{stem}-{i:03d}{ext}").exists():
            i += 1
        return f"{stem}-{i:03d}{ext}"
    
    
    
    
    
    
    
    
