# Configuration Guide

This project uses Python config files in `configs/` to define experiments. Each config file builds a `cfg` dictionary and calls `misc.utils.config_main(parser, cfg)` to run training/evaluation via CLI flags.

Use `configs/base_config.py` as the starting template.

**How To Run A Config**
```bash
# Train
python configs/<config_name>.py --train

# Evaluate (both validation and test)
python configs/<config_name>.py --evaluate

# Evaluate a specific split
python configs/<config_name>.py --evaluate validation
python configs/<config_name>.py --evaluate test

# Finetune only (few-shot mode)
python configs/<config_name>.py --finetune

# Completion evaluation
python configs/<config_name>.py --completion
```

**CLI Flags For Config Files**
These flags are defined in `misc/utils.py:config_main`. Exactly one of `--train`, `--evaluate`, `--finetune`, or `--completion` is required.

| Flag | Argument | Effect |
| --- | --- | --- |
| `--train` | none | Train the model and run evaluation after training. |
| `--evaluate` | `validation` \| `test` \| `both` (optional) | Evaluate a trained model. If no value is provided, runs both validation and test. |
| `--finetune` | none | Run finetuning only (sets `fewshot_exp`). |
| `--completion` | none | Run completion evaluation only. |
| `--fewshot` | none | Enables few-shot behavior (sets `fewshot_exp`). |
| `--seed` | int | Overrides `cfg['seed']`. |
| `--device` | string | Overrides `cfg['device']` (e.g. `cpu`, `cuda`, `cuda:0`, `mps`). |
| `--debug` | none | Enables postmortem debugging on uncaught exceptions. |


**User-Settable `cfg` Keys**
These are the configuration parameters you can define in a config file. Some are optional overrides with defaults derived in `misc/cfg_factory.py`.

| Key | Type / Values | Meaning and default behavior |
| --- | --- | --- |
| `dataset_path` | `Path` | Path to the dataset root (e.g. `datasets/oaxmlc_topics`). |
| `output_path` | `Path` | Root output directory. Experiment outputs live under `output_path/<exp_name_stem>`. |
| `exp_name` | `Path` | Experiment name. Typically `Path(__file__)` so the config filename is copied into the output folder. |
| `device` | string | Device string: `cpu`, `cuda`, `cuda:N`, or `mps`. Can be overridden by `--device`. |
| `method` | string | Algorithm name. Allowed: `protonet`, `maml`, `match`, `xmlcnn`, `attentionxml`, `fastxml`, `hector`, `tamlec`, `siamese`, `bdc`, `lightxml`, `cascadexml`, `parabel`, `dexa`, `ngame`. |
| `learning_rate` | float | Training learning rate for the main optimizer. |
| `learner_rate` | float | Inner-loop learning rate (MAML only). |
| `seq_length` | int | Maximum input sequence length. |
| `voc_size` | int | Maximum vocabulary size. |
| `tokenization_mode` | string | Tokenization mode: `word`, `bpe`, or `unigram`. |
| `n_shot` | int | Support set size per class (few-shot methods). |
| `n_query` | int | Query set size (few-shot methods). |
| `sampling_strategy` | string | Few-shot sampling strategy: `standard` or `min_including`. |
| `n_optim_steps` | int | Number of inner-loop optimization steps (MAML only). |
| `k_list` | list[int] | K values for final metrics (e.g. `list(range(1, 21))`). |
| `k_list_eval_perf` | list[int] | K values for training-time evaluation. |
| `fewshot_exp` | bool | Enables few-shot workflow. This is also set by `--fewshot` or `--finetune`. |
| `selected_task` | int | Task id used for few-shot finetuning. Defaults are derived from dataset name. |
| `batch_size_train` | int | Training batch size. Defaults are derived from dataset + method. |
| `batch_size_eval` | int | Evaluation batch size. Defaults are derived from dataset + method. |
| `patience` | int or `None` | Early-stopping patience. Defaults by method. |
| `min_improvement` | float | Minimum relative improvement to reset early-stopping patience. Default `0.0`. |
| `min_freq` | int | Minimum label frequency for taxonomy pruning. Default `50`. |
| `seed` | int | Random seed. Defaults to `42` if not provided. Can be overridden by `--seed`. |
| `checkpoint_keep_last` | int | Number of training checkpoints to keep. Default `1`. |
| `console_log_fname` | string | Base filename for the console log. Default `all_console.log`. |
| `beam_parameter` | int | HECTOR beam size. Default `10`. |
| `proba_operator` | string | HECTOR probability operator. Default `MAX_PROBA`. |
| `tamlec_params` | dict | Parameter group for HECTOR/TAMLEC (see next section). |

**`tamlec_params` Subkeys**
You can set these in the config; some are auto-filled during preprocessing.

| Key | Type / Values | Meaning and default behavior |
| --- | --- | --- |
| `loss_smoothing` | float | Label smoothing for TAMLEC/HECTOR losses. Default `1e-2`. |
| `width_adaptive` | bool | HECTOR/TAMLEC adaptive width flag. Forced `False` for HECTOR. |
| `decoder_adaptative` | int | Decoder adaptation mode. Forced `0` for HECTOR. |
| `tasks_size` | bool | If `True`, uses task sizes from preprocessing (otherwise `None`). |
| `freeze` | bool | Freeze parts of TAMLEC during training. |
| `with_bias` | bool | Use bias terms in TAMLEC/HECTOR. |
| `accum_iter` | int | Gradient accumulation steps. Defaults are derived from dataset + method. |
| `max_freeze_epochs` | int | Max epochs for freeze phase in TAMLEC. |
| `src_vocab` | auto | Populated during preprocessing (do not set manually). |
| `trg_vocab` | auto | Populated during preprocessing (do not set manually). |
| `abstract_dict` | auto | Populated during preprocessing (do not set manually). |
| `taxos_hector` | auto | Populated during preprocessing (do not set manually). |
| `taxos_tamlec` | auto | Populated during preprocessing (do not set manually). |

**Auto-Derived Fields**
The config factory in `misc/cfg_factory.py` fills in several fields. You typically do not set these directly.

Common auto-derived keys:
- `paths` (dict of all experiment paths)
- `emb_dim` (embedding dimension)
- `all_tasks_key` (task key for global metrics)
- `model_name` (internal model loader key)
- `dataset` (dataset name derived from `dataset_path`)
- `threshold` (default per-method threshold)
- `loss_function` (defaults to `torch.nn.BCELoss`)
- `optimizer` (defaults to `torch.optim.AdamW`)

The `paths` dict contains these keys:
`paths.output`, `paths.dataset`, `paths.metrics`, `paths.train_metrics`, `paths.validation_metrics`, `paths.train_level_metrics`, `paths.test_metrics`, `paths.validation_level_metrics`, `paths.test_level_metrics`, `paths.metrics_completion`, `paths.train_metrics_completion`, `paths.validation_metrics_completion`, `paths.train_level_metrics_completion`, `paths.test_metrics_completion`, `paths.validation_level_metrics_completion`, `paths.test_level_metrics_completion`, `paths.model`, `paths.state_dict`, `paths.model_finetuned`, `paths.state_dict_finetuned`, `paths.checkpoints`, `paths.predictions_folder`, `paths.test_predictions`, `paths.test_labels`, `paths.test_relevant_labels`, `paths.test_predictions_finetuned`, `paths.test_labels_finetuned`, `paths.test_predictions_completion`, `paths.test_labels_completion`, `paths.train_predictions`, `paths.train_labels`, `paths.train_relevant_labels`, `paths.train_predictions_finetuned`, `paths.train_predictions_completion`, `paths.train_labels_finetuned`, `paths.train_labels_completion`, `paths.validation_predictions`, `paths.validation_labels`, `paths.validation_relevant_labels`, `paths.validation_predictions_finetuned`, `paths.validation_labels_finetuned`, `paths.validation_predictions_completion`, `paths.validation_labels_completion`, `paths.preprocessed_data`, `paths.taxos_hector`, `paths.taxos_tamlec`, `paths.taxonomy`, `paths.embeddings`, `paths.task_to_subroot_dexa`, `paths.taxos_dexa`, `paths.taxonomy_dexa`, `paths.src_vocab`, `paths.trg_vocab`, `paths.abstract_dict`, `paths.data`, `paths.tasks_size`, `paths.global_datasets`, `paths.tasks_datasets`, `paths.task_to_subroot`, `paths.label_to_tasks`, `paths.tokenizer`, `paths.vocabulary`, `paths.dataset_stats`, `paths.drawn_tasks`, `paths.paths_per_doc`.

**Constraints Enforced By The Config Factory**
- `tokenization_mode` must be `word`, `bpe`, or `unigram`.
- `sampling_strategy` must be `standard` or `min_including`.
- `fastxml` and `parabel` cannot run in few-shot mode.
- `siamese` and `bdc` require few-shot mode.
