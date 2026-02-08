from __future__ import annotations
from pathlib import Path
import shutil
import torch
import re

MODEL_NAME = { # used to load backbones of algorithms through the ModelLoader class
    'protonet': 'matchbase',
    'maml': 'matchbase',
    'match': 'match',
    'xmlcnn': 'xmlcnn',
    'attentionxml': 'attentionxml',
    'hector': 'hector',
    'tamlec': 'hector',
    'fastxml': 'fastxml',
    'siamese': 'match',
    'bdc': 'matchbase',
    'lightxml': 'match',
    'cascadexml': 'match',
    'parabel': 'parabel',
    'dexa': 'dexa',
    'ngame': 'ngame',
}

FEWSHOT_TASK = {
    'eurlex': 159,
    'magcs': 7,
    'pubmed': 5,
    'oaxmlc_topics': 4,
    'oaxmlc_concepts': 8,
    'oamedtopics': 54,
    'oamedconcepts': 100
}

BATCH_SIZE = {
    'dexa': {
        'eurlex': 64, 'magcs': 64, 'pubmed': 10, 'oaxmlc_topics': 128,
        'oaxmlc_concepts': 64, 'oamedconcepts': 40, 'oamedconcepts0': 40, 
    },
    'hector': {
        'eurlex': 64, 'magcs': 64, 'pubmed': 10, 'oaxmlc_topics': 128,
        'oaxmlc_concepts': 64, 'oamedconcepts': 40, 'oamedconcepts0': 40, 
    },
    'tamlec': {
        'eurlex': 64, 'magcs': 64, 'pubmed': 40, 'oaxmlc_topics': 128,
        'oaxmlc_concepts': 256, 'oamedconcepts': 8, 'oamedconcepts0': 2, 'oamedtopics':4
    },
    'attentionxml': { 'oamedconcepts': 50, 'oamedconcepts0': 50 , 'oaxmlc_concepts': 8},
    'cascadexml':   { 'oamedconcepts': 1024, 'oamedconcepts_abstractonly': 1024, 'oaxmlc_topics':256, 'oaxmlc_concepts':256 },
    'xmlcnn': { 'oamedconcepts': 50, 'oamedconcepts0': 50 , 'oaxmlc_concepts': 64},
    'lightxml': { 'oamedconcepts': 50, 'oamedconcepts0': 50 , 'oaxmlc_concepts': 8},
    'match': { 'oamedconcepts': 50, 'oamedconcepts0': 50 , 'oaxmlc_concepts': 256},
    
}

ACCUM_ITER = {
    'hector': {
        'eurlex': 3, 'magcs': 5, 'pubmed': 3, 'oaxmlc_topics': 5,
        'oaxmlc_concepts': 2, 'oamedconcepts': 2, 'oamedconcepts0': 2,
    },
    'tamlec': {
        'eurlex': 2, 'magcs': 5, 'pubmed': 2, 'oaxmlc_topics': 5,
        'oaxmlc_concepts': 2, 'oamedconcepts': 2, 'oamedconcepts0': 2,
    },
    'attentionxml': { 'oamedconcepts': 2, 'oamedconcepts0': 2, 'oaxmlc_concepts': 2 },
    'cascadexml':   { 'oamedconcepts': 2, 'oamedconcepts_abstractonly': 2, 'oaxmlc_concepts': 2 },
    'match':   { 'oamedconcepts': 2, 'oamedconcepts_abstractonly': 2, 'oaxmlc_concepts': 2 },
    'xmlcnn':   { 'oamedconcepts': 2, 'oamedconcepts_abstractonly': 2, 'oaxmlc_concepts': 2 },
    'lightxml':   { 'oamedconcepts': 2, 'oamedconcepts_abstractonly': 2, 'oaxmlc_concepts': 2 },
    'parabel':   { 'oamedconcepts': 2, 'oamedconcepts_abstractonly': 2, 'oaxmlc_concepts': 2 },
}

PATIENCE = {
    'protonet': 7, 'maml': 7, 'match': 5, 'xmlcnn': 5, 'attentionxml': 5,
    'hector': 3, 'tamlec': 3, 'fastxml': None, 'siamese': 7, 'bdc': 7,
    'lightxml': 5, 'cascadexml': 5, 'parabel': 5,
}

THRESHOLDS = {
    'oaxmlc_topics': {
        'tamlec': 0.309, 'hector': 0.179, 'attentionxml': 0.342, 'xmlcnn': 0.309,
        'match': 0.328, 'lightxml': 0.314, 'fastxml': 0.316, 'cascadexml': 0.305, 'parabel': 0.336,
    },
    'oaxmlc_concepts': {
        'tamlec': 0.424, 'hector': 0.061, 'attentionxml': 0.36, 'xmlcnn': 0.316,
        'match': 0.361, 'lightxml': 0.296, 'fastxml': 0.348, 'cascadexml': 0.52, 'parabel': 0.466,
    },
}

_ALLOWED_METHODS = set(MODEL_NAME.keys())


# ---------- helpers ----------
def _ensure_tamlec_defaults(cfg: dict) -> None:
    cfg.setdefault('tamlec_params', {})
    params = cfg['tamlec_params']
    params.setdefault('loss_smoothing', 1e-2)
    params.setdefault('width_adaptive', False)
    params.setdefault('decoder_adaptative', 0)
    params.setdefault('tasks_size', False)
    params.setdefault('freeze', False)
    params.setdefault('with_bias', False)


def _build_paths(cfg: dict) -> None:
    """Create default cfg['paths'] """
    exp_name = cfg['exp_name']
    if cfg['fewshot_exp']:
        exp_name = Path(exp_name.stem + "_fewshot")
    
    if 'seed' in cfg.keys() and cfg['seed'] is not None:
        exp_output_name = f"{exp_name.stem}{cfg['seed']}"    
    else: 
        exp_output_name = exp_name.stem    
        cfg['seed'] = 42
    
    seed = cfg['seed']
    output_path = cfg['output_path'] / exp_output_name
    predictions_path = output_path / 'predictions'
    preprocessed_data_path = cfg['dataset_path'] / 'preprocessed_data'

    cfg['paths'] = {
        'output': output_path,
        'dataset': cfg['dataset_path'],
        # metrics
        'metrics': output_path / 'metrics.csv',
        'train_metrics': output_path / 'train_metrics.csv',
        'validation_metrics': output_path / f"{cfg['method']}_validation_metrics.csv",
        'train_level_metrics': output_path / f"{cfg['method']}_train_level_metrics.csv",
        'test_metrics': output_path / f"{cfg['method']}_test_metrics.csv",
        'validation_level_metrics': output_path / f"{cfg['method']}_validation_level_metrics.csv",
        'test_level_metrics': output_path / f"{cfg['method']}_test_level_metrics.csv",
        
        # completion metrics
        'metrics_completion': output_path / 'metrics_completion.csv',
        'train_metrics_completion': output_path / 'train_metrics_completion.csv',
        'validation_metrics_completion': output_path / f"{cfg['method']}_validation_metrics_completion.csv",
        'train_level_metrics_completion': output_path / f"{cfg['method']}_train_level_metrics_completion.csv",
        'test_metrics_completion': output_path / f"{cfg['method']}_test_metrics_completion.csv",
        'validation_level_metrics_completion': output_path / f"{cfg['method']}_validation_level_metrics_completion.csv",
        'test_level_metrics_completion': output_path / f"{cfg['method']}_test_level_metrics_completion.csv",
        
        # models
        'model': output_path / 'model.pt',
        'state_dict': output_path / 'model_state_dict.pt',
        'model_finetuned': output_path / 'model_finetuned.pt',
        'state_dict_finetuned': output_path / 'model_state_dict_finetuned.pt',
        'checkpoints': output_path / 'checkpoints',
        # predictions
        'predictions_folder': predictions_path,
        'test_predictions': predictions_path / 'test_predictions.pt',
        'test_labels': predictions_path / 'test_labels.pt',
        'test_relevant_labels': predictions_path / 'relevant_labels.pt',
        'test_predictions_finetuned': predictions_path / 'test_predictions_finetuned.pt',
        'test_labels_finetuned': predictions_path / 'test_labels_finetuned.pt',
        'test_predictions_completion': predictions_path / 'test_predictions_completion.pt',
        'test_labels_completion': predictions_path / 'test_labels_completion.pt',
        # train predictions
        'train_predictions': predictions_path / 'train_predictions.pt',
        'train_labels': predictions_path / 'train_labels.pt',
        'train_relevant_labels': predictions_path / 'relevant_labels.pt',
        'train_predictions_finetuned': predictions_path / 'train_predictions_finetuned.pt',
        'train_predictions_completion': predictions_path / 'train_predictions_completion.pt',
        'train_labels_finetuned': predictions_path / 'train_labels_finetuned.pt',
        'train_labels_completion': predictions_path / 'train_labels_completion.pt',
        # validation predictions
        'validation_predictions': predictions_path / 'validation_predictions.pt',
        'validation_labels': predictions_path / 'validation_labels.pt',
        'validation_relevant_labels': predictions_path / 'relevant_labels.pt',
        'validation_predictions_finetuned': predictions_path / 'validation_predictions_finetuned.pt',
        'validation_labels_finetuned': predictions_path / 'validation_labels_finetuned.pt',
        'validation_predictions_completion': predictions_path / 'validation_predictions_completion.pt',
        'validation_labels_completion': predictions_path / 'validation_labels_completion.pt',
        # preprocessed data
        'preprocessed_data': preprocessed_data_path,
        'taxos_hector': preprocessed_data_path / 'taxos_hector.pt',
        'taxos_tamlec': preprocessed_data_path / 'taxos_tamlec.pt',
        
        'taxonomy': preprocessed_data_path / 'taxonomy.pt',
        'embeddings': preprocessed_data_path / 'embeddings.pt',
        
        'task_to_subroot_dexa': preprocessed_data_path / f'seed_{seed}/task_to_subroot_dexa.pt',
        'taxos_dexa': preprocessed_data_path / f'seed_{seed}/taxos_dexa.pt',
        'taxonomy_dexa': preprocessed_data_path / f'seed_{seed}/taxonomy_dexa.pt',
        
        'src_vocab': preprocessed_data_path / 'src_vocab.pt',
        'trg_vocab': preprocessed_data_path / 'trg_vocab.pt',
        'abstract_dict': preprocessed_data_path / 'abstract_dict.pt',
        'data': preprocessed_data_path / 'documents',
        
        # split's seed specific
        'tasks_size': preprocessed_data_path / f"seed_{seed}/tasks_size.pt",
        'global_datasets': preprocessed_data_path / f"seed_{seed}/global_datasets.pt",
        'tasks_datasets': preprocessed_data_path / f"seed_{seed}/tasks_datasets.pt",
        'task_to_subroot': preprocessed_data_path / f"seed_{seed}/task_to_subroot.pt",
        'label_to_tasks': preprocessed_data_path / f"seed_{seed}/label_to_tasks.pt",
        
        'tokenizer': preprocessed_data_path / 'tokenizer.model',
        'vocabulary': preprocessed_data_path / 'tokenizer.vocab',
        'dataset_stats': preprocessed_data_path / 'dataset_stats',
        'drawn_tasks': preprocessed_data_path / 'dataset_stats' / 'drawn_tasks',
        "paths_per_doc": preprocessed_data_path / 'paths_per_doc/'
    }

    

def _common_finalize(cfg: dict) -> None:
    """Everything after paths that is common across methods, preserving original logic."""
    

    cfg['emb_dim'] = 300
    cfg['all_tasks_key'] = 'global'
    cfg['model_name'] = MODEL_NAME[cfg['method']]
    cfg['dataset'] = cfg['paths']['dataset'].name
    try:
        cfg['threshold'] = THRESHOLDS[cfg['dataset']][cfg['method']]
    except KeyError:
        cfg['threshold'] = 0.5

    cfg['loss_function'] = torch.nn.BCELoss()
    cfg['optimizer'] = torch.optim.AdamW


    dataset = re.sub(r'\d+', '', cfg['dataset'])
    # batch size logic 
    if cfg['method'] in ['hector', 'tamlec', 'attentionxml', 'lightxml', 'match', 'xmlcnn','parabel', 'cascadexml']:
        if ('batch_size_train' not in cfg) :
            try:
                cfg['batch_size_train'] = BATCH_SIZE[cfg['method']][dataset]
            except KeyError:
                cfg['batch_size_train'] = 64
                cfg['tamlec_params']['accum_iter'] = 5
        if  ('batch_size_eval' not in cfg):
            try:
                cfg['batch_size_eval']  = BATCH_SIZE[cfg['method']][dataset]
            except KeyError:
                cfg['batch_size_eval']  = 128
                cfg['tamlec_params']['accum_iter'] = 5
        cfg['tamlec_params']['accum_iter'] = cfg["tamlec_params"].get('accum_iter', ACCUM_ITER[cfg['method']].get(cfg['dataset'], 5)) 
    elif ('batch_size_train' not in cfg) or ('batch_size_eval' not in cfg):
        cfg['batch_size_train'] = 128
        cfg['batch_size_eval']  = 1024

    print(f"Batch size train: {cfg['batch_size_train']}, eval: {cfg['batch_size_eval']}")
    if "patience" not in cfg.keys():
        cfg['patience'] = PATIENCE.get(cfg['method'], 5)
    print("Early stopping patience:",  cfg['patience'])
    if 'selected_task' not in cfg.keys():
        cfg['selected_task'] = FEWSHOT_TASK.get(cfg['dataset'], 0)

    # hector defaults block 
    if cfg['method'] == 'hector':
        cfg['tamlec_params']['width_adaptive'] = False
        cfg['tamlec_params']['decoder_adaptative'] = 0
        cfg['tamlec_params']['tasks_size'] = False
        cfg['tamlec_params']['freeze'] = False
        cfg['tamlec_params']['with_bias'] = False

def _make_dirs(cfg: dict) -> None:
    """Make directories for model outputs"""
    cfg['paths']['output'].mkdir(exist_ok=True, parents=True)
    cfg['paths']['predictions_folder'].mkdir(exist_ok=True, parents=True)
    cfg['paths']['preprocessed_data'].mkdir(exist_ok=True, parents=True)
    cfg['paths']['data'].mkdir(exist_ok=True, parents=True)
    cfg['paths']['drawn_tasks'].mkdir(exist_ok=True, parents=True)
    src = Path(cfg['exp_name'])
    dst_dir = Path(cfg['paths']['output'])
    dst = dst_dir / src.name  # Copy into the output directory

    # Compare absolute (resolved) paths to avoid self-copy
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

def _clean_up_assert(cfg: dict) -> None:
    # clean up 
    del cfg['output_path']
    del cfg['dataset_path']
    # assertions 
    assert cfg['tokenization_mode'] in ['word', 'bpe', 'unigram'], \
        f"{cfg['tokenization_mode']} should be ['word', 'bpe', 'unigram']"
    assert cfg['sampling_strategy'] in ['standard', 'min_including'], \
        f"{cfg['sampling_strategy']} should be ['standard', 'min_including']"
    assert cfg['method'] != 'fastxml' or not cfg['fewshot_exp'], "Cannot do fewshot experiment with fastxml"
    assert cfg['method'] != 'parabel' or not cfg['fewshot_exp'], "Cannot do fewshot experiment with parabel"
    assert cfg['method'] != 'siamese' or cfg['fewshot_exp'],     "With siamese, only fewshot experiment is available"
    assert cfg['method'] != 'bdc' or cfg['fewshot_exp'],         "With BDC, only fewshot experiment is available"
   

# ---------- per-method builders ----------
def _build_for_generic(cfg: dict) -> dict:
    """Default builder for methods that don't need special pre-path tweaks."""
    assert cfg['method'] in _ALLOWED_METHODS, \
        f"{cfg['method']} not available, choose in {sorted(_ALLOWED_METHODS)}"
    _build_paths(cfg)
    _ensure_tamlec_defaults(cfg)
    _common_finalize(cfg)
    return cfg


def _build_for_hector(cfg: dict) -> dict:
    # same as generic; the hector defaults happen in _common_finalize
    return _build_for_generic(cfg)


def _build_for_tamlec(cfg: dict) -> dict:
    # identical path; accum_iter is set in _common_finalize from ACCUM_ITER
    return _build_for_generic(cfg)


def _build_for_attentionxml(cfg: dict) -> dict:
    return _build_for_generic(cfg)


def _build_for_cascadexml(cfg: dict) -> dict:
    return _build_for_generic(cfg)


# Map method -> builder
_BUILDER_REGISTRY = {
    'match': _build_for_generic,
    'xmlcnn': _build_for_generic,
    'attentionxml': _build_for_attentionxml,
    'hector': _build_for_hector,
    'tamlec': _build_for_tamlec,
    'fastxml': _build_for_generic,
    'lightxml': _build_for_generic,
    'cascadexml': _build_for_cascadexml,
    'parabel': _build_for_generic,
}


def build_cfg_from_factory(cfg: dict) -> dict:
    """
    Public entry point. Consumes the same 'cfg' you pass to Experiment today
    and mutates/returns it exactly like your current constructor would.

    It preserves:
      - paths (with folders/log copy)
      - thresholds, patience, batch sizes, accum_iter
      - tamlec defaults, hector overrides
      - assertions and printed batch sizes
    """
    method = cfg.get('method', '').lower()
    cfg.setdefault('checkpoint_keep_last', 1)
    builder = _BUILDER_REGISTRY.get(method, _build_for_generic)
    cfg.setdefault('checkpoint_keep_last', 1)
    return builder(cfg)