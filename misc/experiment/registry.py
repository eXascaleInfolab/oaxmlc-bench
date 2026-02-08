from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class MethodSpec:
    '''
    Registry entry describing a method's components and setup hooks.
    '''
    name: str
    algclass_factory: Callable[[], type]
    dataclass_factory: Callable[[], type]
    one_hot_labels: bool
    build_args: Callable[[dict, dict], dict]
    pre_setup: Optional[Callable[[dict], None]] = None
    init_message: Optional[str] = None


def build_base_alg_args(cfg: dict, metrics_handler: dict) -> dict:
    '''
    Build the base argument dictionary passed to algorithm constructors.

    :param cfg: Experiment configuration dictionary.
    :type cfg: dict
    :param metrics_handler: Metrics handlers mapping for logging.
    :type metrics_handler: dict
    :return: Base algorithm argument dictionary.
    :rtype: dict[Any, Any]
    '''
    return {
        "metrics_handler": metrics_handler,
        "taxonomy": None,
        "optimizer": cfg["optimizer"],
        "loss_function": cfg["loss_function"],
        "embeddings_loader": None,
        "method": cfg["method"],
        "state_dict_path": cfg["paths"]["state_dict"],
        "state_dict_finetuned_path": cfg["paths"]["state_dict_finetuned"],
        "model_path": cfg["paths"]["model"],
        "model_finetuned_path": cfg["paths"]["model_finetuned"],
        "device": cfg["device"],
        "verbose": True,
        "patience": cfg["patience"],
        "all_tasks_key": cfg["all_tasks_key"],
        "selected_task": cfg.get("selected_task", None),
        "learning_rate": cfg["learning_rate"],
        "min_improvement": cfg.get("min_improvement", 0.0),
    }


# ----- Lazy imports -----

def _alg_ngame():
    from algorithms.ngame import NgameAlg

    return NgameAlg


def _alg_dexa():
    from algorithms.dexa import DexaAlg

    return DexaAlg


def _alg_lightxml():
    from algorithms.lightxml import LightxmlAlg

    return LightxmlAlg


def _alg_xmlcnn():
    from algorithms.xmlcnn import XmlcnnAlg

    return XmlcnnAlg


def _alg_match():
    from algorithms.match import MatchAlg

    return MatchAlg


def _alg_attentionxml():
    from algorithms.attentionxml import AttentionxmlAlg

    return AttentionxmlAlg


def _alg_cascadexml():
    from algorithms.cascadexml import CascadexmlAlg

    return CascadexmlAlg


def _alg_fastxml():
    from algorithms.fastxml import FastxmlAlg

    return FastxmlAlg


def _alg_hector():
    from algorithms.hector import HectorAlg

    return HectorAlg


def _alg_parabel():
    from algorithms.parabel import ParabelAlg

    return ParabelAlg


def _alg_tamlec():
    from algorithms.tamlec import TamlecAlg

    return TamlecAlg


def _dm_dexa():
    from datahandler.data_modules.dexa import DataModuleDexa

    return DataModuleDexa


def _dm_base():
    from datahandler.data_modules.base import DataModuleBase

    return DataModuleBase


def _dm_fastxml_parabel():
    from datahandler.data_modules.fastxml_parabel import DataModuleFastxmlParabel

    return DataModuleFastxmlParabel


def _dm_hector_tamlec():
    from datahandler.data_modules.hector_tamlec import DataModuleHectorTamlec

    return DataModuleHectorTamlec


# ----- Method-specific builders -----

def _set_encoder_distilbert(cfg: dict) -> None:
    cfg["encoder_name"] = "sentence-transformers/msmarco-distilbert-base-v4"


def _build_dexa_like_args(cfg: dict, _base: dict) -> dict:
    return {
        "taxonomy_path": cfg["paths"]["taxonomy_dexa"],
        "data_path": cfg["paths"]["dataset"],
        "tokenizer_path": cfg["paths"]["dataset"] / "preprocessed_data" / "dexa_tokenizer.model",
        "voc_size": cfg["voc_size"],
        "tokenization_mode": cfg["tokenization_mode"],
        "seed": cfg["seed"],
        "device": cfg["device"],
        "n_doc_train": cfg["n_global_train"],
        "n_doc_test": cfg["n_global_test"],
        "output_dir": cfg["paths"]["output"],
        "batch_size_siamese": cfg["batch_size_train"],
        "batch_size_extreme": cfg["batch_size_eval"],
        "fewshot_exp": cfg["fewshot_exp"],
        "selected_task": cfg.get("selected_task", None),
        "task_to_subroot": cfg["task_to_subroot"],
    }


def _build_hector_args(cfg: dict, base: dict) -> dict:
    return {
        "task_to_subroot": cfg["task_to_subroot"],
        "metrics_handler": base["metrics_handler"],
        "taxonomy": None,
        "beam_parameter": cfg.get("beam_parameter", 10),
        "proba_operator": cfg.get("proba_operator", "MAX_PROBA"),
        "batch_size_eval": cfg.get("batch_size_eval", 256),
        "src_vocab": cfg["tamlec_params"]["src_vocab"],
        "trg_vocab": cfg["tamlec_params"]["trg_vocab"],
        "abstract_dict": cfg["tamlec_params"]["abstract_dict"],
        "taxos_hector": cfg["tamlec_params"]["taxos_hector"],
        "width_adaptive": cfg["tamlec_params"]["width_adaptive"],
        "decoder_adaptative": cfg["tamlec_params"]["decoder_adaptative"],
        "tasks_size": cfg["tamlec_params"]["tasks_size"],
        "with_bias": cfg["tamlec_params"]["with_bias"],
        "accum_iter": cfg["tamlec_params"]["accum_iter"],
        "loss_smoothing": cfg["tamlec_params"]["loss_smoothing"],
        "seq_length": cfg["seq_length"],
        "k_list_eval_perf": cfg["k_list_eval_perf"],
        "optimizer": cfg["optimizer"],
        "loss_function": cfg["loss_function"],
        "embeddings_loader": None,
        "method": cfg["method"],
        "state_dict_path": cfg["paths"]["state_dict"],
        "state_dict_finetuned_path": cfg["paths"]["state_dict_finetuned"],
        "model_path": cfg["paths"]["model"],
        "model_finetuned_path": cfg["paths"]["model_finetuned"],
        "device": cfg["device"],
        "verbose": True,
        "patience": cfg["patience"],
        "all_tasks_key": cfg["all_tasks_key"],
        "selected_task": cfg.get("selected_task", None),
    }


def _build_tamlec_args(cfg: dict, _base: dict) -> dict:
    return {
        "metrics_handler": _base["metrics_handler"],
        "taxonomy": None,
        "src_vocab": cfg["tamlec_params"]["src_vocab"],
        "trg_vocab": cfg["tamlec_params"]["trg_vocab"],
        "abstract_dict": cfg["tamlec_params"]["abstract_dict"],
        "taxos_tamlec": cfg["tamlec_params"]["taxos_tamlec"],
        "width_adaptive": cfg["tamlec_params"]["width_adaptive"],
        "decoder_adaptative": cfg["tamlec_params"]["decoder_adaptative"],
        "tasks_size": cfg["tamlec_params"]["tasks_size"],
        "with_bias": cfg["tamlec_params"]["with_bias"],
        "accum_iter": cfg["tamlec_params"]["accum_iter"],
        "loss_smoothing": cfg["tamlec_params"]["loss_smoothing"],
        "seq_length": cfg["seq_length"],
        "freeze": cfg["tamlec_params"]["freeze"],
        "label_to_task": cfg["label_to_tasks"],
        "k_list_eval_perf": cfg["k_list_eval_perf"],
        "method": cfg["method"],
        "state_dict_path": cfg["paths"]["state_dict"],
        "state_dict_finetuned_path": cfg["paths"]["state_dict_finetuned"],
        "model_path": cfg["paths"]["model"],
        "model_finetuned_path": cfg["paths"]["model_finetuned"],
        "device": cfg["device"],
        "patience": cfg["patience"],
        "all_tasks_key": cfg["all_tasks_key"],
        "selected_task": cfg.get("selected_task", None),
        "learning_rate": cfg["learning_rate"],
        "fewshot_exp": cfg["fewshot_exp"],
        "max_freeze_epochs": cfg["tamlec_params"]["max_freeze_epochs"],
        "task_to_subroot": cfg["task_to_subroot"],
    }


SUPPORTED_METHODS = {
    "ngame",
    "dexa",
    "protonet",
    "maml",
    "match",
    "xmlcnn",
    "attentionxml",
    "hector",
    "tamlec",
    "fastxml",
    "siamese",
    "bdc",
    "lightxml",
    "cascadexml",
    "parabel",
}


METHODS: Dict[str, MethodSpec] = {
    "ngame": MethodSpec(
        name="ngame",
        algclass_factory=_alg_ngame,
        dataclass_factory=_dm_dexa,
        one_hot_labels=True,
        build_args=_build_dexa_like_args,
        pre_setup=_set_encoder_distilbert,
        init_message="Starting NGAME experiment",
    ),
    "dexa": MethodSpec(
        name="dexa",
        algclass_factory=_alg_dexa,
        dataclass_factory=_dm_dexa,
        one_hot_labels=True,
        build_args=_build_dexa_like_args,
        pre_setup=_set_encoder_distilbert,
        init_message="Starting Dexa experiment",
    ),
    "lightxml": MethodSpec(
        name="lightxml",
        algclass_factory=_alg_lightxml,
        dataclass_factory=_dm_base,
        one_hot_labels=True,
        build_args=lambda cfg, base: base,
        init_message="Starting LightXML experiment",
    ),
    "xmlcnn": MethodSpec(
        name="xmlcnn",
        algclass_factory=_alg_xmlcnn,
        dataclass_factory=_dm_base,
        one_hot_labels=True,
        build_args=lambda cfg, base: base,
        init_message="Starting XMLcnn experiment",
    ),
    "match": MethodSpec(
        name="match",
        algclass_factory=_alg_match,
        dataclass_factory=_dm_base,
        one_hot_labels=True,
        build_args=lambda cfg, base: base,
        init_message="Starting MatchXML experiment",
    ),
    "attentionxml": MethodSpec(
        name="attentionxml",
        algclass_factory=_alg_attentionxml,
        dataclass_factory=_dm_base,
        one_hot_labels=True,
        build_args=lambda cfg, base: base,
        init_message="Starting AttentionXML experiment",
    ),
    "cascadexml": MethodSpec(
        name="cascadexml",
        algclass_factory=_alg_cascadexml,
        dataclass_factory=_dm_base,
        one_hot_labels=True,
        build_args=lambda cfg, base: base,
        init_message="Starting CascadeXML experiment",
    ),
    "fastxml": MethodSpec(
        name="fastxml",
        algclass_factory=_alg_fastxml,
        dataclass_factory=_dm_fastxml_parabel,
        one_hot_labels=False,
        build_args=lambda cfg, base: base,
        init_message="Starting FastXML experiment",
    ),
    "hector": MethodSpec(
        name="hector",
        algclass_factory=_alg_hector,
        dataclass_factory=_dm_hector_tamlec,
        one_hot_labels=False,
        build_args=_build_hector_args,
        init_message="Starting Hector experiment",
    ),
    "parabel": MethodSpec(
        name="parabel",
        algclass_factory=_alg_parabel,
        dataclass_factory=_dm_fastxml_parabel,
        one_hot_labels=False,
        build_args=lambda cfg, base: base,
        init_message="Starting Parabel experiment",
    ),
    "tamlec": MethodSpec(
        name="tamlec",
        algclass_factory=_alg_tamlec,
        dataclass_factory=_dm_hector_tamlec,
        one_hot_labels=False,
        build_args=_build_tamlec_args,
        init_message="Starting Tamlec experiment",
    ),
}


def get_method_spec(method: str) -> MethodSpec:
    '''
    Fetch the MethodSpec for a registered method.

    :param method: Method name key.
    :return: MethodSpec for the given method.
    :raises NotImplementedError: If the method is not registered.
    '''
    if method not in METHODS:
        raise NotImplementedError(f"Method {method} not implemented yet.")
    return METHODS[method]
