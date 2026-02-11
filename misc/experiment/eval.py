from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import orjson
import torch
from tqdm import tqdm

from misc.metrics.streaming import StreamingXMLMetrics
from misc.utils.metrics_handler import MetricsHandler


@torch.no_grad()
def eval_step_completion(exp, algorithm, dataloaders, split, verbose=False, save_pred=False):
    '''
    Run completion evaluation for a split and write metrics to disk.

    Supports global and per-task evaluation, with optional prediction/label
    batch saving for later analysis.

    :param exp: Experiment object with config, taxonomy, and paths.
    :param algorithm: Model/algorithm instance providing completion inference.
    :param dataloaders: Dict of dataloaders for the given split.
    :param split: Dataset split name (e.g., "validation", "test").
    :param verbose: If True, print metrics to stdout.
    :param save_pred: If True, persist prediction/label batches to disk.
    :return: Metrics dict for the global completion evaluation, or {} if no
        scored samples.
    '''
    taxo = exp.cfg["taxonomy"]
    idx_to_label = getattr(taxo, "idx_to_label", None)
    label_to_level = getattr(taxo, "label_to_level", {})

    def _level_of_idx(i: int) -> int:
        if isinstance(label_to_level, dict) and i in label_to_level:
            return int(label_to_level[i])
        if idx_to_label is not None:
            lab = idx_to_label[int(i)]
            if isinstance(label_to_level, dict) and lab in label_to_level:
                return int(label_to_level[lab])
        return 0

    if exp.cfg["method"] == "tamlec":
        mean_metrics, n_docs_per_metric = {}, {}
        mean_level_metrics, n_docs_level_metrics = {}, {}

        mh = MetricsHandler(
            columns=["model", "task", "finetuning", "metric", "value"],
            output_path=exp.cfg["paths"][f"{split}_metrics_completion"],
        )
        level_mh = MetricsHandler(
            columns=["model", "level", "finetuning", "metric", "value"],
            output_path=exp.cfg["paths"][f"{split}_level_metrics_completion"],
        )

        n_full_plus_pad = taxo.n_nodes + 1

        for _, (doc_files, column_indices, task_id2, batches_indices) in enumerate(
            tqdm(dataloaders[f"tasks_{split}"], leave=False, desc=f"Completion eval on {split} tasks")
        ):
            rel_cols = [int(c) for c in column_indices if _level_of_idx(int(c)) > 1]
            if not rel_cols:
                continue
            rel_cols_t = torch.as_tensor(rel_cols, dtype=torch.long)

            task_acc = StreamingXMLMetrics(
                k_list=exp.cfg["k_list"],
                n_labels=len(rel_cols),
                loss_fn=exp.cfg["loss_function"],
                threshold=exp.cfg["threshold"],
                device="cpu",
            )

            level_to_pos = {}
            for p, lab_idx in enumerate(rel_cols):
                level_to_pos.setdefault(_level_of_idx(lab_idx), []).append(p)

            level_accs = {
                lvl: StreamingXMLMetrics(
                    k_list=exp.cfg["k_list"],
                    n_labels=len(pos),
                    loss_fn=exp.cfg["loss_function"],
                    threshold=exp.cfg["threshold"],
                    device="cpu",
                )
                for lvl, pos in level_to_pos.items()
            }

            any_scored = False
            for batch in batches_indices:
                toks = []
                labels = []
                for j in batch:
                    with open(doc_files[j], "rb") as f:
                        text_tokenized, labs = orjson.loads(f.read())
                    toks.append(torch.tensor(text_tokenized, dtype=torch.int32))
                    labels.append(torch.tensor(labs, dtype=torch.int32))
                input_data = torch.vstack(toks).to(exp.cfg["device"])

                preds_list, gts_list = algorithm.inference_eval_completion(
                    input_data=input_data,
                    labels=labels,
                    task_id=task_id2,
                )

                preds_full = torch.stack([torch.as_tensor(p, device="cpu") for p in preds_list], 0).float()
                labels_full = torch.zeros(preds_full.size(0), n_full_plus_pad, dtype=torch.float32)
                for i, gt in enumerate(gts_list):
                    gt = torch.as_tensor(gt, dtype=torch.long).cpu()
                    if gt.numel():
                        labels_full[i].scatter_(0, gt, 1.0)
                labels_full[:, -1] = 0.0

                preds_task = preds_full[:, rel_cols_t]
                labels_task = labels_full[:, rel_cols_t]
                if labels_task.sum().item() == 0:
                    continue

                any_scored = True
                task_acc.update(preds_task, labels_task)
                for lvl, pos in level_to_pos.items():
                    pos_t = torch.as_tensor(pos, dtype=torch.long)
                    level_accs[lvl].update(preds_task[:, pos_t], labels_task[:, pos_t])

            if not any_scored:
                continue

            task_metrics, n_docs_dict = task_acc.compute()
            for m, v in task_metrics.items():
                mh.add_row({"model": exp.cfg["method"], "task": task_id2, "finetuning": False, "metric": m, "value": v})
                mean_metrics.setdefault(m, []).append(v)
                n_docs_per_metric.setdefault(m, []).append(n_docs_dict[m])

            for lvl, acc in level_accs.items():
                lvl_metrics, lvl_n_docs = acc.compute()
                for m, v in lvl_metrics.items():
                    mean_level_metrics.setdefault(lvl, {}).setdefault(m, []).append(v)
                    n_docs_level_metrics.setdefault(lvl, {}).setdefault(m, []).append(lvl_n_docs[m])

        for m, vals in mean_metrics.items():
            mh.add_row({
                "model": exp.cfg["method"],
                "task": exp.cfg["all_tasks_key"],
                "finetuning": False,
                "metric": m,
                "value": float(np.average(vals, weights=n_docs_per_metric[m])),
            })

        for lvl, md in mean_level_metrics.items():
            for m, vals in md.items():
                level_mh.add_row({
                    "model": exp.cfg["method"],
                    "level": int(lvl),
                    "finetuning": False,
                    "metric": m,
                    "value": float(np.average(vals, weights=n_docs_level_metrics[lvl][m])),
                })
        return

    mh = MetricsHandler(
        columns=["model", "task", "finetuning", "metric", "value"],
        output_path=exp.cfg["paths"][f"{split}_metrics_completion"],
    )
    level_mh = MetricsHandler(
        columns=["model", "level", "finetuning", "metric", "value"],
        output_path=exp.cfg["paths"][f"{split}_level_metrics_completion"],
    )

    pred_dir = lab_dir = None
    if save_pred:
        pred_path = Path(exp.cfg["paths"][f"{split}_predictions_completion"])
        lab_path = Path(exp.cfg["paths"][f"{split}_labels_completion"])
        pred_dir = pred_path.parent / (pred_path.stem + "_batches")
        lab_dir = lab_path.parent / (lab_path.stem + "_batches")
        for d in (pred_dir, lab_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    relevant_idx = None
    global_acc = None
    level_to_pos = {}
    level_accs = {}
    n_full_plus_pad = taxo.n_nodes + 1
    batch_idx = 0
    any_scored = False

    for batch in dataloaders[f"global_{split}"]:
        if len(batch) == 4:
            input_data, _, column_indices, paths_and_children = batch
        else:
            input_data, _, column_indices = batch
            paths_and_children = None
        if relevant_idx is None:
            cols = column_indices
            if isinstance(cols, (list, tuple)) and len(cols) == 1 and isinstance(cols[0], (list, tuple, torch.Tensor)):
                cols = cols[0]
            if isinstance(cols, torch.Tensor):
                cols = cols.detach().cpu().tolist()

            all_relevant_labels = [int(c) for c in cols if _level_of_idx(int(c)) > 1]
            relevant_idx = torch.as_tensor(all_relevant_labels, dtype=torch.long)

            global_acc = StreamingXMLMetrics(
                k_list=exp.cfg["k_list"],
                n_labels=len(relevant_idx),
                loss_fn=exp.cfg["loss_function"],
                threshold=exp.cfg["threshold"],
                device="cpu",
            )

            for pos, lab_idx in enumerate(all_relevant_labels):
                level_to_pos.setdefault(_level_of_idx(lab_idx), []).append(pos)

            level_accs = {
                lvl: StreamingXMLMetrics(
                    k_list=exp.cfg["k_list"],
                    n_labels=len(pos_list),
                    loss_fn=exp.cfg["loss_function"],
                    threshold=exp.cfg["threshold"],
                    device="cpu",
                )
                for lvl, pos_list in level_to_pos.items()
            }

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(algorithm.device, non_blocking=True)

        preds_list, gts_list = algorithm.inference_eval_completion(
            input_data=input_data,
            paths_and_children=paths_and_children,
        )

        preds_full = torch.stack([torch.as_tensor(p, device="cpu") for p in preds_list], 0).float()
        B = preds_full.size(0)

        labels_full = torch.zeros(B, n_full_plus_pad, dtype=torch.float32)
        for i, gt in enumerate(gts_list):
            gt = torch.as_tensor(gt, dtype=torch.long, device="cpu")
            if gt.numel():
                labels_full[i].scatter_(0, gt, 1.0)
        labels_full[:, -1] = 0.0

        preds_rel = preds_full[:, relevant_idx]
        labels_rel = labels_full[:, relevant_idx]
        if labels_rel.sum().item() == 0:
            continue

        any_scored = True
        global_acc.update(preds_rel, labels_rel)

        for lvl, pos_list in level_to_pos.items():
            pos_t = torch.as_tensor(pos_list, dtype=torch.long)
            level_accs[lvl].update(preds_rel[:, pos_t], labels_rel[:, pos_t])

        if save_pred:
            torch.save(preds_full, pred_dir / f"pred_batch_{batch_idx:06d}.pt")
            torch.save(labels_full.bool(), lab_dir / f"lab_batch_{batch_idx:06d}.pt")
            batch_idx += 1

    if not any_scored:
        return {}

    metrics_global, _ = global_acc.compute()
    if verbose:
        print(f"> Results on the {split} set (completion)")
    for m, v in metrics_global.items():
        mh.add_row({"model": exp.cfg["method"], "task": exp.cfg["all_tasks_key"], "finetuning": False, "metric": m, "value": v})
        if verbose:
            print(f">> {m} -> {v}")

    for lvl, acc in level_accs.items():
        lvl_metrics, _ = acc.compute()
        if verbose:
            print(f"\n>> Completion metrics at level {lvl}")
        for m, v in lvl_metrics.items():
            level_mh.add_row({"model": exp.cfg["method"], "level": int(lvl), "finetuning": False, "metric": m, "value": float(v)})
            if verbose:
                print(f">>> {m} -> {v}")

    return metrics_global


@torch.no_grad()
def eval_step(
    exp,
    algorithm,
    dataloaders,
    split,
    metrics_handler,
    verbose=False,
    save_pred=False,
    eval_finetuning=False,
) -> dict:
    '''
    Run evaluation for a split (global and/or per-task) and log metrics.

    Handles multiple dataloader formats, supports saving per-batch
    predictions/labels, and optionally evaluates finetuned outputs.

    :param exp: Experiment object with config, taxonomy, and paths.
    :param algorithm: Model/algorithm instance providing inference methods.
    :param dataloaders: Dict of dataloaders for the given split.
    :param split: Dataset split name (e.g., "validation", "test").
    :param metrics_handler: Key into ``exp.metrics_handler`` for logging.
    :param verbose: If True, print metrics to stdout.
    :param save_pred: If True, persist prediction/label batches to disk.
    :param eval_finetuning: If True, evaluate finetuned predictions/labels.
    :return: Dict mapping task identifiers to their metrics dicts.
    :rtype: dict[Any, Any]
    '''

    returned_dict = {}

    if f"global_{split}" in dataloaders.keys():
        taxo = exp.cfg["taxonomy"]
        level_to_indices = {}
        for level, labels_at_level in taxo.level_to_labels.items():
            if level == 0:
                continue
            idxs = [taxo.label_to_idx[lab] for lab in labels_at_level]
            level_to_indices[level] = torch.as_tensor(idxs, dtype=torch.long)

        level_accs = {
            level: StreamingXMLMetrics(
                k_list=exp.cfg["k_list"],
                n_labels=len(idxs),
                loss_fn=exp.cfg["loss_function"],
                threshold=exp.cfg["threshold"],
                device="cpu",
            )
            for level, idxs in level_to_indices.items()
        }

        global_acc = None

        if not eval_finetuning:
            pred_key = f"{split}_predictions"
            lab_key = f"{split}_labels"
        else:
            pred_key = f"{split}_predictions_finetuned"
            lab_key = f"{split}_labels_finetuned"

        pred_path = Path(exp.cfg["paths"][pred_key])
        lab_path = Path(exp.cfg["paths"][lab_key])

        pred_dir = pred_path.parent / (pred_path.stem + "_batches")
        lab_dir = lab_path.parent / (lab_path.stem + "_batches")

        if save_pred:
            for d in [pred_dir, lab_dir]:
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True, exist_ok=True)

        batch_idx = 0

        if exp.one_hot_labels:
            for input_data, labels, column_indices in tqdm(
                dataloaders[f"global_{split}"], leave=False, desc=f"Evaluating model on {split}"
            ):
                if isinstance(input_data, torch.Tensor):
                    input_data = input_data.to(exp.cfg["device"])

                preds = algorithm.inference_eval(input_data, labels=labels)

                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()

                
                global_acc = StreamingXMLMetrics(
                    k_list=exp.cfg["k_list"],
                    n_labels=labels_cpu.shape[1] - 1,
                    loss_fn=exp.cfg["loss_function"],
                    threshold=exp.cfg["threshold"],
                    device="cpu",
                )
                # Filter root label
                filtered_preds = preds_cpu[:, 1:]
                filtered_labels = labels_cpu[:, 1:]
                global_acc.update(filtered_preds, filtered_labels)

                for level, idxs in level_to_indices.items():
                    level_preds = preds_cpu[:, idxs]
                    level_labels = labels_cpu[:, idxs]
                    level_accs[level].update(level_preds, level_labels)

                if save_pred:
                    batch_pred_file = pred_dir / f"pred_batch_{batch_idx:06d}.pt"
                    batch_lab_file = lab_dir / f"lab_batch_{batch_idx:06d}.pt"
                    torch.save(preds_cpu, batch_pred_file)
                    torch.save(labels_cpu.bool(), batch_lab_file)
                    batch_idx += 1

        else:
            
            n_labels_full = algorithm.taxonomy.n_nodes
            n_labels = n_labels_full - 1 
            
                
            global_acc = StreamingXMLMetrics(
                k_list=exp.cfg["k_list"],
                n_labels=n_labels,
                loss_fn=exp.cfg["loss_function"],
                threshold=exp.cfg["threshold"],
                device="cpu",
            )
        
            for batched_input, batched_labels, _ in tqdm(
                dataloaders[f"global_{split}"], leave=False, desc=f"Evaluating model on {split}"
            ):
                preds = algorithm.inference_eval(batched_input, labels=batched_labels)
                preds_cpu = preds.detach().cpu()
                
                
                B = len(batched_labels)
                
                dense_labels = torch.zeros(B, n_labels_full, dtype=torch.float32)
                for i, idx in enumerate(batched_labels):
                    idx = torch.as_tensor(idx, dtype=torch.long)
                    if idx.numel() > 0:
                        dense_labels[i].scatter_(0, idx, 1.0)
                labels_cpu = dense_labels


                # Filter out root label
                filtered_preds = preds_cpu[:, 1:]
                filtered_labels = labels_cpu[:, 1:]
                global_acc.update(filtered_preds, filtered_labels)

                for level, idxs in level_to_indices.items():
                    level_preds = preds_cpu[:, idxs]
                    level_labels = labels_cpu[:, idxs]
                    level_accs[level].update(level_preds, level_labels)

                if save_pred:
                    batch_pred_file = pred_dir / f"pred_batch_{batch_idx:06d}.pt"
                    batch_lab_file = lab_dir / f"lab_batch_{batch_idx:06d}.pt"
                    torch.save(preds_cpu, batch_pred_file)
                    torch.save(labels_cpu.bool(), batch_lab_file)
                    batch_idx += 1

        metrics_global, _n_docs = global_acc.compute()

        if verbose:
            print(f"> Results on the {split} set:")

        for metric_name, metric_value in metrics_global.items():
            new_row_dict = {
                "model": exp.cfg["method"],
                "task": exp.cfg["all_tasks_key"],
                "finetuning": eval_finetuning,
                "metric": metric_name,
                "value": metric_value,
            }
            exp.metrics_handler[metrics_handler].add_row(new_row_dict)
            if verbose:
                print(f">> {metric_name} -> {metric_value}")

        returned_dict[exp.cfg["all_tasks_key"]] = metrics_global

        

        level_metrics_handler = MetricsHandler(
            columns=["model", "level", "finetuning", "metric", "value"],
            output_path=exp.cfg["paths"][f"{split}_level_metrics"],
        )

        level_metrics = {}
        for level, acc in level_accs.items():
            level_m, _ = acc.compute()
            level_metrics[level] = level_m

            print(f"\n>> Metrics at level {level}")
            for metric_name, metric_value in level_m.items():
                new_row_dict = {
                    "model": exp.cfg["method"],
                    "level": level,
                    "finetuning": eval_finetuning,
                    "metric": metric_name,
                    "value": metric_value,
                }
                level_metrics_handler.add_row(new_row_dict)
                print(f">>> {metric_name} -> {metric_value}")

    if f"tasks_{split}" in dataloaders.keys():

        if dataloaders[f"tasks_{split}"].collate_fn.__name__ == "collate_no_batch":
            for task_id, (input_data, labels, task_relevant_classes) in enumerate(
                tqdm(
                    dataloaders[f"tasks_{split}"],
                    leave=False,
                    desc=f"Evalutating model on {split} tasks",
                )
            ):
                task_labels_full = []
                for _, sample_label in zip(input_data, labels):
                    one_hot_label = torch.zeros(algorithm.taxonomy.n_nodes, dtype=torch.float32)
                    one_hot_label.scatter_(0, torch.as_tensor(sample_label, dtype=torch.long), 1.0)
                    task_labels_full.append(one_hot_label)
                task_labels_full = torch.stack(task_labels_full, dim=0)

                task_predictions_full = algorithm.inference_eval(input_data)

                task_relevant_classes = torch.as_tensor(task_relevant_classes, dtype=torch.long)
                filtered_predictions = task_predictions_full[:, task_relevant_classes]
                filtered_labels = task_labels_full[:, task_relevant_classes]

                task_acc = StreamingXMLMetrics(
                    k_list=exp.cfg["k_list"],
                    n_labels=filtered_predictions.shape[1],
                    loss_fn=exp.cfg["loss_function"],
                    threshold=exp.cfg["threshold"],
                    device="cpu",
                )
                task_acc.update(
                    filtered_predictions.detach().cpu(),
                    filtered_labels.detach().cpu(),
                )
                task_metrics, _n_docs = task_acc.compute()

                for metric_name, metric_value in task_metrics.items():
                    new_row_dict = {
                        "model": exp.cfg["method"],
                        "task": task_id,
                        "finetuning": False,
                        "metric": metric_name,
                        "value": metric_value,
                    }
                    exp.metrics_handler[f"eval_{split}"].add_row(new_row_dict)
                returned_dict[task_id] = task_metrics

        elif dataloaders[f"tasks_{split}"].collate_fn.__name__ == "_collate_hector_tamlec":
            for task_id, (doc_files, column_indices, task_id2, batches_indices) in enumerate(
                tqdm(dataloaders[f"tasks_{split}"], leave=False, desc=f"Evaluating model on {split} tasks")
            ):
                task_relevant_classes = torch.as_tensor(column_indices, dtype=torch.long)

                task_acc = None
                n_labels_full_plus_pad = exp.cfg["taxonomy"].n_nodes + 1

                for batch in batches_indices:
                    toks = []
                    labels_in_batch = []

                    for j in batch:
                        with open(doc_files[j], "rb") as f:
                            text_tokenized, label = orjson.loads(f.read())

                        toks.append(torch.tensor(text_tokenized, dtype=torch.int32))
                        labels_in_batch.append(label)

                    input_data = torch.vstack(toks).to(exp.cfg["device"])

                    preds_full_batch = algorithm.inference_eval(
                        input_data,
                        labels=labels_in_batch,
                        task_id=task_id2,
                    ).to(torch.float32)

                    B = len(labels_in_batch)
                    labels_full_batch = torch.zeros(B, n_labels_full_plus_pad, dtype=torch.float32)

                    for i, lab in enumerate(labels_in_batch):
                        lab = torch.as_tensor(lab, dtype=torch.long)
                        labels_full_batch[i].scatter_(0, lab, 1.0)
                        assert labels_full_batch[i, -1] == 0.0

                    preds_task_batch = preds_full_batch[:, task_relevant_classes]
                    labels_task_batch = labels_full_batch[:, task_relevant_classes]

                    if task_acc is None:
                        task_acc = StreamingXMLMetrics(
                            k_list=exp.cfg["k_list"],
                            n_labels=preds_task_batch.shape[1],
                            loss_fn=exp.cfg["loss_function"],
                            threshold=exp.cfg["threshold"],
                            device="cpu",
                        )

                    task_acc.update(
                        preds_task_batch.detach().cpu(),
                        labels_task_batch.detach().cpu(),
                    )

                task_metrics, _n_docs = task_acc.compute()

                for metric_name, metric_value in task_metrics.items():
                    exp.metrics_handler[f"eval_{split}"].add_row({
                        "model": exp.cfg["method"],
                        "task": task_id2,
                        "finetuning": eval_finetuning,
                        "metric": metric_name,
                        "value": metric_value,
                    })

                returned_dict[task_id2] = task_metrics

        else:
            for task_id, (batched_input, batched_labels, task_relevant_classes) in enumerate(
                tqdm(
                    dataloaders[f"tasks_{split}"],
                    leave=False,
                    desc=f"Evalutating model on {split} tasks",
                )
            ):
                task_relevant_classes = torch.as_tensor(task_relevant_classes, dtype=torch.long)

                task_acc = None

                for input_data, labels in zip(batched_input, batched_labels):
                    input_data = input_data.to(exp.cfg["device"])
                    predictions = algorithm.inference_eval(input_data)

                    preds_cpu = predictions.detach().cpu()
                    labels_cpu = labels.cpu()

                    filtered_preds = preds_cpu[:, task_relevant_classes]
                    filtered_labels = labels_cpu[:, task_relevant_classes]

                    if task_acc is None:
                        task_acc = StreamingXMLMetrics(
                            k_list=exp.cfg["k_list"],
                            n_labels=filtered_preds.shape[1],
                            loss_fn=exp.cfg["loss_function"],
                            threshold=exp.cfg["threshold"],
                            device="cpu",
                        )

                    task_acc.update(filtered_preds, filtered_labels)

                task_metrics, _ = task_acc.compute()

                for metric_name, metric_value in task_metrics.items():
                    exp.metrics_handler[f"eval_{split}"].add_row({
                        "model": exp.cfg["method"],
                        "task": task_id,
                        "finetuning": eval_finetuning,
                        "metric": metric_name,
                        "value": metric_value,
                    })

                returned_dict[task_id] = task_metrics

    return returned_dict
