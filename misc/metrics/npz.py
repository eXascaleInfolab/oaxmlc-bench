import torch
import numpy as np
from scipy.sparse import load_npz
from misc.metrics.streaming import StreamingXMLMetrics

def stream_metrics_per_task_from_taxos_tamlec(
    pred_npz_path,
    xy_path,
    taxos_tamlec,          # torch.load(".../taxos_tamlec.pt")
    k_list=(1,3,5,10),
    batch_size=512,
    threshold=0.5,
    as_probs="sigmoid",
    device="cpu",
    loss_fn=None,
):
    '''
    Stream metrics per task using TAMLEC taxonomy structure.

    Predictions are read from a CSR ``.npz`` file and labels from an ``X_Y.txt``
    file, both streamed in batches to avoid loading everything in memory.
    Task-to-label indices are derived from ``taxos_tamlec`` and label index 0
    is skipped (indices are shifted by -1).

    :param pred_npz_path: Path to the CSR ``.npz`` predictions file.
    :param xy_path: Path to the ``X_Y.txt`` labels file.
    :param taxos_tamlec: Taxonomy data (e.g., from ``torch.load(.../taxos_tamlec.pt)``).
    :param k_list: Iterable of k values for top-k metrics.
    :param batch_size: Batch size for streaming predictions/labels.
    :param threshold: Decision threshold for binarizing predictions.
    :param as_probs: If "sigmoid", apply sigmoid to logits; if None, keep as-is.
    :param device: Device for metric computation.
    :param loss_fn: Optional loss function accepting (predictions, labels).
    :return: Dict mapping task_id -> metrics dict.
    '''
    pred_csr = load_npz(pred_npz_path)
    n_docs, n_labels = pred_csr.shape

    # task -> label indices (exactly like your _task_relevant_labels_for)
    task_to_idx = {}
    for task_id in range(len(taxos_tamlec)):
        idxs = []
        for _, children in taxos_tamlec[task_id][1].items():
            idxs.extend(children)
        if len(idxs) == 0:
            continue
        task_to_idx[task_id] = torch.tensor(sorted(set(idxs)), dtype=torch.long)

    accs = {
        task: StreamingXMLMetrics(
            k_list=k_list,
            n_labels=len(idxs),
            loss_fn=loss_fn,
            threshold=threshold,
            device=device,
        )
        for task, idxs in task_to_idx.items()
    }

    pred_it = iter_prediction_batches_from_csr(pred_csr, batch_size=batch_size, as_probs=as_probs)
    lab_it  = iter_label_batches_xy_txt(xy_path, n_labels=n_labels, batch_size=batch_size)

    seen = 0
    for preds, labs in zip(pred_it, lab_it):
        for task, idxs in task_to_idx.items():
            idxs = idxs - 1 # Skip label 0
            accs[task].update(preds[:, idxs], labs[:, idxs])
        seen += preds.size(0)

    assert seen == n_docs
    return {task: acc.compute()[0] for task, acc in accs.items()}

# Functions to stream data from csr and npz + X_Y.txt files (for https://github.com/Extreme-classification/* models)

def iter_label_batches_xy_txt(xy_path: str, n_labels: int, batch_size: int):
    """
    Streams labels from the X_Y.txt format you showed:
      first line: "N L"
      next N lines: "idx:val idx:val ..."
    Yields dense torch.FloatTensor [B, L] with 0/1 values.
    """
    with open(xy_path, "r") as f:
        header = f.readline().strip().split()
        if len(header) >= 2:
            # optionally verify n_labels matches
            # N, L = map(int, header[:2])
            pass

        batch = torch.zeros((batch_size, n_labels), dtype=torch.float32)
        b = 0

        for line in f:
            line = line.strip()
            if line:
                for tok in line.split():
                    # "5:1.0"
                    i_str, v_str = tok.split(":")
                    batch[b, int(i_str)] = 1.0  
            b += 1

            if b == batch_size:
                yield batch
                batch.zero_()
                b = 0

        if b > 0:
            yield batch[:b].clone()

def iter_prediction_batches_from_csr(csr, batch_size: int, as_probs = None):
    """
    Streams predictions from a scipy.sparse CSR matrix (already loaded).
    Yields dense torch.FloatTensor [B, L].
      as_probs:
        - "sigmoid": apply sigmoid (good if csr stores logits)
        - None: no transform (good if csr stores probabilities already)
    """
    n = csr.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # CSR slicing keeps it sparse; densify only this chunk
        dense = csr[start:end].toarray().astype(np.float32)  # [B, L]
        preds = torch.from_numpy(dense)

        if as_probs == "sigmoid":
            preds = torch.sigmoid(preds)
        elif as_probs is None:
            pass
        else:
            raise ValueError(f"Unknown as_probs={as_probs}")

        yield preds

def stream_metrics_from_npz_and_xy(
    pred_npz_path: str,
    xy_path: str,
    k_list=(1, 3, 5),
    batch_size=512,
    threshold=0.5,
    as_probs="sigmoid",   
    device="cpu",
    loss_fn=None,
):
    '''
    Stream overall XML metrics from predictions (.npz) and labels (X_Y.txt).

    :param pred_npz_path: Path to the CSR ``.npz`` predictions file.
    :type pred_npz_path: str
    :param xy_path: Path to the ``X_Y.txt`` labels file.
    :type xy_path: str
    :param k_list: Iterable of k values for top-k metrics.
    :param batch_size: Batch size for streaming predictions/labels.
    :param threshold: Decision threshold for binarizing predictions.
    :param as_probs: If "sigmoid", apply sigmoid to logits; if None, keep as-is.
    :param device: Device for metric computation.
    :param loss_fn: Optional loss function accepting (predictions, labels).
    :return: Output of ``StreamingXMLMetrics.compute()`` (metrics and counts).
    '''
    pred_csr = load_npz(pred_npz_path)  # CSR
    n_docs, n_labels = pred_csr.shape

    acc = StreamingXMLMetrics(
        k_list=k_list,
        n_labels=n_labels,
        loss_fn=loss_fn,
        threshold=threshold,
        device=device,
    )

    pred_iter = iter_prediction_batches_from_csr(pred_csr, batch_size=batch_size, as_probs=as_probs)
    lab_iter  = iter_label_batches_xy_txt(xy_path, n_labels=n_labels, batch_size=batch_size)

    seen = 0
    for preds, labs in zip(pred_iter, lab_iter):
        if preds.shape != labs.shape:
            raise RuntimeError(f"Shape mismatch: preds {preds.shape} vs labs {labs.shape}")

        acc.update(preds, labs)
        seen += preds.size(0)

    if seen != n_docs:
        # zip() stops at the shorter iterator
        raise RuntimeError(f"Processed {seen} docs, but predictions have {n_docs}. Check label file length.")

    return acc.compute()

def stream_metrics_per_level_from_npz_and_xy(
    pred_npz_path: str,
    xy_path: str,
    taxonomy,
    k_list=(1,3,5),
    batch_size=512,
    threshold=0.5,
    as_probs="sigmoid",
    device="cpu",
    loss_fn=None,
):
    '''
    Stream metrics per taxonomy level (excluding root).

    Labels are filtered per level using ``taxonomy.level_to_labels`` and
    ``taxonomy.label_to_idx``. Label index 0 is skipped (indices shifted by -1).

    :param pred_npz_path: Path to the CSR ``.npz`` predictions file.
    :type pred_npz_path: str
    :param xy_path: Path to the ``X_Y.txt`` labels file.
    :type xy_path: str
    :param taxonomy: Taxonomy object with ``level_to_labels`` and ``label_to_idx``.
    :param k_list: Iterable of k values for top-k metrics.
    :param batch_size: Batch size for streaming predictions/labels.
    :param threshold: Decision threshold for binarizing predictions.
    :param as_probs: If "sigmoid", apply sigmoid to logits; if None, keep as-is.
    :param device: Device for metric computation.
    :param loss_fn: Optional loss function accepting (predictions, labels).
    :return: Dict mapping level -> metrics dict.
    '''
    pred_csr = load_npz(pred_npz_path)
    n_docs, n_labels = pred_csr.shape

    # precompute label index tensors per level (on CPU; small)
    level_to_idx = {}
    for level, nodes in taxonomy.level_to_labels.items():
        if level == 0:
            continue
        idx = [taxonomy.label_to_idx[lab] for lab in nodes]
        level_to_idx[level] = torch.tensor(idx, dtype=torch.long)

    # one StreamingXMLMetrics per level
    accs = {
        level: StreamingXMLMetrics(
            k_list=k_list,
            n_labels=len(idxs),
            loss_fn=loss_fn,
            threshold=threshold,
            device=device,
        )
        for level, idxs in level_to_idx.items()
    }

    pred_iter = iter_prediction_batches_from_csr(pred_csr, batch_size=batch_size, as_probs=as_probs)
    lab_iter  = iter_label_batches_xy_txt(xy_path, n_labels=n_labels, batch_size=batch_size)

    seen = 0
    for preds, labs in zip(pred_iter, lab_iter):
        for level, idxs in level_to_idx.items():
            idxs = idxs - 1
            p = preds[:, idxs]
            y = labs[:, idxs]
            accs[level].update(p, y)
        seen += preds.size(0)

    if seen != n_docs:
        raise RuntimeError(f"Processed {seen} docs, but predictions have {n_docs}.")

    return {level: acc.compute()[0] for level, acc in accs.items()}  # just metrics per level

def stream_completion_metrics_per_level_from_npz_and_xy(
    pred_npz_path: str,
    xy_path: str,
    taxonomy,
    k_list=(1,3,5),
    batch_size=512,
    threshold=0.5,
    as_probs="sigmoid",
    device="cpu",
    loss_fn=None,
):
    '''
    Stream completion metrics per taxonomy level (levels > 1 only).

    Label indices are filtered by level and shifted by -1 to skip label 0.

    :param pred_npz_path: Path to the CSR ``.npz`` predictions file.
    :type pred_npz_path: str
    :param xy_path: Path to the ``X_Y.txt`` labels file.
    :type xy_path: str
    :param taxonomy: Taxonomy object with ``level_to_labels`` and ``label_to_idx``.
    :param k_list: Iterable of k values for top-k metrics.
    :param batch_size: Batch size for streaming predictions/labels.
    :param threshold: Decision threshold for binarizing predictions.
    :param as_probs: If "sigmoid", apply sigmoid to logits; if None, keep as-is.
    :param device: Device for metric computation.
    :param loss_fn: Optional loss function accepting (predictions, labels).
    :return: Dict mapping level -> metrics dict.
    '''
    pred_csr = load_npz(pred_npz_path)
    n_docs, n_labels = pred_csr.shape

    # precompute label index tensors per level (on CPU; small)
    level_to_idx = {}
    for level, nodes in taxonomy.level_to_labels.items():
        if level <= 1:
            continue
        idx = [taxonomy.label_to_idx[lab] for lab in nodes]
        level_to_idx[level] = torch.tensor(idx, dtype=torch.long)

    # one StreamingXMLMetrics per level
    accs = {
        level: StreamingXMLMetrics(
            k_list=k_list,
            n_labels=len(idxs),
            loss_fn=loss_fn,
            threshold=threshold,
            device=device,
        )
        for level, idxs in level_to_idx.items()
    }

    pred_iter = iter_prediction_batches_from_csr(pred_csr, batch_size=batch_size, as_probs=as_probs)
    lab_iter  = iter_label_batches_xy_txt(xy_path, n_labels=n_labels, batch_size=batch_size)

    seen = 0
    for preds, labs in zip(pred_iter, lab_iter):
        for level, idxs in level_to_idx.items():
            idxs = idxs - 1
            p = preds[:, idxs]
            y = labs[:, idxs]
            accs[level].update(p, y)
        seen += preds.size(0)

    if seen != n_docs:
        raise RuntimeError(f"Processed {seen} docs, but predictions have {n_docs}.")

    return {level: acc.compute()[0] for level, acc in accs.items()}  # just metrics per level

def stream_completion_metrics_from_npz_and_xy(
    pred_npz_path: str,
    xy_path: str,
    taxonomy,
    k_list=(1, 3, 5),
    batch_size=512,
    threshold=0.5,
    as_probs="sigmoid",
    device="cpu",
    loss_fn=None,
):
    '''
    Stream completion metrics over all labels with level > 1.

    :param pred_npz_path: Path to the CSR ``.npz`` predictions file.
    :type pred_npz_path: str
    :param xy_path: Path to the ``X_Y.txt`` labels file.
    :type xy_path: str
    :param taxonomy: Taxonomy object with ``level_to_labels`` and ``label_to_idx``.
    :param k_list: Iterable of k values for top-k metrics.
    :param batch_size: Batch size for streaming predictions/labels.
    :param threshold: Decision threshold for binarizing predictions.
    :param as_probs: If "sigmoid", apply sigmoid to logits; if None, keep as-is.
    :param device: Device for metric computation.
    :param loss_fn: Optional loss function accepting (predictions, labels).
    :return: Output of ``StreamingXMLMetrics.compute()`` (metrics and counts).
    '''
    pred_csr = load_npz(pred_npz_path)
    n_docs, n_labels = pred_csr.shape

    # global "completion": keep only labels with level > 1
    keep = []
    for level, nodes in taxonomy.level_to_labels.items():
        if level <= 1:
            continue
        keep.extend(taxonomy.label_to_idx[lab] for lab in nodes)

    keep = torch.tensor(keep, dtype=torch.long) - 1  
    acc = StreamingXMLMetrics(
        k_list=k_list,
        n_labels=len(keep),
        loss_fn=loss_fn,
        threshold=threshold,
        device=device,
    )

    pred_iter = iter_prediction_batches_from_csr(pred_csr, batch_size=batch_size, as_probs=as_probs)
    lab_iter  = iter_label_batches_xy_txt(xy_path, n_labels=n_labels, batch_size=batch_size)

    seen = 0
    for preds, labs in zip(pred_iter, lab_iter):
        p = preds[:, keep]
        y = labs[:, keep]
        acc.update(p, y)
        seen += preds.size(0)

    if seen != n_docs:
        raise RuntimeError(f"Processed {seen} docs, but predictions have {n_docs}. Check label file length.")

    return acc.compute()
