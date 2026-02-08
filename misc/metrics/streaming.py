import torch
import numpy as np
from scipy.stats import hmean
from misc.metrics.metrics import (
    reciprocal_rank,
    minimum_rank_all,
    precision_recall,
    precision_recall_at_k,
    mean_average_precision_at_k,
    ranked_sum_at_k,
    ndcg_k,
)
    
class StreamingXMLMetrics:
    """
    Streaming version of get_xml_metrics that never stores [N_docs, N_labels].

    Usage:
        acc = StreamingXMLMetrics(k_list, n_labels, loss_fn, threshold)
        for each batch:
            acc.update(predictions, labels)
        metrics, n_docs = acc.compute()
    """
    def __init__(self, k_list, n_labels, loss_fn=None, threshold=0.5, device="cpu"):
        self.k_list = list(k_list)
        self.loss_fn = loss_fn
        self.threshold = threshold
        self.device = device

        # Loss accumulation (assume reduction='mean' over elements)
        self.loss_sum = 0.0
        self.loss_weight = 0  # number of elements contributing

        # Ranking metrics
        self.sum_precision_at_k = {k: 0.0 for k in self.k_list}
        self.sum_recall_at_k = {k: 0.0 for k in self.k_list}
        self.sum_f1_at_k = {k: 0.0 for k in self.k_list}
        self.sum_rankedsum_at_k = {k: 0.0 for k in self.k_list}
        self.sum_ndcg_at_k = {k: 0.0 for k in self.k_list}
        self.sum_map_at_k = {k: 0.0 for k in self.k_list}
        self.count_at_k = {k: 0 for k in self.k_list}

        # Micro
        self.micro_tp = 0
        self.micro_fp = 0
        self.micro_fn = 0

        # Macro: per-class counts
        self.macro_tp_per_class = torch.zeros(n_labels, dtype=torch.long)
        self.macro_fp_per_class = torch.zeros(n_labels, dtype=torch.long)
        self.macro_fn_per_class = torch.zeros(n_labels, dtype=torch.long)

        # Reciprocal rank / minimum rank
        self.sum_reciprocal_rank = 0.0
        self.sum_min_rank = 0.0
        self.count_docs = 0  # total docs for those metrics

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update running metrics with a new batch.

        :param predictions: Tensor of shape (B, L) with probabilities in [0, 1].
        :param labels: Binary tensor of shape (B, L).
        :return: None.
        """
        
        assert predictions.shape == labels.shape, (
            f"Shape mismatch: predictions.shape={predictions.shape}, "
            f"labels.shape={labels.shape}"
        )
        B, L = predictions.shape
        self.count_docs += B

        # Move to metrics device (CPU is fine)
        predictions = predictions.to(self.device)
        labels = labels.to(self.device)

        # ----- Loss -----
        if self.loss_fn is not None:
            # assume reduction='mean' over all elements
            loss_batch = self.loss_fn(predictions, labels)
            num_elems = predictions.numel()
            self.loss_sum += loss_batch.item() * num_elems
            self.loss_weight += num_elems

        # ----- Ranking metrics per k -----
        n_labels_per_doc = torch.sum(labels, dim=1)  # [B]

        for k in self.k_list:
            mask = (n_labels_per_doc >= k)
            if mask.sum() == 0:
                continue

            filt_pred = predictions[mask]
            filt_labels = labels[mask]
            n_batch_docs = filt_pred.size(0)

            prec_k, rec_k = precision_recall_at_k(filt_pred, filt_labels, k)
            f1_k = hmean([prec_k, rec_k])

            rs_k = ranked_sum_at_k(filt_pred, filt_labels, k)
            ndcg_k_val = ndcg_k(filt_pred, filt_labels, k)
            map_k_val = mean_average_precision_at_k(filt_pred, filt_labels, k)

            self.sum_precision_at_k[k] += prec_k * n_batch_docs
            self.sum_recall_at_k[k] += rec_k * n_batch_docs
            self.sum_f1_at_k[k] += f1_k * n_batch_docs
            self.sum_rankedsum_at_k[k] += rs_k * n_batch_docs
            self.sum_ndcg_at_k[k] += ndcg_k_val * n_batch_docs
            self.sum_map_at_k[k] += map_k_val * n_batch_docs
            self.count_at_k[k] += n_batch_docs

        # ----- Micro / Macro counts -----
        binary_predictions = (predictions > self.threshold).float()

        preds_plus_labels = binary_predictions + labels
        preds_minus_labels = binary_predictions - labels

        tp = (preds_plus_labels == 2).sum()
        fp = (preds_minus_labels == 1).sum()
        fn = (preds_minus_labels == -1).sum()

        self.micro_tp += tp.item()
        self.micro_fp += fp.item()
        self.micro_fn += fn.item()

        tp_c = (preds_plus_labels == 2).sum(dim=0)
        fp_c = (preds_minus_labels == 1).sum(dim=0)
        fn_c = (preds_minus_labels == -1).sum(dim=0)

        self.macro_tp_per_class += tp_c.cpu()
        self.macro_fp_per_class += fp_c.cpu()
        self.macro_fn_per_class += fn_c.cpu()

        # ----- RR & Min-rank -----
        rr_batch = reciprocal_rank(predictions, labels)
        min_rank_batch = minimum_rank_all(predictions, labels)
        self.sum_reciprocal_rank += rr_batch * B
        self.sum_min_rank += min_rank_batch * B

    def compute(self):
        '''
        Finalize and return accumulated metrics.

        :return: Tuple of (metrics_dict, n_docs_dict) where n_docs_dict records
            the number of samples contributing to each metric.
        '''
        metrics = {}
        n_docs = {}

        # Loss
        if self.loss_fn is not None and self.loss_weight > 0:
            metrics['loss'] = self.loss_sum / self.loss_weight
            n_docs['loss'] = self.count_docs

        # Ranking metrics
        for k in self.k_list:
            if self.count_at_k[k] == 0:
                continue
            metrics[f"precision@{k}"] = self.sum_precision_at_k[k] / self.count_at_k[k]
            n_docs[f"precision@{k}"] = self.count_at_k[k]

            metrics[f"recall@{k}"] = self.sum_recall_at_k[k] / self.count_at_k[k]
            n_docs[f"recall@{k}"] = self.count_at_k[k]

            metrics[f"f1score@{k}"] = self.sum_f1_at_k[k] / self.count_at_k[k]
            n_docs[f"f1score@{k}"] = self.count_at_k[k]

            metrics[f"rankedsum@{k}"] = self.sum_rankedsum_at_k[k] / self.count_at_k[k]
            n_docs[f"rankedsum@{k}"] = self.count_at_k[k]

            metrics[f"ndcg@{k}"] = self.sum_ndcg_at_k[k] / self.count_at_k[k]
            n_docs[f"ndcg@{k}"] = self.count_at_k[k]

            metrics[f"map@{k}"] = self.sum_map_at_k[k] / self.count_at_k[k]
            n_docs[f"map@{k}"] = self.count_at_k[k]

        # Micro
        tp = torch.tensor(self.micro_tp, dtype=torch.float32)
        fp = torch.tensor(self.micro_fp, dtype=torch.float32)
        fn = torch.tensor(self.micro_fn, dtype=torch.float32)

        micro_precision = torch.nan_to_num(tp / (tp + fp), nan=0.0).item()
        micro_recall = torch.nan_to_num(tp / (tp + fn), nan=0.0).item()
        micro_f1 = hmean([micro_precision, micro_recall])

        metrics["micro_precision"] = micro_precision
        metrics["micro_recall"] = micro_recall
        metrics["micro_f1score"] = micro_f1
        n_docs["micro_precision"] = self.count_docs
        n_docs["micro_recall"] = self.count_docs
        n_docs["micro_f1score"] = self.count_docs

        # Macro
        tp_c = self.macro_tp_per_class.to(torch.float32)
        fp_c = self.macro_fp_per_class.to(torch.float32)
        fn_c = self.macro_fn_per_class.to(torch.float32)

        prec_per_class = torch.nan_to_num(tp_c / (tp_c + fp_c), nan=0.0)
        rec_per_class = torch.nan_to_num(tp_c / (tp_c + fn_c), nan=0.0)

        macro_precision = prec_per_class.mean().item()
        macro_recall = rec_per_class.mean().item()
        macro_f1 = hmean([macro_precision, macro_recall])

        metrics["macro_precision"] = macro_precision
        metrics["macro_recall"] = macro_recall
        metrics["macro_f1score"] = macro_f1
        n_docs["macro_precision"] = self.count_docs
        n_docs["macro_recall"] = self.count_docs
        n_docs["macro_f1score"] = self.count_docs

        # Reciprocal rank / min rank
        if self.count_docs > 0:
            metrics["reciprocal_rank"] = self.sum_reciprocal_rank / self.count_docs
            metrics["min_rank"] = self.sum_min_rank / self.count_docs
            n_docs["reciprocal_rank"] = self.count_docs
            n_docs["min_rank"] = self.count_docs

        return metrics, n_docs
