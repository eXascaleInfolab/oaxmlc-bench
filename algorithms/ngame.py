
import torch
import numpy as np
import warnings
import shutil
from algorithms.base_algorithm import AbstractAlgorithm
from models.NGAME.ngame.runner import run_ngame
from pathlib import Path
import os

class NgameAlg(AbstractAlgorithm):
    """NGAME algorithm wrapper for running the external NGAME pipeline."""
    def __init__(
            self,
            taxonomy_path,
            data_path,
            batch_size_siamese,
            batch_size_extreme,
            voc_size,
            tokenization_mode,
            seed,
            n_doc_train,
            n_doc_test,
            output_dir,
            seq_length = 32,
            device = 'cpu',
            fewshot_exp = False,
            selected_task = None,
            task_to_subroot = None,
            #metrics_handler,
            **kwargs,
            ): 
        """Initialize NGAME configuration parameters and ANN settings."""
        self.device = device
        # set device as only cuda device visible
        if self.device.startswith("cuda"):
            # set CUDA_VISIBLE_DEVICES for CUDA devices
            os.environ["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1]
            print(f"> Using device: {self.device}, CUDA_VISIBLE_DEVICES={torch.cuda.device_count()}")
        elif self.device == "mps":
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                print(f"> Using device: {self.device} (MPS available)")
            else:
                warnings.warn("MPS requested but not available, falling back to CPU")
                self.device = "cpu"
                print("> Using device: cpu")
        else:
            print("> Using device: cpu")
        self.task_to_subroot = task_to_subroot
        self.fewshot_exp = fewshot_exp
        self.selected_task = selected_task
        self.batch_size_siamese = batch_size_siamese
        self.batch_size_extreme = batch_size_extreme
        self.seed = seed
        self.data_path = data_path
        self.taxonomy = torch.load(taxonomy_path)
        self.n_doc_train = n_doc_train
        self.n_doc_test = n_doc_test
        self.seq_length = seq_length
        self.voc_size = voc_size
        self.tokenization_mode = tokenization_mode
        self.encoder_name = "sentence-transformers/msmarco-distilbert-base-v4"
        self.output_dir = output_dir
        
        def floor_pow2(n: int) -> int:
            return 1 << (n.bit_length() - 1)

        self.L = self.taxonomy.n_nodes - 1
        self.H = 32
        self.V = max(1, self.L - self.H)
        self.C = floor_pow2(self.V)

        self.n_train = int(self.n_doc_train)
        self.n_test  = int(self.n_doc_test)

        # ---- robust k clamp  ----
        self.requested_k = 100

        # k must be feasible for ANY index that might be queried.
        # In your case there is definitely a label ANN index of size L (=197).
        self.k_label = max(1, min(self.requested_k, self.L - 1))

        # if also using a doc index, clamp that too (usually much bigger)
        self.k_doc = max(1, min(self.requested_k, self.n_train - 1))

        # If NGAME uses ONE num_nbrs for both, you must use the minimum => k_label
        self.safe_k = min(self.k_label, self.k_doc)

        # ---- very safe HNSW params ----
        self.M = 256
        self.efS = max(4000, 30 * self.safe_k)   # "search" ef (must be >> k)
        self.efC = self.efS                       # "construction" ef
        
    def run_init(self) -> None:
        """Optional extra initialization hook (tokenizers, caches, etc.)."""
        pass
    
    def train(self, **args):
        """
        Run the NGAME training pipeline.
        """
        

        
        # -------------------------
        # Base config 
        # -------------------------
        config = {
            "global": {
                "dataset": self.data_path.stem,
                "feature_type": "sequential",
                "num_labels": self.L,
                "arch": "STransformer",
                "A": 0.6,
                "B": 2.6,
                "share_weights": True,
                "surrogate_method": 1,
                "max_length": 32,
                "embedding_dims": 768,
                "top_k": min(100, self.L),
                "num_points": self.n_train + self.n_test,
                "beta": 1.0,
                "eval_method": "score_fusion",
                "encoder_name": "msmarco-distilbert-base-v4",
                "save_predictions": True,

                
                "lbl_feat_fname": f"preprocessed_data/seed_{self.seed}/lbl_input_ids.npy,preprocessed_data/seed_{self.seed}/lbl_attention_mask.npy",
                "trn_label_fname": f"preprocessed_data/seed_{self.seed}/trn_X_Y.txt",
                "trn_feat_fname": f"preprocessed_data/seed_{self.seed}/trn_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/trn_doc_attention_mask.npy",
                "trn_filter_fname": f"preprocessed_data/seed_{self.seed}/filter_labels_train.txt",

                "val_label_fname": f"preprocessed_data/seed_{self.seed}/tst_X_Y.txt",
                "val_feat_fname": f"preprocessed_data/seed_{self.seed}/tst_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/tst_doc_attention_mask.npy",
                "val_filter_fname": f"preprocessed_data/seed_{self.seed}/filter_labels_test.txt",

                "tst_label_fname": f"preprocessed_data/seed_{self.seed}/tst_X_Y.txt",
                "tst_feat_fname": f"preprocessed_data/seed_{self.seed}/tst_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/tst_doc_attention_mask.npy",
                "tst_filter_fname": f"preprocessed_data/seed_{self.seed}/filter_labels_test.txt",
            },
            "siamese": {
                "num_epochs": 100,
                "warmup_steps": 50,
                "batch_type": "doc",
                "loss": "triplet_margin_ohnm",
                "model_type": "siamese",
                "network_type": "siamese",
                "metric": "cosine",
                "validate_after": 50,
                "learning_rate": 0.0002,
                "batch_size": self.batch_size_siamese,
                "normalize": True,
                "margin": 0.3,
                "optim": "AdamW",
                "loss_num_positives": 1,
                "loss_num_negatives": 10,
                "loss_agressive": True,
                "init": "auto",
                "validate": True,
                "save_intermediate": True,
                "sampling_asynchronous": True,
                "sampling_type": "implicit",
                "sampling_curr_epochs": [10, 25, 50, 75, 100, 125, 150, 200],
                "sampling_refresh_interval": 5,
                "sampling_init_cluster_size": 1,
                "sampling_threads": 8,
                "use_amp": True,
                "weight_decay": 0.01,
                "inference_method": None,
            },
            "extreme": {
                "num_epochs": 50,
                "warmup_steps": 200,
                "batch_type": "doc",
                "optim": "Adam",
                "loss": "triplet_margin_ohnm",
                "loss_num_positives": 3,
                "loss_num_negatives": 5,
                "loss_agressive": True,
                "learning_rate": 0.001,
                "batch_size": self.batch_size_extreme,
                "sampling_curr_epochs": [10, 15, 20, 25, 30, 35, 40],
                "sampling_refresh_interval": 5,
                "sampling_type": "implicit",
                "num_centroids": 1,

                # We'll override efC/efS/M/num_nbrs safely below
                "ann_method": "hnswlib",
                "ann_threads": 12,
                "num_nbrs": self.safe_k,
                "num_neighbours": self.safe_k,
                "M": self.M,
                "efS": self.efS,
                "efC": self.efC,

                "margin": 0.1,
                "ann_threads": 12,
                "beta": 0.5,
                "surrogate_mapping": None,
                "freeze_encoder": True,
                "validate": True,
                "model_type": "xc",
                "network_type": "xc",
                "normalize": True,
                "init": "intermediate",
                "save_intermediate": False,
                "inference_method": "dual_mips",
                "use_amp": False,
            },
        }

        
        # bump version so NGAME doesn't reuse an old run dir with old args
        version = f"{0}_{self.seed}_k{self.safe_k}_M{self.M}_efS{self.efS}"

        if self.fewshot_exp:
            version += f"_fewshot_task{self.selected_task}"
            # Create temporary masked data files
            self._create_fewshot_masked_files(phase="train")
            # Update config to point to masked files
            config["global"]["trn_label_fname"] = f"preprocessed_data/seed_{self.seed}/fewshot_trn_X_Y.txt"
            config["global"]["trn_feat_fname"] = f"preprocessed_data/seed_{self.seed}/fewshot_trn_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/fewshot_trn_doc_attention_mask.npy"
        
        print(f"[NGAME ANN] n_train={self.n_train}, L={self.L}, num_nbrs={self.safe_k}, M={self.M}, efS={self.efS}, efC={self.efC}")

        
        
        run_ngame(
            pipeline="ngame",
            data_dir="datasets/",
            work_dir="./models/NGAME/ngame/",
            output_dir = self.output_dir,
            version=version,
            seed=self.seed,
            config=config,
        )
        
    def finetune(self, **args):
        """Fine-tune NGAME on held-out task documents."""
        if not self.fewshot_exp:
            return
        
        # Create masked fine-tune data
        self._create_fewshot_masked_files(phase="finetune")
        
        # Get the trained model directory from the training run
        trained_model_dir = self.output_dir  # From previous train() call
        
        # Create fine-tune output dir
        finetune_output_dir = self.output_dir / "finetune"
        finetune_output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Create fine-tune config
        finetune_config = {
            "global": {
                "dataset": self.data_path.stem,
                "feature_type": "sequential",
                "num_labels": self.L,
                "arch": "STransformer",
                "A": 0.6,
                "B": 2.6,
                "share_weights": True,
                "surrogate_method": 1,
                "max_length": 32,
                "embedding_dims": 768,
                "top_k": min(100, self.L),
                "num_points": self.n_train + self.n_test,
                "beta": 1.0,
                "eval_method": "score_fusion",
                "encoder_name": "msmarco-distilbert-base-v4",
                "save_predictions": True,

                "trn_label_fname": f"preprocessed_data/seed_{self.seed}/fewshot_finetune_X_Y.txt",
                "trn_filter_fname": f"preprocessed_data/seed_{self.seed}/filter_labels_train.txt",

                "trn_feat_fname": f"preprocessed_data/seed_{self.seed}/fewshot_finetune_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/fewshot_finetune_doc_attention_mask.npy",
                "lbl_feat_fname": f"preprocessed_data/seed_{self.seed}/lbl_input_ids.npy,preprocessed_data/seed_{self.seed}/lbl_attention_mask.npy",
                "val_label_fname": f"preprocessed_data/seed_{self.seed}/tst_X_Y.txt",
                "val_feat_fname": f"preprocessed_data/seed_{self.seed}/tst_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/tst_doc_attention_mask.npy",
                "val_filter_fname": f"preprocessed_data/seed_{self.seed}/filter_labels_test.txt",

                "tst_label_fname": f"preprocessed_data/seed_{self.seed}/tst_X_Y.txt",
                "tst_feat_fname": f"preprocessed_data/seed_{self.seed}/tst_doc_input_ids.npy,preprocessed_data/seed_{self.seed}/tst_doc_attention_mask.npy",
                "tst_filter_fname": f"preprocessed_data/seed_{self.seed}/filter_labels_test.txt",
            },
            "siamese": {
                "num_epochs": 5,
                "warmup_steps": 2,
                "batch_type": "doc",
                "loss": "triplet_margin_ohnm",
                "model_type": "siamese",
                "network_type": "siamese",
                "metric": "cosine",
                "validate_after": 5,
                "learning_rate": 0.00002,
                "batch_size": self.batch_size_siamese,
                "normalize": True,
                "margin": 0.3,
                "optim": "AdamW",
                "loss_num_positives": 1,
                "loss_num_negatives": 10,
                "loss_agressive": True,
                "init": "auto",
                "validate": True,
                "save_intermediate": True,
                "sampling_asynchronous": True,
                "sampling_type": "implicit",
                "sampling_curr_epochs": [10, 25, 50, 75, 100, 125, 150, 200],
                "sampling_refresh_interval": 5,
                "sampling_init_cluster_size": 1,
                "sampling_threads": 8,
                "use_amp": True,
                "weight_decay": 0.01,
                "inference_method": None,
            },
            "extreme": {
                "num_epochs": 5,
                "warmup_steps": 2,
                "batch_type": "doc",
                "optim": "Adam",
                "loss": "triplet_margin_ohnm",
                "loss_num_positives": 3,
                "loss_num_negatives": 5,
                "loss_agressive": True,
                "learning_rate": 0.001,
                "batch_size": self.batch_size_extreme,
                "sampling_curr_epochs": [10, 15, 20, 25, 30, 35, 40],
                "sampling_refresh_interval": 5,
                "sampling_type": "implicit",
                "num_centroids": 1,

                # We'll override efC/efS/M/num_nbrs safely below
                "ann_method": "hnswlib",
                "ann_threads": 12,
                "num_nbrs": self.safe_k,
                "num_neighbours": self.safe_k,
                "M": self.M,
                "efS": self.efS,
                "efC": self.efC,

                "margin": 0.1,
                "ann_threads": 12,
                "beta": 0.5,
                "surrogate_mapping": None,
                "freeze_encoder": True,
                "validate": True,
                "model_type": "xc",
                "network_type": "xc",
                "normalize": True,
                "init": "intermediate",
                "save_intermediate": False,
                "inference_method": "dual_mips",
                "use_amp": False,
            },
        }
        
        
        
        run_ngame(
            pipeline="ngame",
            data_dir="datasets/",
            work_dir="./models/NGAME/ngame/",
            output_dir=finetune_output_dir,
            version=f"{self.selected_task}_finetune",
            seed=self.seed,
            config=finetune_config,
            checkpoint_dir=trained_model_dir,
        )
        
        
        

    def save_model(self) -> None:
        """NGAME manages model artifacts internally; no-op."""
        pass
        
    def load_model(self) -> None:
        """
        No-op: NGAME evaluation uses exported artifacts.
        """
        
    
    def load_model_finetuned(self) -> None:
        """No-op: finetuned artifacts handled by NGAME pipeline."""
        pass
    
    @torch.no_grad()
    def inference_eval(self, input_data, **kwargs):
        """
        Placeholder for compatibility; NGAME evaluation is external.
        """ 


    

    
    def _create_fewshot_masked_files(self, phase: str) -> None:
        """
        Create masked versions of training data files.
        phase:
        - "train": exclude docs of selected task
        - "finetune": keep only docs of selected task
        """
        # compute / fetch cached task docs (0-based row indices AFTER header)
        task_docs = getattr(self, "_fewshot_mask_cache", None)
        if task_docs is None:
            task_docs = self._compute_fewshot_mask()

        n_docs = self._count_train_docs()  # number of body lines (docs), excludes header

        if phase == "train":
            mask = np.ones(n_docs, dtype=bool)
            mask[np.asarray(task_docs, dtype=np.int64)] = False
            output_prefix = "fewshot_trn"
        else:
            mask = np.zeros(n_docs, dtype=bool)
            mask[np.asarray(task_docs, dtype=np.int64)] = True
            output_prefix = "fewshot_finetune"

        base_dir = Path(self.data_path) / "preprocessed_data"
        seed_dir = base_dir / f"seed_{self.seed}"
        if not seed_dir.exists():
            seed_dir.mkdir(parents=True, exist_ok=True)
            to_copy = [
                "lbl_input_ids.npy","lbl_attention_mask.npy",
                "trn_X_Y.txt","trn_doc_input_ids.npy","trn_doc_attention_mask.npy","filter_labels_train.txt",
                "tst_X_Y.txt","tst_doc_input_ids.npy","tst_doc_attention_mask.npy","filter_labels_test.txt",
            ]
            for fname in to_copy:
                src = base_dir / fname
                dst = seed_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
        data_dir = seed_dir
        # labels
        self._write_masked_labels(
            input_file=data_dir / "trn_X_Y.txt",
            output_file=data_dir / f"{output_prefix}_X_Y.txt",
            mask=mask,
        )

        # features
        self._write_masked_features(
            input_file=data_dir / "trn_doc_input_ids.npy",
            output_file=data_dir / f"{output_prefix}_doc_input_ids.npy",
            mask=mask,
        )
        self._write_masked_features(
            input_file=data_dir / "trn_doc_attention_mask.npy",
            output_file=data_dir / f"{output_prefix}_doc_attention_mask.npy",
            mask=mask,
        )


    def _write_masked_labels(self, input_file: Path, output_file: Path, mask: np.ndarray) -> None:
        """
        Mask trn_X_Y.txt while preserving format:
        header: "<n_docs> <n_labels>"
        then n_docs lines of sparse labels "k:1.0 ..."
        We must rewrite header with updated n_docs.
        """
        with open(input_file, "r") as f_in:
            header = next(f_in, "").strip()
            if not header:
                raise ValueError(f"Empty label file: {input_file}")
            parts = header.split()
            if len(parts) < 2:
                raise ValueError(f"Bad header in {input_file}: {header!r}")
            n_labels = int(parts[1])  # keep same second number

            kept = int(mask.sum())

            with open(output_file, "w") as f_out:
                f_out.write(f"{kept} {n_labels}\n")
                for doc_idx, line in enumerate(f_in):  # doc_idx is 0-based over docs
                    if mask[doc_idx]:
                        f_out.write(line)


    def _write_masked_features(self, input_file: Path, output_file: Path, mask: np.ndarray) -> None:
        features = np.load(input_file)
        # sanity check (optional but helpful)
        if features.shape[0] != mask.shape[0]:
            raise ValueError(
                f"Feature rows ({features.shape[0]}) != #docs in mask ({mask.shape[0]}). "
                f"File: {input_file}"
            )
        np.save(output_file, features[mask])


    def _compute_fewshot_mask(self):
        """
        Returns doc indices (0-based, aligned with numpy arrays and body lines of trn_X_Y.txt)
        that contain at least one label in the selected task subtree.
        IMPORTANT: trn_X_Y.txt stores COMPACT ids (lab-1), i.e. 0..L-1.
        """
        # taxonomy ids (full): include task root + descendants, exclude root=0
        task_label_ids_full = self.get_task_label_ids(self.taxonomy, self.selected_task, include_task_itself=True)

        # convert to compact ids used in trn_X_Y.txt: compact = full_id - 1
        task_label_ids_compact = { (lab_id - 1) for lab_id in task_label_ids_full if lab_id != 0 }

        data_dir = Path(self.data_path) / f"preprocessed_data/seed_{self.seed}"
        task_docs: list[int] = []

        with open(data_dir / "trn_X_Y.txt", "r") as f:
            header = next(f, None)  # skip header
            if header is None:
                raise ValueError(f"Empty file: {data_dir / 'trn_X_Y.txt'}")

            for doc_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # tokens like "14:1.0"
                for tok in line.split():
                    lab_compact = int(tok.split(":")[0])
                    if lab_compact in task_label_ids_compact:
                        task_docs.append(doc_idx)
                        break

        self._fewshot_mask_cache = task_docs
        return task_docs


    def _count_train_docs(self) -> int:
        """Counts number of document rows (excluding the header) in trn_X_Y.txt."""
        data_dir = Path(self.data_path) / f"preprocessed_data/seed_{self.seed}"
        with open(data_dir / "trn_X_Y.txt", "r") as f:
            next(f, None)  # header
            return sum(1 for _ in f)


    def get_task_label_ids(self, taxonomy, selected_task: int, include_task_itself: bool = True):
        """
        selected_task is a TASK INDEX (0..T-1), not a label id.
        We map it to the subtree root label id via task_to_subroot.
        Returns FULL taxonomy ids (root=0, others 1..L) of that subtree.
        """
        if self.task_to_subroot is None:
            raise ValueError("DexaAlg needs task_to_subroot to map task index -> subtree root label id")

        if int(selected_task) not in self.task_to_subroot:
            raise ValueError(f"selected_task={selected_task} not found in task_to_subroot keys")

        root_label_id = int(self.task_to_subroot[int(selected_task)])  # FULL label id in taxonomy space
        root_label_name = taxonomy.idx_to_label[root_label_id]

        descendants = taxonomy.all_children(root_label_name)  # label names
        ids = set()
        if include_task_itself:
            ids.add(root_label_id)
        for lab_name in descendants:
            ids.add(taxonomy.label_to_idx[lab_name])
        return ids
