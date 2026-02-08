import torch
import numpy as np
from sklearn.cluster import KMeans
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from models.cascadexml import CascadeXML
from models.model_loader import ModelLoader
from misc import utils
from algorithms.base_algorithm import BaseAlg
from pathlib import Path

class CascadexmlAlg(BaseAlg):
    """CascadeXML algorithm wrapper with label clustering."""
    def __init__(
            self,  
            metrics_handler,
            taxonomy,
            optimizer, # Optimizer class from torch.optim
            loss_function: torch.nn.Module, # Any torch loss function (e.g. torch.nn.BCELoss()
            embeddings_loader, # Embeddings loader to initialize the BaseBertModel
            state_dict_path: str = "", # Path to a state dict to load the model from
            state_dict_finetuned_path: str = "", # Path to a state dict to load the finetuned model from
            model_path: Path = Path(), # Path to a model to load the model from
            model_finetuned_path: str = "", # Path to a model to load the finetuned model from
            patience: int = 10, # Maximum number of epochs without improvement before early stopping
            all_tasks_key: str = "", # Key used to store global (all-tasks
            selected_task: int = 0, # Task index to fine-tune if few-shot mode is enabled
            learning_rate: float = 5e-5, # Learning rate for the optimizer
            emb_dim: int = 300, # Embedding dimension for the BaseBertModel
            device: str = 'cpu', # Device to use (cpu, cuda:0,
            model_name: str = "match", # backbone model name to load
            checkpoint_manager = None,
            resume: bool = True,
            min_improvement: float = .0,
            **kwargs
            
            ): 
        """Initialize CascadeXML model and build label clusters."""
        self.taxonomy = taxonomy
        super().__init__(
            metrics_handler,
            taxonomy=taxonomy,
            loss_function = loss_function, # Any torch loss function (e.g. torch.nn.BCELoss() 
            method = "cascadexml", # Identifier of the model/method (e.g., "lightxml").
            device = device, # Device to use (cpu, cuda:0, ...)
            state_dict_path = state_dict_path, # Path to a state dict to load the model from
            state_dict_finetuned_path = state_dict_finetuned_path, # Path to a state dict to load the finetuned model from
            model_path = model_path, # Path to a model to load the model from
            model_finetuned_path = model_finetuned_path, # Path to a model to load the finetuned model from
            verbose = True,
            patience = patience, # Maximum number of epochs without improvement before early stopping
            all_tasks_key = all_tasks_key, # Key used to store global (all-tasks) metrics
            selected_task = selected_task, # Task index to fine-tune if few-shot mode is enabled
            checkpoint_manager = checkpoint_manager,
            resume = resume,
            min_improvement = min_improvement
        )
        
        self.device = device
        self.learning_rate = learning_rate
        print(f"> Get label embeddings...")
        self.backbone = ModelLoader.get_model(model_name, self.method, emb_dim, device, embeddings=embeddings_loader)
        self.label_embeddings = self.labels_tfidf()

        print(f"> Creating clusters for {len(self.label_embeddings)} labels (root excluded)...")
        # Create two levels of clusters
        # Fine level, so just above all labels
        # Original paper makes a clustering with maximum size of 2 by cluster, so we do the same here
        fine_cluster, fine_centers, self.fine_cluster_map = self.get_even_clusters(self.label_embeddings, cluster_size=2)
        fine_cluster = dict(sorted(fine_cluster.items(), key=lambda x: x[0]))
        print(f">> Got {len(fine_centers)} clusters at level 2")
        # Coarse level, first level
        coarse_cluster, coarse_centers, self.coarse_cluster_map = self.get_even_clusters(fine_centers, cluster_size=2)
        coarse_cluster = dict(sorted(coarse_cluster.items(), key=lambda x: x[0]))
        print(f">> Got {len(coarse_centers)} clusters at level 1")
        self.clusters = [list(coarse_cluster.values()), list(fine_cluster.values())]

        print(f"> Loading model...")
        self.model = CascadeXML(
            taxonomy=self.taxonomy,
            emb_dim=emb_dim,
            device=self.device,
            backbone=self.backbone,
            clusters=self.clusters,
        )
        self.model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)
        self.optimizer_class = optimizer
        

    # From https://stackoverflow.com/questions/5452576/k-means-algorithm-variation-with-equal-cluster-size?rq=1
    def get_even_clusters(self, X, cluster_size):
        """Cluster vectors into roughly equal-sized clusters."""
        n_clusters = int(np.ceil(len(X)/cluster_size))
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, verbose=1, random_state=16)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, X.shape[-1]).repeat(cluster_size, 1).reshape(-1, X.shape[-1])
        distance_matrix = cdist(X, centers)
        clusters = linear_sum_assignment(distance_matrix)[1]//cluster_size
        clustering = {}
        cluster_map = {}
        for idx, cluster_idx in enumerate(clusters):
            try:
                clustering[cluster_idx].append(idx)
            except KeyError:
                clustering[cluster_idx] = [idx]
            cluster_map[idx] = cluster_idx
        centers = np.array([X[clusters == i].mean(axis=0) for i in range(n_clusters)])

        # Cluster map is a tensor so we can use vectorized lookup with other tensors
        cluster_map = dict(sorted(cluster_map.items(), key=lambda x: x[0]))
        cluster_map = torch.tensor(list(cluster_map.values()), dtype=torch.int64, device=self.device)

        return clustering, centers, cluster_map


    def labels_tfidf(self):
        """Compute TF-IDF label embeddings from label abstracts."""
        plain_texts = []
        # Root is skipped since it is never predicted
        # Get all abstracts from the labels
        for idx in range(1, self.taxonomy.n_nodes):
            label = self.taxonomy.idx_to_label[idx]
            abstract = self.taxonomy.label_to_abstract[label]
            abstract = abstract.lower().strip().replace('\n', '')
            # Remove all special characters, keep only letters, numbers and spaces
            abstract = re.sub(r"[^\w\s]", '', abstract)
            plain_texts.append(abstract)
        vectorizer = TfidfVectorizer()
        lab_embs = vectorizer.fit_transform(plain_texts)
        # Transform SciPy sparse matrix to Numpy array
        lab_embs = np.asarray(lab_embs.todense())

        return lab_embs


    def load_model(self):
        """Load a saved state dict and reinitialize the optimizer."""
        with open(self.state_dict_path, "rb") as f:
            self.model.load_state_dict(torch.load(f, weights_only=True, map_location=self.device))
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
    
    def load_model_finetuned(self):
        """Load a finetuned state dict and reinitialize the optimizer."""
        with open(self.state_dict_finetuned_path, "rb") as f:
            self.model.load_state_dict(torch.load(f, weights_only=True, map_location=self.device))
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)

    def inference_eval(self, input_data, **kwargs):
        """Run a forward pass and return dense predictions."""
        # Set model to eval() modifies the behavior of certain layers e.g. dropout, normalization layers
        self.model.eval()
        all_probs, all_candidates, _ = self.model(input_data)
        predictions = []
        # -1 so we take predictions of the last level of the clustering, i.e. our class labels
        for pred, score in zip(all_candidates[-1], all_probs[-1]):
            assert len(pred) == len(score)
            # Highest index is a <PAD> class, this is because of the balanced clustering
            # and the way the code is written in the original model
            # Trick is to have one more class representing the <PAD>
            # And then make sure it is equal to 0
            pred_sample = torch.zeros(self.taxonomy.n_nodes+1, dtype=torch.float32)
            # Shift back to our class indices, since we did not cluster the root node
            pred += 1
            pred_sample.scatter_(dim=0, index=pred.cpu(), src=score.cpu())
            # <PAD>  probability should be zero
            assert pred_sample[-1] == 0.
            # Also check that the root has a zero probability, so that the shift is working
            assert pred_sample[0] == 0.
            # Remove the padding probability
            predictions.append(pred_sample[:-1])
        predictions = torch.stack(predictions, dim=0)

        return predictions


    def optimization_loop(self, input_data, labels):
        """Run one training step with multi-level CascadeXML labels."""
        # Model in training mode, e.g. activate dropout if specified
        self.model.train()
        # Labels for cascadexml are a list of lists of tensors
        # There is one list per level in the clustering (first list is for highest level, last list is all class labels)
        # Each list is of size batch_size
        coarse_level = []
        fine_level = []
        class_level = []
        for label in labels:
            class_labels = (label == 1.).nonzero(as_tuple=True)[0]
            # Remove the root
            class_labels = class_labels[class_labels != 0]
            class_labels -= 1
            class_level.append(class_labels)
            # Construct labels for the other levels in the clustering
            fine_labels = torch.unique(self.fine_cluster_map[class_labels])
            coarse_labels = torch.unique(self.coarse_cluster_map[fine_labels])
            fine_level.append(fine_labels)
            coarse_level.append(coarse_labels)
        cascadexml_labels = [coarse_level, fine_level, class_level]

        # Get predictions
        all_probs, all_candidates, loss = self.model(input_data, all_labels=cascadexml_labels)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Compute gradient norms
        gradient_norms = []
        gradient_lengths = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradient_norms.append(torch.norm(param.grad).cpu())
                gradient_lengths.append(len(param.grad))
        self.optimizer.step()
        return loss.cpu().item(), (gradient_norms, gradient_lengths)


    def run_init(self):
        """No-op initialization hook for CascadeXML."""
        pass
