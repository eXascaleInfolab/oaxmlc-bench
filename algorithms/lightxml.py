import torch
from models.LightXML.src.model import LightXML

from models.model_loader import ModelLoader
from algorithms.base_algorithm import BaseAlg
from pathlib import Path
from datahandler.taxonomy import Taxonomy
class LightxmlAlg(BaseAlg):
    """LightXML algorithm wrapper with two-stage candidate handling."""
    def __init__(
            self,  
            metrics_handler,
            taxonomy: Taxonomy ,
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
            min_improvement: float = .02,
            **kwargs            
            ): 
        """Initialize LightXML model, optimizer, and runtime settings."""
        super().__init__(
            metrics_handler,
            taxonomy=taxonomy,
            loss_function = loss_function, # Any torch loss function (e.g. torch.nn.BCELoss() 
            method = "lightxml", # Identifier of the model/method (e.g., "lightxml").
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
        self.taxonomy = taxonomy
        self.learning_rate = learning_rate
        print(f"> Initialize model...")
        self.backbone = ModelLoader.get_model(model_name, self.method, emb_dim, device, embeddings=embeddings_loader)
        # Free the raw embedding tensor once the backbone has its own copy
        del embeddings_loader
        self.model = LightXML(
            n_labels=self.taxonomy.n_nodes,
            bert=self.backbone,
        ).to(device)
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.optimizer_class = optimizer
        self.sigmoid = torch.nn.Sigmoid()

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
    

    @torch.no_grad()
    def inference_eval(self, input_data, **kwargs):
        """Run a forward pass and return dense probabilities."""
        self.model.eval()
        out = self.model(input_data)

        # single-stage → logits tensor
        if isinstance(out, torch.Tensor):
            return torch.sigmoid(out).float()   # <— ensure fp32

        # two-stage → (group_logits, candidates, candidates_scores)
        group_logits, candidates, candidates_scores = out
        candidates_scores = candidates_scores.to(dtype=torch.float32)  # <— fp32
        B, C = candidates_scores.shape
        nL = self.taxonomy.n_nodes
        dense = torch.zeros(B, nL, device=candidates_scores.device, dtype=torch.float32)
        dense.scatter_(1, candidates, candidates_scores)
        return dense  # already fp32

    def optimization_loop(self, input_data, labels):
        """Run one training step and return loss and gradient stats."""
        # Model in training mode, e.g. activate dropout if specified
        self.model.train()
        _logits, loss = self.model(input_data, labels)
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
        """No-op initialization hook for LightXML."""
        pass
