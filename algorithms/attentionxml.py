import torch
import numpy as np
from ruamel.yaml import YAML
from pathlib import Path
from models.AttentionXML.deepxml.networks import AttentionRNN
from models.AttentionXML.deepxml.models import Model

from algorithms.base_algorithm import BaseAlg


class AttentionxmlAlg(BaseAlg):
    """AttentionXML algorithm wrapper using BaseAlg utilities."""
    def __init__(
            self,  
            metrics_handler,
            taxonomy,
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
            checkpoint_manager = None,
            resume: bool = True,
            min_improvement: float = .0,
            **kwargs
            ):
        """Initialize AttentionXML model, optimizer, and runtime settings."""
        super().__init__(
            metrics_handler,
            loss_function = loss_function, # Any torch loss function (e.g. torch.nn.BCELoss() 
            method = "attentionxml", # Identifier of the model/method (e.g., "lightxml").
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
        print(f"> Initialize model...")
        self.learning_rate = learning_rate
        self.device = device
        self.model_path = model_path
        self.taxonomy = taxonomy
        yaml = YAML(typ='safe')
        model_cnf = yaml.load(Path("models/AttentionXML/configure/models/baseconfig.yaml"))
        
        self.model = Model(
            network=AttentionRNN,
            labels_num=self.taxonomy.n_nodes,
            model_path=model_path.with_name("model_state_dict_attention.pt"),
            emb_init=np.asarray(embeddings_loader.cpu(), dtype=np.float32),
            emb_size=emb_dim,
            **model_cnf['model']
        )
        # Free the source embedding tensor now that the model holds its own copy
        del embeddings_loader
        self.model.get_optimizer(lr=self.learning_rate)
        self.sigmoid = torch.nn.Sigmoid()

    def load_model(self):
        """Load a serialized model and recreate its optimizer."""
        with open(self.model_path, "rb") as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.get_optimizer(lr=self.learning_rate)
    
    def load_model_finetuned(self):
        """Load a finetuned serialized model and recreate its optimizer."""
        with open(self.model_finetuned_path, "rb") as f:
            self.model = torch.load(f, map_location=self.device)
        self.model.get_optimizer(lr=self.learning_rate)
    
    def inference_eval(self, input_data, **kwargs):
        """Run a forward pass in eval mode and return sigmoid probabilities."""
        # Set model to eval() modifies the behavior of certain layers e.g. dropout, normalization layers
        self.model.model.eval()
        scores = self.model.model(input_data)
        predictions = self.sigmoid(scores)
        return predictions

    def optimization_loop(self, input_data, labels):
        """Run one training step and return loss and gradient stats."""
        # Model in training mode, e.g. activate dropout if specified
        self.model.model.train()
        train_loss, gradients_and_lengths = self.model.train_step(input_data, labels)
        return train_loss, gradients_and_lengths

    def run_init(self):
        """No-op initialization hook for AttentionXML."""
        pass
