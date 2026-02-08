# Adapted from https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py
import torch


class MetaLearner(torch.nn.Module):
    def __init__(self, learner, cfg):
        super(MetaLearner, self).__init__()
        self.cfg = cfg
        assert self.cfg['model_name'] != 'metalearner', f"> {self.cfg['model_name']} cannot be used as model!"
        self.learner = learner


    def initialize_fc_layer(self, out_dim):
        self.fc_layer = torch.nn.Linear(self.cfg['emb_dim'], out_dim).to(self.cfg['device'])
        self.fc_layer.apply(self.init_weights)


    # Custom method to initialize the weights of the model
    def init_weights(self, module):
        # Skip if module has no weights
        if hasattr(module, 'weight'):
            # Weights cannot be fewer than 2D for Kaiming initialization
                if len(module.weight.shape) > 1:
                    torch.nn.init.xavier_normal_(module.weight, gain=1)


    def forward(self, input_data):
        # Get the document embeddings
        outputs = self.learner(input_data)
        doc_embedding = outputs['document_embedding']
        # Pass it the through the linear layer
        predictions = self.fc_layer(doc_embedding)
        # Add a sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        return predictions
