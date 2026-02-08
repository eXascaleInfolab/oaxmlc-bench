import torch


def get_model(
        method: str = None, # It's the method name from which this model is called (match, lightxml, protonet, maml, cascadexml)
        emb_dim: int = 300, # Embedding dimension for the BaseBertModel
        device:str = 'cpu', # Device to use (cpu, cuda:0, ...)
        embeddings: torch.Tensor  = None, # Pretrained embeddings to use in the embedding layer
        ):
    
    model = XMLCNN(
        method,
        emb_dim,
        device,
        embeddings
        )
    return model


class XMLCNN(torch.nn.Module):
    def __init__(
        self, 
        method,
        emb_dim,
        device,
        embeddings
        ):
        
        super(XMLCNN, self).__init__()
        self.device = device
        self.n_features = 32
        self.n_chunks = 128
        self.hidden_dim = 512
        self.dropout = torch.nn.Dropout(0.5)

        self.convolutions = torch.nn.ModuleList([
            torch.nn.Conv1d(emb_dim, self.n_features, kernel_size=2, padding='same'),
            torch.nn.Conv1d(emb_dim, self.n_features, kernel_size=4, padding='same'),
            torch.nn.Conv1d(emb_dim, self.n_features, kernel_size=8, padding='same'),
        ])
        self.bottleneck = torch.nn.Linear(3*self.n_features*self.n_chunks, self.hidden_dim)
        # Initialize weights with the function defined below
        self.convolutions.apply(self.init_weights)
        self.bottleneck.apply(self.init_weights)

        self.pooling = torch.nn.AdaptiveMaxPool1d(self.n_chunks)
        self.relu = torch.nn.ReLU()
        # Padding index set at 0, freeze=False to enable training on embeddings
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings, padding_idx=0, freeze=False)
        # Used in taxonomy to aggregate doc and label embeddings
        self.sigmoid = torch.nn.Sigmoid()

    def init_plaincls(self, labels_num):
        self.labels_num = labels_num
        self.lastlayer = torch.nn.Linear(self.hidden_dim, self.labels_num).to(self.device)


    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        # Reshape to (N,E,S), N is the batch size, E is the embedding size, S the sequence length
        # Input shape should be (N,S,E), output shape is (N,E), N is the batch size, S the sequence length and E is the embedding size
        emb = torch.transpose(emb, 1, 2)
        # Apply the three different convolutions on the embeddings
        convolutions = [self.relu(conv(emb)) for conv in self.convolutions]
        # Apply pooling, and flatten to remove one dimension
        poolings = []
        for conv_value in convolutions:
            pooling_value = self.pooling(conv_value)
            flatten_value = torch.flatten(pooling_value, start_dim=1)
            poolings.append(flatten_value)
        # Concatenate the features into a 1D vector and apply a linear layer
        concatenation = torch.concat(poolings, dim=1)
        doc_emb = self.bottleneck(concatenation)
        doc_emb = self.relu(doc_emb)
        doc_emb = self.dropout(doc_emb)
        logits = self.lastlayer(doc_emb)
        predictions = self.sigmoid(logits)

        return predictions

    # Custom method to initialize the weights of the model
    def init_weights(self, module):
        # Skip if module has no weights
        if hasattr(module, 'weight'):
            # Weights cannot be fewer than 2D for Kaiming initialization
            if len(module.weight.shape) > 1:
                # Kaiming seems better for relu activation than Xavier
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
