import torchtext
#torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import GloVe
import torch


class EmbeddingHandler:
    """EmbeddingHandler Class

    This class manages embeddings, specifically pre-trained GloVe embeddings, for a given vocabulary. It can handle unknown words by initializing them with random vectors and optionally includes special token embeddings for `<pad>` and `<unk>`.

    Attributes:
        cfg (dict): Configuration dictionary
        full_embeddings (torchtext.vocab.GloVe or None): GloVe embeddings object loaded on demand.
        unk_words (int): Counter for the number of unknown words in the vocabulary.

    Methods:
        get_glove_embeddings(vocabulary, special_embs):
            Retrieves embeddings for the given vocabulary using GloVe. Handles unknown words and optionally includes special token embeddings.

            Args:
                vocabulary (list of str): List of words in the vocabulary.
                special_embs (bool): Whether to include special embeddings for `<pad>` and `<unk>` tokens.

            Returns:
                torch.tensor: A tensor of shape `(vocabulary_size, emb_dim)` containing the embeddings.
    """


    def __init__(self, cfg):
        self.cfg = cfg
        self.full_embeddings = None
        self.unk_words = 0

    def get_random_embeddings(self, vocabulary, special_embs):
        # Remove special tokens if needed
        if '<pad>' in vocabulary and not special_embs: vocabulary.remove('<pad>')
        if '<unk>' in vocabulary and not special_embs: vocabulary.remove('<unk>')
        vocab_emb = torch.normal(0.0, 0.1, size=(len(vocabulary), self.cfg['emb_dim']))

        if special_embs:
            # Embedding for <pad>, will have index 0
            embeddings = torch.zeros(1, self.cfg['emb_dim'])
            # Embedding for <unk>, will have index 1
            embeddings = torch.concat([embeddings, torch.normal(0.0, 0.1, size=(1, self.cfg['emb_dim']))], dim=0)
            embeddings = torch.concat([embeddings, vocab_emb], dim=0)
        else:
            embeddings = vocab_emb

        return embeddings


    def get_glove_embeddings(self, vocabulary, special_embs):
        if self.full_embeddings is None:
            # Count unknown words and return a random normal(0.0,0.1) vector
            def unk_init(_input):
                self.unk_words += 1
                return torch.normal(0.0, 0.1, size=(self.cfg['emb_dim'],)) 
            print(f"> Loading pre-trained GloVe embeddings...")
            # Can have a look at the possible version of GloVe here:
            # https://pytorch.org/text/stable/_modules/torchtext/vocab/vectors.html#GloVe
            # For words in vocabulary that are not in GloVe, initialize a random normal(0.0,0.1) vector
            self.full_embeddings = GloVe(name='840B', dim=self.cfg['emb_dim'], unk_init=unk_init)  
        print(f"> Lookup GloVe embeddings...")
        # Reset count
        self.unk_words = 0
        # Remove special tokens if needed
        if '<pad>' in vocabulary and not special_embs: vocabulary.remove('<pad>')
        if '<unk>' in vocabulary and not special_embs: vocabulary.remove('<unk>')
        vocab_emb = self.full_embeddings.get_vecs_by_tokens(vocabulary)    
        if special_embs:
            # Embedding for <pad>, will have index 0
            embeddings = torch.zeros(1, self.cfg['emb_dim'])
            # Embedding for <unk>, will have index 1
            embeddings = torch.concat([embeddings, torch.normal(0.0, 0.1, size=(1, self.cfg['emb_dim']))], dim=0)
            embeddings = torch.concat([embeddings, vocab_emb], dim=0)
        else:
            embeddings = vocab_emb 
        print(f">> {self.unk_words}/{self.cfg['voc_size']} ({self.unk_words/self.cfg['voc_size']*100:.1f}%) words not found in GloVe embeddings")
        # Drop the full GloVe store as soon as the trimmed matrix is built to avoid a second huge copy in RAM.
        self.full_embeddings = None
        del vocab_emb
        return embeddings
