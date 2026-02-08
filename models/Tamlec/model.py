
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.Tamlec.config as Cf

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, decoders_custom : list, src_embed, tgt_embed, generators : list, adapter):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoders_custom = nn.ModuleList(decoders_custom)
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generators = nn.ModuleList(generators)
        self.adapter = adapter

    def forward(self, src, tgt, src_mask, tgt_mask, child_mask, task_id):
        "Take in and process masked src and target sequences."
        out =  self.encode(src, src_mask)  #src_encoded : bs x max_padding_src x emb_src_dim
        out = self.adapter(out)  # src_transformed : bs x max_padding_src x emb_tgt_dim
        out = self.decode(out, src_mask, tgt, tgt_mask, task_id)  # tgt_decoded: bs x emb_tgt_dim
        generator = self.generators[task_id]
        out = generator(out,tgt_mask=tgt_mask, child_mask=child_mask)   # y_pred : bs x len(label_vocab)
        return out

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, task_id):
        tgt = self.tgt_embed(tgt)  # bs x #candidates x emb_tgt_dim
        tgt_decoded = self.decoder(tgt, memory, src_mask, tgt_mask)
        if len(self.decoders_custom)>0 :
            tgt_decoded = self.decoders_custom[task_id](tgt_decoded, memory, src_mask, tgt_mask)  # bs x #candidates x emb_tgt_dim
        tgt_decoded = tgt_decoded.masked_fill(torch.transpose(tgt_mask, 1, 2) == 0, 0.)  # bs x #candidate x emb_tgt_dim
        return tgt_decoded

   
    def to_cuda(self):
        self.cuda()
     

    def freeze(self):
        if 0 & Cf.QUICK_DEBUG :
            print(" Pre Freeze: listing non-frozen weights")
            print("====="*20)
            for name, param in self.named_parameters():
                if param.requires_grad == True : print(name)


        self.encoder.freeze()
        self.decoder.freeze()
        self.src_embed[0].freeze()
        self.tgt_embed.freeze()
        self.adapter.freeze()

        

        if 0 & Cf.QUICK_DEBUG :
            print("====="*20)
            print("====="*20)
            print("====="*20)
            print("POst freeze : listing non-frozen weights")
            for name, param in self.named_parameters():
                if param.requires_grad == True : print(name)
      

    def get_generator_weights(self,task_id : int):
        generator : Generator= self.generators[task_id]
        weight  = generator.get_embedding()
        return weight 

    def get_custom_decoder_weights(self,task_id : int):
        decoder : Decoder = self.decoders_custom[task_id]
        weights  = decoder.get_weight()
        return weights

    
        
        


class Generator(nn.Module):
    "Define standard linear transformation + softmax to convert the decoder output to next-token probabilities."
    def __init__(self, d_model, vocab_size, with_bias : bool = False ):
        super(Generator, self).__init__()
        self.with_bias = with_bias
        self.proj = nn.Linear(d_model, vocab_size, bias=with_bias)
        

    def forward(self, x, tgt_mask,child_mask=None):
        x = x*torch.transpose(tgt_mask,1,2)

        x = torch.sum(x, dim=1)

        y = self.proj(x)  # bs x max_padding_tgt x 1

        

        if child_mask is not None : 
            y = y.masked_fill(child_mask == 0, -1e10)
            
        
        y = F.softmax(y, dim=-1)

        
        


        return y.squeeze(-1)

    def get_embedding(self):
        w = self.proj.weight
        w = w.cpu().detach().numpy()
        if not self.with_bias :
            return w
        b = self.proj.bias
        b = b.cpu().detach().numpy()
        return w, b

    def freeze(self):
        self.proj.weight.requires_grad = False
        if self.with_bias  : self.proj.bias.requires_grad = False


class Adapter(nn.Module):
    """
    Define standard linear transformation, from source dim to target dim
    """
    def __init__(self, d_src, d_tgt):
        super(Adapter, self).__init__()
        self.proj = nn.Linear(d_src, d_tgt)

    def forward(self, x):
        y = self.proj(x)
        return y

    def freeze(self):
        self.proj.weight.requires_grad = False
        self.proj.bias.requires_grad = False


def clones(module, N):
    "Produce N identical layers (for encoder/decoder)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    def freeze(self):
        for layer in self.layers:
            layer.freeze()
        self.norm.freeze()
    
        


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # for residual connections
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections"
        x = self.sublayer[0](x, lambda x: self.self_attn(query=x, key=x, value=x, mask=mask))
        return self.sublayer[1](x, self.feed_forward)
    
    def freeze(self):
        self.self_attn.freeze()
        self.feed_forward.freeze()
        for layer in self.sublayer : 
            layer.freeze()
            


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
    def freeze(self):
        for layer in self.layers:
            layer.freeze()
        self.norm.freeze()

    def get_weight(self):
        weights={}
        for layer_number in range(len(self.layers)):
            layer :DecoderLayer= self.layers[layer_number]
            layer_weight = layer.get_weights()
            weights["layer {}".format(layer_number)] =layer_weight
        
        weights["layer norm"] =self.norm.get_weights()
        return weights
        


            
        


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)  # for residual connections

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(query=x, key=x, value=x, mask=tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(query=x, key=m, value=m, mask=src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    def freeze(self):
        self.self_attn.freeze()
        self.src_attn.freeze()
        self.feed_forward.freeze()
        for layer in self.sublayer : 
            layer.freeze()

    def get_weights(self):


        weights = {
            'self_attn' : self.self_attn.get_weights(),
            'src_attn' : self.src_attn.get_weights(),
            'feed_forward' : self.feed_forward.get_weights(),
        }

        for layer_number in range(len(self.sublayer)):
            layer :SublayerConnection= self.sublayer[layer_number]
            layer_weight = layer.get_weights()
            weights["sublayer {}".format(layer_number)] =layer_weight

        return weights        
        


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
    def freeze(self):
        self.norm.freeze()

    def get_weights(self):
        return self.norm.get_weights()


class LayerNorm(nn.Module):
    "Same as torch.nn.LayerNorm"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    def freeze(self):
        self.a_2.requires_grad = False
        self.b_2.requires_grad = False

    def get_weights(self):
        return {
            "a_2" : self.a_2.cpu().detach().numpy(),
            "b_2" : self.b_2.cpu().detach().numpy()
        }


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
    
    def freeze(self):
        for layer in self.linears :
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

    def get_weights(self):

        weights = {}

        for layer_number in range(len(self.linears)):
            layer :nn.Linear= self.linears[layer_number]
            layer_weight = {
            "w" : layer.weight.cpu().detach().numpy(),
            "b" : layer.bias.cpu().detach().numpy(),
        }
            weights["sublayer {}".format(layer_number)] =layer_weight
        
        return weights


class PositionwiseFeedForward(nn.Module):
    "Implements FFN (Equation 2) for encoder/decoder layers."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
    def freeze(self):
        for layer in [self.w_1,self.w_2] :
            layer.weight.requires_grad = False
            layer.bias.requires_grad = False

    def get_weights(self):

        weights = {}

        for layer_number in range(2):
            layer :nn.Linear= [self.w_1,self.w_2][layer_number]
            layer_weight = {
            "w" : layer.weight.cpu().detach().numpy(),
            "b" : layer.bias.cpu().detach().numpy(),
        }
            weights["sublayer {}".format(layer_number)] =layer_weight
        
        return weights


class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx, emb_init=None, need_training=True):
        super(Embeddings, self).__init__()

        if emb_init is not None:
            emb_weight = torch.from_numpy(emb_init).float()
            self.lut = nn.Embedding.from_pretrained(
                emb_weight,
                padding_idx=padding_idx,
                sparse=False #True
            )
        else:
            self.lut = nn.Embedding(
                vocab_size,
                embedding_dim,
                padding_idx=padding_idx,
                sparse= False #True
            )

        self.lut.weight.requires_grad = need_training
        self.d_model = embedding_dim

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    def get_embedding(self):
        w = self.lut.weight
        w = w.cpu().detach().numpy()
        return w
    
    def freeze(self):
        self.lut.weight.requires_grad = False


class PositionalEncoding(nn.Module):
    """
    Encode information about the position of the tokens in the sequence.
    Sine and cosine functions of different frequencies.
    See section 3.5.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N_src, N_tgt, d_src, d_tgt, d_ff, h, dropout, emb_src_init=None, emb_tgt_init=None, lvl_mask=None, n_tasks=1, N_custom_target = 0, with_bias = False):
    c = copy.deepcopy
    encoder_attn = MultiHeadedAttention(h, d_src)
    encoder_ff = PositionwiseFeedForward(d_src, d_ff, dropout)
    decoder_attn = MultiHeadedAttention(h, d_tgt)
    decoder_ff = PositionwiseFeedForward(d_tgt, d_ff, dropout)
    position = PositionalEncoding(d_src, dropout)

    encoder = Encoder(
        layer=EncoderLayer(
            size=d_src,
            self_attn=c(encoder_attn),
            feed_forward=c(encoder_ff),
            dropout=dropout
        ),
        N=N_src
    )
    decoder = Decoder(
        layer=DecoderLayer(
            size=d_tgt,
            self_attn=c(decoder_attn),
            src_attn=c(decoder_attn),
            feed_forward=c(decoder_ff),
            dropout=dropout
        ),
        N=N_tgt-N_custom_target
    )

    if N_custom_target>0 : 
        decoders_custom = [ 
            Decoder(
            DecoderLayer(
            size=d_tgt,
            self_attn=c(decoder_attn),
            src_attn=c(decoder_attn),
            feed_forward=c(decoder_ff),
            dropout=dropout
            ),
        N=N_custom_target) for _ in range(n_tasks)
        ]
    else :
        decoders_custom = []

    src_embed = Embeddings(
        vocab_size=len(src_vocab),
        embedding_dim=d_src,
        padding_idx=src_vocab[Cf.padding_token],
        emb_init=emb_src_init,
        need_training=True
    )
        
    tgt_embed = Embeddings(
        vocab_size=len(tgt_vocab),
        embedding_dim=d_tgt,
        padding_idx=tgt_vocab[Cf.padding_token],
        emb_init=emb_tgt_init,
        need_training=True
    )

    generators =  [ Generator(
        d_model=d_tgt,
        vocab_size=len(tgt_vocab),
        with_bias=with_bias
    )
    for _ in range(n_tasks)
    ]

    adapter = Adapter(d_src, d_tgt)
            
    model = EncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        decoders_custom=decoders_custom,
        src_embed=nn.Sequential(src_embed, c(position)),
        tgt_embed=tgt_embed,
        generators=generators,
        adapter=adapter) 
        
        

    print("Initializing weights")
    for name, param in model.named_parameters():
        if name == "src_embed.0.lut.weight" and emb_src_init is not None:
            print(f"skipped {name}")
        elif name == "tgt_embed.lut.weight" and emb_tgt_init is not None:
            print(f"skipped {name}")
        elif param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return model


