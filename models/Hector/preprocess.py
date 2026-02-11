import argparse
from collections import defaultdict
import configparser
import json
import logging
import os
import time
from tqdm import tqdm

import nltk
import numpy as np
import pandas as pd
import torch
#from gensim.models import KeyedVectors


import models.Hector.config as Cf

_NLTK_CONTENT_POS_TAGS = ["CD", "FW", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS", "RB", "RBR", "RBS", "SYM", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]


'''
Need a vocab_src (content of papers)
and vocab_trgt (for the labels)

specials=["<pad>",  "<unk>"]

default value should be set to unk
vocab_src.set_default_index(vocab_src["<unk>"])
vocab_tgt.set_default_index(vocab_tgt["<unk>"])
'''





def _process_string(s):
    """
    Filter stopwords and punctuation
    """
    tokens = nltk.tokenize.word_tokenize(s)
    tokens_w_pos_tag = nltk.pos_tag(tokens)
    content_tokens = [x[0] for x in tokens_w_pos_tag if x[1] in _NLTK_CONTENT_POS_TAGS]

    return " ".join(content_tokens)


def build_embedding_src(d_src, vocab_src):
    emb_src_size = d_src
    pad = vocab_src[Cf.padding_token]

    #assert w2v_model["the"].shape[0] == emb_src_size  # check that GloVe embedding size corresponds to emb_src_size

    emb_src_init = []
    for word in vocab_src.get_itos():
        if word == pad:
            emb = np.zeros(emb_src_size)
        else:
            emb = np.random.uniform(-1.0, 1.0, emb_src_size)
        emb_src_init.append(emb)

    emb_init = np.asarray(emb_src_init)

    return emb_init
    


def init_label_emb(w2v_model, text, emb_tgt_size):
    token_embeddings = []
    for token in text.split(" "):
        if token in w2v_model:
            token_embeddings.append(w2v_model[token])

    if len(token_embeddings) == 0:
        label_emb = np.random.uniform(-1.0, 1.0, emb_tgt_size)
    else:
        label_emb = np.mean(np.array(token_embeddings), axis=0)

    return label_emb


def build_embedding_tgt(d_tgt, vocab_tgt,abstract_dict):
    """
    abstract dict must be {label : abstract}
    """
    emb_tgt_size = d_tgt
    pad = vocab_tgt[Cf.padding_token]

    label2text = {}
    for label in abstract_dict:
        
        
        label2text[label] = _process_string(abstract_dict[label])

    emb_tgt_init = []
    for label in vocab_tgt.get_itos():
        if label == pad:
            emb = np.zeros(emb_tgt_size)
        else:
            emb = np.random.uniform(-1.0, 1.0, emb_tgt_size)
        emb_tgt_init.append(emb)

    emb_init = np.asarray(emb_tgt_init)
    return emb_init


def build_init_embedding(vocab_src ,vocab_tgt, d_src, d_tgt, abstract_dict):
    

    print("Building embedding src")
    emb_init_src = build_embedding_src(d_src, vocab_src)

    print("Building embedding tgt")
    emb_init_label = build_embedding_tgt(d_tgt, vocab_tgt,abstract_dict)

    return emb_init_src,emb_init_label

