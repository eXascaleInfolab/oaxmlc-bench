"""
For a dataset, create tokenized files in the folder {tokenizer-type}-{maxlen} folder inside the database folder
Sample usage: python -W ignore -u create_tokenized_files.py --data-dir /scratch/Workspace/data/LF-AmazonTitles-131K --tokenizer-type bert-base-uncased --max-length 32 --out_dir .
"""
import torch.multiprocessing as mp
import torch
from transformers import AutoTokenizer
from pathlib import Path
import os
import numpy as np
import time
import functools
import argparse


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])


def convert(corpus, tokenizer, max_len, num_threads, bsz=100000): 
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask

@timeit
def tokenize_dump(corpus, tokenization_dir,
                  tokenizer, max_len, prefix,
                  num_threads, batch_size=10000000):
    ind = np.zeros(shape=(len(corpus), max_len), dtype='int64')
    mask = np.zeros(shape=(len(corpus), max_len), dtype='int64')

    for i in range(0, len(corpus), batch_size):
        _ids, _mask = convert(
            corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ind[i: i + _ids.shape[0], :] = _ids
        mask[i: i + _ids.shape[0], :] = _mask

    np.save(f"{tokenization_dir}/{prefix}_input_ids.npy", ind)
    np.save(f"{tokenization_dir}/{prefix}_attention_mask.npy", mask)


def _main(args):
    data_dir = args.data_dir
    max_len = args.max_length
    out_dir = args.out_dir
    
    Y = [x.strip() for x in open(
        f'{data_dir}/lbl.raw.txt', "r", encoding="latin").readlines()]
    trnX = [x.strip() for x in open(
        f'{data_dir}/trn.raw.txt', "r", encoding="latin").readlines()]
    tstX = [x.strip() for x in open(
        f'{data_dir}/tst.raw.txt', "r", encoding="latin").readlines()]

    print(f"#labels: {len(Y)}; #train: {len(trnX)}; #test: {len(tstX)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_type, do_lower_case=True)

    tokenization_dir = f"{data_dir}/{out_dir}"
    os.makedirs(tokenization_dir, exist_ok=True)

    print(f"Dumping files in {tokenization_dir}...")
    print("Dumping for trnX...")
    tokenize_dump(
        trnX, tokenization_dir, tokenizer, max_len, "trn_doc", args.num_threads)
    print("Dumping for tstX...")
    tokenize_dump(
        tstX, tokenization_dir, tokenizer, max_len, "tst_doc", args.num_threads)
    print("Dumping for Y...")
    tokenize_dump(
        Y, tokenization_dir, tokenizer, max_len, "lbl", args.num_threads)

def main(args):
    import json
    data_dir = Path(args.data_dir)
    max_len = args.max_length
    out_dir = args.out_dir
    
    global_datasets = torch.load("global_datasets.pt")
    
    
    # The idea is to:
    # 
    # 1) load training tokenized documents in trnX
    #    build the XY_train.txt file where for each line there's the list of labels
    # 2) Use idx_to_label dictionary to, for each key, find the label text in the ontology,
    #    tokenize it and put it in lbl_input_ids.npy
    #
    # Tokenization should create 32 long arrays of numbers padded with zeros and attention_masks should
    # be arrays of ones and zeros indicating which are the non_padding tokens with ones.
    
    
    trn_input_ids = []
    trn_attention_mask = []
    with open(output_dir / "trn_X_Y.txt", "w") as x_y_f:
        for idx in global_datasets[0]['train']:
            doc_path = data_dir/f"preprocessed_data/documents/{idx}.json"
            with open(doc_path, "r") as doc_f:
                data = json.load(doc_f)
                doc_tokens, doc_labels = data[0], data[1]    
                trn_input_ids.append(
                    [ doc_tokens[i] if i<len(doc_tokens) else 0 for i in range(32) ]
                )
                trn_attention_mask.append(
                    [ 1 if i<len(doc_tokens) else 0 for i in range(32)]    
                )
                x_y_line = ""
                for lab in doc_labels:
                    if lab == 0: continue
                    x_y_line += lab + ":1.0 "
                x_y_f.write(x_y_line)
                
    
    # Load the tokenizer
    
    # Iterate over the keys of taxo.idx_to_label in increasing order and use the idx_to_title to get
    # tokenize the label descriptions and tokenize them, then save them in lbl_input_ids and lbl_attention_mask
    taxo = torch.load(data_dir/"preprocessed_data/taxonomy.pt")
    taxo_lbl_idxs = sorted(taxo.idx_to_label.keys())
    
    lbl_input_ids = []
    lbl_attention_mask = [] 
    for lbl_id in taxo_lbl_idxs:
        label = taxo.idx_to_lable[lbl_id]
        label_txt = taxo.label_to_title[label]
        label_tokenized, label_attention_mask = tokenize(label_txt)
        lbl_input_ids.append(label_tokenized)
        lbl_attention_mask.append(label_attention_mask)
    
    np.save(lbl_input_ids, path...)
    np.save(lbl_attention_mask, path...)
    
    
        
    
            
            
    
    Y = [x.strip() for x in open(
        f'{data_dir}/lbl.raw.txt', "r", encoding="latin").readlines()]
    
    trnX = [x.strip() for x in open(
        f'{data_dir}/trn.raw.txt', "r", encoding="latin").readlines()]
    
    tstX = [x.strip() for x in open(
        f'{data_dir}/tst.raw.txt', "r", encoding="latin").readlines()]

    print(f"#labels: {len(Y)}; #train: {len(trnX)}; #test: {len(tstX)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_type, do_lower_case=True)

    tokenization_dir = f"{data_dir}/{out_dir}"
    os.makedirs(tokenization_dir, exist_ok=True)

    print(f"Dumping files in {tokenization_dir}...")
    print("Dumping for trnX...")
    tokenize_dump(
        trnX, tokenization_dir, tokenizer, max_len, "trn_doc", args.num_threads)
    print("Dumping for tstX...")
    tokenize_dump(
        tstX, tokenization_dir, tokenizer, max_len, "tst_doc", args.num_threads)
    print("Dumping for Y...")
    tokenize_dump(
        Y, tokenization_dir, tokenizer, max_len, "lbl", args.num_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data directory path - with {trn,tst,lbl}.raw.txt")
    parser.add_argument(
        "--max-length",
        type=int,
        help="Max length for tokenizer",
        default=32)
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        help="Tokenizer to use",
        default="bert-base-uncased")
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Number of threads to use",
        default=24)
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Dump folder inside dataset folder",
        default="")

    args = parser.parse_args()
    main(args)
