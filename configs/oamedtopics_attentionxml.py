import sys
import os
# Set the cwd as outside the configs folder so that the imports work properly
sys.path.insert(0, os.getcwd())
from pathlib import Path


cfg = {
    # ['eurlex', 'magcs', 'pubmed', 'oatopics', 'oaconcepts']
    'dataset_path': Path("datasets/oamedtopics"),
	'output_path': Path("output"),
	# Experiment name, better not to change
    'exp_name': Path(__file__),
    'device': 'cuda:6',
	# ['protonet', 'maml', 'match', 'xmlcnn', 'attentionxml', 'fastxml', 'hector', 'tamlec', 'siamese', 'bdc', 'lightxml', 'cascadexml', 'parabel']
    'method': 'attentionxml',
    # This is also the meta-learner learning rate in maml
    'learning_rate': 5e-5,
    # Only used in maml, this is the learning rate of the learner model
    'learner_rate': 1e-5,
    # Length of the input sequences
    'seq_length': 128,
    # (Maximum) Vocabulary size
    'voc_size': 10000,
    # How to tokenize the texts
    # ['word', 'bpe', 'unigram']
    'tokenization_mode': 'word',
    # Only used in protonet and maml, number of items per label (at most)
    'n_shot': 3,
    # Only used in protonet and maml, maximum number of items in the query set
    'n_query': 128,
    # Method to used to sample support and query sets
    # ['standard', 'min_including']
    'sampling_strategy': 'standard',
    # Only used in maml, number of optimization steps per task
    'n_optim_steps': 5,
    # k to evaluate in the metrics for the final evaluation
    'k_list': list(range(1, 21)),
    # While training only, not final evaluation
    'k_list_eval_perf': [1,2,3,5],
    # Launch a fewshot experiment
    'fewshot_exp': False,
    'selected_task': 54,
    'tamlec_params': {
        'loss_smoothing': 1e-2,
        # These parameters cannot be modified for hector and will be defaulted afterwards
        'width_adaptive': False,
        'decoder_adaptative': 0,
        'tasks_size': False,
        'freeze': False,
        'with_bias': False,
    },
    "min_freq": 10,
}
import argparse
from misc.utils.cli import config_main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config_main(parser,cfg)
    
        