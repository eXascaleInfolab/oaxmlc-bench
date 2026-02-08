# OAXMLC-Bench 𓊳

This repository contains the official implementation of  
**"Benchmarking Extreme Multi-Label Classification for Semantic Annotation with Multi-Taxonomy Datasets"**

*Pietro Caforio\*, Christophe Broillet\*, Philippe Cudré-Mauroux, Julien Audiffren (University of Fribourg)*

\**Equal contribution*

---

## Overview
Extreme Multi-Label Classification (XMLC) is the task of predicting relevant labels from massive tag sets. While many large label collections are organized into taxonomies (hierarchical relationships), and recent state-of-the-art methods have shown that leveraging this structure significantly boosts performance, comprehensive evaluation of XMLC remains a challenge. This makes it difficult to properly compare state-of-the-art XMLC methods, highlight their strengths and limitations, and isolate the influence of taxonomy from other dataset characteristics.

For these reasons, we introduce **OAXMLC**, a comprehensive benchmark designed to evaluate how XMLC algorithms leverage taxonomic information across multiple tasks:
- Classification  
- Sub-category analysis  
- Completion  
- Few-shot learning  

To achieve this, we produced two new large XMLC datasets, each featuring two distinct sets of labels and taxonomies. By benchmarking a wide range of recent XMLC methods (both taxonomy-aware and taxonomy-agnostic), we analyze how tasks, datasets, and taxonomic properties impact model performance.

## Introduction
This framework allows benchmarking of the main state-of-the-art XMLC methods on classic XMLC datasets (MAG-CS, EURLex, ...) as well as multi-taxonomy datasets (OAXMLC, OAMED-XMLC), using both taxonomy-aware and taxonomy-agnostic algorithms.

Currently implemented methods are:

| Method        | Venue / Publication | Year | Algorithm Type |
|---------------|--------------------|------|----------------|
| MATCH [[1]](#1)         | WWW                | 2021 | Deep learning, taxonomy-aware (Transformer) |
| XML-CNN [[2]](#2)       | SIGIR              | 2017 | Deep learning (CNN-based) |
| AttentionXML [[3]](#3)  | NeurIPS            | 2019 | Deep learning, label-tree attention |
| FastXML [[4]](#4)       | KDD                | 2016 | Tree-based, non-deep learning |
| HECTOR [[5]](#5)        | WWW                | 2024 | Deep learning, taxonomy-aware (Seq2Seq) |
| TAMLEC [[6]](#6)       | CIKM               | 2025 | Deep learning, taxonomy-aware (parallel / path-based) |
| LightXML [[7]](#7)      | AAAI               | 2021 | Deep learning (Transformer, negative sampling) |
| CascadeXML [[8]](#8)    | NeurIPS            | 2022 | Deep learning (multi-resolution Transformer) |
| Parabel [[9]](#9)       | WWW                | 2018 | Tree-based, embedding-based |
| NGAME [[10]](#10)         | WSDM               | 2023 | Deep learning, Siamese / metric learning |
| DEXA [[11]](#11)          | KDD                | 2023 | Deep learning, Siamese with auxiliary parameters |

## Setup

### 1. Environment
Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate xmlc
pip install -r requirements.txt
```

### 2. FastXML Compilation
FastXML requires Cython compilation:

```bash
cd models/FastXML
python setup.py develop
```

### 3. Required External Dependencies

**HECTOR** and **TAMLEC** require pretrained *GloVe* embeddings. We use the `GloVe.840B.300d` version, available from the official website: https://nlp.stanford.edu/projects/glove/.  
The downloaded file must be placed in the `.vector_cache` directory at the root of the repository. The path to the pretrained embeddings (parameter `path_to_glove`) can be modified in `algorithms/hector.py` and `algorithms/tamlec.py`.

## Repository Structure

```text
exascaleinfolab-xmlc-fewshot/
├── algorithms/        # Algorithm wrappers (one file per method)
├── models/            # Original or adapted model implementations
├── datahandler/       # Datasets, taxonomies, samplers
├── configs/           # Experiment configuration files
├── misc/              # Metrics, utilities, experiment driver
├── environment.yml
├── environment_macos.yml
└── README.md
```

## Data

The two new multi-taxonomy datasets introduced by the benchmark, **OAXMLC** and **OAMED-XMLC**, can be downloaded from:
- https://zenodo.org/records/15695796  
- https://zenodo.org/records/17092264  

For each dataset, a `documents.json` file is provided, along with `concepts.zip` and `topics.zip` archives. These contain:
- `ontology.json`: label titles and descriptions  
- `taxonomy.txt`: taxonomy structure  

Files must be placed in the `datasets/` directory as follows:

```text
datasets/
├── oamedtopics/
│   ├── documents.json
│   ├── taxonomy.txt
│   └── ontology.json
├── oamedconcepts/
│   ├── documents.json
│   ├── taxonomy.txt
│   └── ontology.json
├── oaxmlc_topics/
│   ├── documents.json
│   ├── taxonomy.txt
│   └── ontology.json
└── oaxmlc_concepts/
    ├── documents.json
    ├── taxonomy.txt
    └── ontology.json
```

Additional datasets can be downloaded from:
- MAG-CS: https://drive.google.com/file/d/1P_MWGSy0JVq-nPpNQNtD6VbnF3gQfd3C/view  
- PubMed: https://drive.google.com/file/d/1I-PnPhTF81G5lXi_GhEXe2gJJ-zawKqT/view  
- EURLex: https://drive.google.com/file/d/1uBH-o_kZbKJxXhtFdWnzutuGHUGgcK9U/view  

## How To Run Experiments

The entry point for all experiments is the configuration file corresponding to the desired dataset and algorithm.

### 1. Classification Training (Standard XMLC)

```bash
python configs/{dataset}_{algorithm}.py --train --seed <seed> --device <device>
```

Example:

```bash
python configs/oamedconcepts_tamlec.py --train --seed 42 --device cuda:0
```

Results are saved in `output/{dataset}_{algorithm}{seed}/`.

### 2. Few-Shot Training

```bash
python configs/{dataset}_{algorithm}.py --train --fewshot --seed <seed> --device <device>
```

Example:

```bash
python configs/oamedconcepts_tamlec.py --train --fewshot --seed 42 --device cuda:0
```

Results are saved in `output/{dataset}_{algorithm}_fewshot{seed}/`.

### 3. Completion Experiment

After training a standard classification model, completion experiments can be run as follows:

```bash
python configs/{dataset}_{algorithm}.py --completion --seed <seed> --device <device>
```

Example:

```bash
python configs/oamedconcepts_tamlec.py --completion --seed 42 --device cuda:0
```

Completion metrics are appended to the corresponding classification output folder.

Further details on configuration files are available in `configs/readme.md`.

## Add a New Method

New methods can be added by creating a new file in the `algorithms/` directory. The recommended approach is to inherit from `algorithms/base_algorithm.py` and override the following methods:

- `__init__(self, config, ...)`
- `run_init(self)`
- `optimization_loop(self, input_data)`
- `inference_eval(self, input_data)`
- `load_model(self)`

See `algorithms/xmlcnn.py` for a minimal reference implementation.

## References

<a id="1">[1]</a>
Zhang, Y., Shen, Z., Dong, Y., Wang, K., & Han, J. (2021, April). MATCH: Metadata-aware text classification in a large hierarchy. In Proceedings of the Web Conference 2021 (pp. 3246-3257).

<a id="2">[2]</a>
Liu, J., Chang, W. C., Wu, Y., & Yang, Y. (2017, August). Deep learning for extreme multi-label text classification. In Proceedings of the 40th international ACM SIGIR conference on research and development in information retrieval (pp. 115-124).

<a id="3">[3]</a>
You, R., Zhang, Z., Wang, Z., Dai, S., Mamitsuka, H., & Zhu, S. (2019). Attentionxml: Label tree-based attention-aware deep model for high-performance extreme multi-label text classification. Advances in neural information processing systems, 32.

<a id="4">[4]</a>
Jain, H., Prabhu, Y., & Varma, M. (2016, August). Extreme multi-label loss functions for recommendation, tagging, ranking & other missing label applications. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 935-944).

<a id="5">[5]</a>
Ostapuk, N., Audiffren, J., Dolamic, L., Mermoud, A., & Cudré-Mauroux, P. (2024, May). Follow the Path: Hierarchy-Aware Extreme Multi-Label Completion for Semantic Text Tagging. In Proceedings of the ACM on Web Conference 2024 (pp. 2094-2105).

<a id="6">[6]</a>
Audiffren, J., Broillet, C., Dolamic, L., & Cudré-Mauroux, P. (2024). Extreme Multi-label Completion for Semantic Document Labelling with Taxonomy-Aware Parallel Learning. arXiv preprint arXiv:2412.13809.

<a id="7">[7]</a>
Jiang, T., Wang, D., Sun, L., Yang, H., Zhao, Z., & Zhuang, F. (2021, May). Lightxml: Transformer with dynamic negative sampling for high-performance extreme multi-label text classification. In Proceedings of the AAAI conference on artificial intelligence (Vol. 35, No. 9, pp. 7987-7994).

<a id="8">[8]</a>
Kharbanda, S., Banerjee, A., Schultheis, E., & Babbar, R. (2022). Cascadexml: Rethinking transformers for end-to-end multi-resolution training in extreme multi-label classification. Advances in neural information processing systems, 35, 2074-2087.

<a id="9">[9]</a>
Prabhu, Y., Kag, A., Harsola, S., Agrawal, R., & Varma, M. (2018, April). Parabel: Partitioned label trees for extreme classification with application to dynamic search advertising. In Proceedings of the 2018 World Wide Web Conference (pp. 993-1002).


<a id="10">[10]</a>
Dahiya, K. and Gupta, N. and Saini, D. and Soni, A. and Wang, Y. and Dave, K. and Jiao, J. and Gururaj, K. and Dey, P. and Singh, A. and Hada, D. and Jain, V. and Paliwal, B. and Mittal, A. and Mehta, S. and Ramjee, R. and Agarwal, S. and Kar, P. and Varma, M. (2023, March). NGAME: Negative mining-aware mini-batching for extreme classification. Proceedings of the Sixteenth ACM International Conference on Web Search and Data Mining (pp. 258–266) 



<a id="11">[11]</a>
Kunal Dahiya, Sachin Yadav, Sushant Sondhi, Deepak Saini, Sonu Mehta, Jian Jiao, Sumeet Agarwal, Purushottam Kar, and Manik Varma. 2023. Deep Encoders with Auxiliary Parameters for Extreme Classification. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23). Association for Computing Machinery, New York, NY, USA, 358–367