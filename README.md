# Neural Reranking for Dependency Parsing

This repository contains code for the neural rerankers
(`RCNN`, `RCNN-shared` and `GCN`) from the paper
[Neural Reranking for Dependency Parsing: An Evaluation](https://aclanthology.org/2020.acl-main.379/)
published at ACL 2020.

## Usage

### Requirements
* Python 3.6
* Install dependencies in [requirements.txt](requirements.txt)

### Packages
* [reranker_rcnn](rcnnrr%2Freranker_rcnn): RCNN reranker
* [reranker_rcnn_shared](rcnnrr%2Freranker_rcnn_shared): RCNN-shared reranker
* [reranker_gcn](rcnnrr%2Freranker_gcn): GCN reranker

### Training
Run:
```shell
PYTHONPATH=`pwd` python rcnnrr/<package>/experiment.py train --help
```
to see possible arguments.

For example, to train a GCN reranker on the sample dataset, run:
```shell
PYTHONPATH=`pwd` python rcnnrr/reranker_gcn/experiment.py train \
  --train_file data/sample/train.jsonl \
  --dev_file data/sample/dev.jsonl \
  --test_file data/sample/test.jsonl \
  --model_dir runs/sample-gcn \
  --word_dim 5 \
  --tag_dim 3 \
  --use_characters True \
  --char_dim 4 \
  --char_hidden_dim 7 \
  --lstm_hidden_dim 7 \
  --gcn_hidden_dim 6 \
  --num_lstms 2 \
  --num_gcns 2 \
  --interval 1 \
  --max_epoch 1 \
  --learning_rate 0.001 \
  --train_batch_size 1 \
  --loss ranking_max
```

### Evaluation
Run:
```shell
PYTHONPATH=`pwd` python rcnnrr/<package>/experiment.py eval --help
```
to see possible arguments.

For example, to evaluate a trained GCN reranker on the sample dataset, run:

```shell
PYTHONPATH=`pwd` python rcnnrr/reranker_gcn/experiment.py eval \
  --test_file data/sample/test.jsonl \
  --output_file runs/sample-gcn/result.conll \
  --model_dir runs/sample-gcn \
  --type conllx
```

### Tensorboard

```shell
tensorboard --logdir runs/sample-gcn/log
```

### Scripts

* [reduce_embedding_size_pca.py](scripts%2Freduce_embedding_size_pca.py):
  reduce embedding dimension using PCA.
  The script is a modification of [Raunak et al. (2019)](https://github.com/vyraun/Half-Size).
* [convert_bracket2conll.py](scripts%2Fconvert_bracket2conll.py):
  convert parsed trees in bracket format (output by Liang and Sagae (2010)'s [parser](https://github.com/lianghuang3/lineardpparser))
  to CoNLL format
* [convert_jsonl2conll.py](scripts%2Fconvert_jsonl2conll.py):
  convert parsed trees in JSONL format (output by the rerankers)
  to CoNLL format
* [convert_bracket2json.py](scripts%2Fconvert_bracket2json.py):
  convert *k*-best trees in bracket format (output by Liang and Sagae (2010)'s [parser](https://github.com/lianghuang3/lineardpparser))
  to JSONL format
* [convert_nbest2json.py](scripts%2Fconvert_nbest2json.py):
  convert *k*-best trees output by the graph-based parser from the MATE tools to JSONL format
* [analyze_kbest.py](scripts%2Fanalyze_kbest.py):
  analyze UAS and LAS from the *k*-best trees
* [mixture_reranker.py](scripts%2Fmixture_reranker.py):
  combine results based on the base parser and the reranker scores
* [parsing_eval_jsonl.py](scripts%2Fparsing_eval_jsonl.py):
  return UAS and LAS on parsing results in JSONL format

## Reproduction

See [reproduction](REPRODUCTION.md).

## Citation

```bib
@inproceedings{do-rehbein-2020-neural,
    title = "Neural Reranking for Dependency Parsing: An Evaluation",
    author = "Do, Bich-Ngoc and Rehbein, Ines",
    editor = "Jurafsky, Dan and Chai, Joyce and Schluter, Natalie and Tetreault, Joel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.379",
    doi = "10.18653/v1/2020.acl-main.379",
    pages = "4123--4133",
}
```