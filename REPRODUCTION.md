# Reproduction

## Table of Content

- [Data](#data)
- [Embeddings](#embeddings)
- [Experiments](#experiments)

## Data

### Preprocessing

#### English WSJ

1. Clone the repository [bichngocdo/lineardpparser](https://github.com/bichngocdo/lineardpparser):
   ```shell
   git clone https://github.com/bichngocdo/lineardpparser.git
   cd lineardpparser
   ```
   Install the package `svector` according to instruction.

   Prepare data files:
   ```shell
   cp data/02-21.dep.4way data/train.dep
   cp data/022.dep.4way data/dev.dep
   cp data/23.dep.autotag data/test.dep
   ```

2. Extract forests:
    ```shell
   for dataset in train dev test; do
     cat data/$dataset.dep | python code/parser.py -w models/model.max.b8 -b8 --forest > $dataset.forest
   done
   ```

3. Extract *k*-best lists from _forests_:

   Extract lists with `k=10`:
   ```shell
   for dataset in train dev test; do
     cat $dataset.forest | python code/forest.py -k 10 > $dataset.forest.kbest10.dep
   done
   ```

   Similarly, extract lists with `k=64`.
   ```shell
   for dataset in train dev test; do
     cat $dataset.forest | python code/forest.py -k 64 > $dataset.forest.kbest10.dep
   done
   ```

4. Extract *k*-best lists using _beam_ search:
   ```shell
   for dataset in train dev test; do
     cat $dataset.dep | python code/parser.py -w models/model.max.b8 -b64 -k64 > $dataset.kbest64.dep
   done
   ```

5. Convert data to JSONL format with [convert_bracket2json.py](scripts%2Fconvert_bracket2json.py):
   ```shell
   for dataset in train dev test; do
     python ../scripts/convert_bracket2json.py data/$dataset.dep $dataset.forest.kbest10.dep $dataset.forest.kbest10.json
     python ../scripts/convert_bracket2json.py data/$dataset.dep $dataset.forest.kbest10.dep $dataset.forest.kbest64.json
     python ../scripts/convert_bracket2json.py data/$dataset.dep $dataset.forest.kbest10.dep $dataset.kbest64.json
   done
   ```

#### German SPMRL

1. Get the [SPMRL 2014 Shared Task](http://www.spmrl.org/spmrl2014-sharedtask.html) data from the organizers.

2. Train a parsing model on the original split and the predicted POS tags provided by the organizers
   using the _graph-based_ dependency parser (`anna`) in the [MATE tools](https://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/matetools/).
   The parser used in our experiments was _modified_ by the authors of MATE to produce _k_-best trees.

3. Using the modified parser, extract 50-best trees for the train, dev and test sets.

#### Czech UD

1. Get the [Czech-PDT UD Treebank](https://github.com/UniversalDependencies/UD_Czech-PDT/tree/ef906d2ab9d904b77fbbd43e91cef68f058f5e44).

2. Use [MarMoT](http://cistern.cis.lmu.de/marmot/) to create predicted POS tags (UD) by 5-way jackknifing.

3. Extract 50-best trees from the train, dev and test sets similar to [German](#german-spmrl).

### Analyze

Get the information about the top tree, oracle worst and oracle best of the _k_-best tree files similar to our report:
```shell
python scripts/analyze_kbest.py <kbest_json_file>
```


## Embeddings

### Download

#### English

Download the 50 dimension [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings trained on 6B tokens.

#### German

The embeddings for German are 100-dimension dependency-based word embeddings
trained using [`word2vecf`](https://bitbucket.org/yoavgo/word2vecf)
on the [SdeWaC](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sdewac/) corpus
with word and context cutoff frequencies of 20 (other parameters take the default values).

#### Czech

*  Download the fastText embeddings for Czech in [text format](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz) and extract the embeddings.
*  Reduce the dimension of the embeddings from 300 to 100 using PCA:
   ```shell
   python scripts/reduce_embedding_size_pca.py cc.cs.300.vec cc.cs.100.vec -n 100
   ```

### Filter Embeddings

For performance, all embeddings are filtered out words that do not present in the data.


## Experiments

All trained models contain:
* File `config.cfg` that records all parameters used to produce the model.
* Folder `log` records training and evaluation metrics, which can be viewed by `tensorboard`.
* See more information at [data](data) and [models](models).
