# Data

## English

* `en-ptb`: the Penn Treebank dataset downloaded from
  [Huang and Sagae (2020)](https://github.com/lianghuang3/lineardpparser/tree/master/data).
* `en-ptb-forest{10,64}`: _k_-best list extracted from the parsed _forests_
  using the pre-trained [model](https://github.com/lianghuang3/lineardpparser/tree/master/models)
  by Huang and Sagae (2020).

## German

The German data are originally from the SPMRL 2014 Shared Task.

* `de-spmrl-kbest50`: top 50 parse trees produced by the graph-based parser
  from the MATE tools
* `de-spmrl-kbest50-top10`: top 10 highest scored trees extracted from the top 50 trees above

## Czech

The Czech dataset is the Universal Dependencies (UD) Treebank downloaded from the [official repository](https://github.com/UniversalDependencies/UD_Czech-PDT/).
It is then automatically tagged using MarMoT with 5-way jackknifing.

* `cs-ud`: the original data with multiword expressions removed
* `cs-ud-kbest50`: similar to German.
* `cs-ud-kbest50-top10`: similar to German
