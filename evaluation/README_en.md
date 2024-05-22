## Evaluation Module
English | [简体中文](./README.md)

This module integrates mainstream evaluation tools for GEC tasks, including M2Scorer, ERRANT and ChERRANT. Additionally, we also provide scripts for converting and ground truth of some datasets.

During the process of training, UnifiedGEC calculate micro-level PRF for the results of models, so if users want to evaluate models in a macro way, they can use this evaluation module.

### Usage

First, users should use our provided script to convert outputs of the models to the format required by scorers:

```shell
python convert.py --predict_file $PREDICT_FILE --dataset $DATASET
```

Correspondence between datasets and scorers:

| Dataset                        | scorer   |
|--------------------------------|----------|
| CoNLL14、FCE、NLPCC18            | M2Scorer |
| AKCES-GEC、Falko-MERLIN、Cowsl2h | ERRANT   |
| MuCGEC                         | ChERRANT |

#### M2Scorer

Official repository: https://github.com/nusnlp/m2scorer

For English datasets (CoNLL14、FCE)，use M2scorer directly for evaluation：

```shell
cd m2scorer
m2scorer/m2scorer predict.txt m2scorer/conll14.gold
```

For Chinese datasets (NLPCC18)，**pkunlp tools for segmentation is required**. We also provide converting scripts:

```shell
cd m2scorer
python pkunlp/convert_output.py --input_file predict.txt --output_file seg_predict.txt
m2scorer/m2scorer seg_predict.txt m2scorer/nlpcc18.gold
```

#### ERRANT

Official repository: https://github.com/chrisjbryant/errant

Usage is referenced from official repository:

```shell
cd errant
errant_parallel -orig source.txt -cor target.txt -out ref.m2
errant_parallel -orig source.txt -cor predict.txt -out hyp.m2
errant_compare -hyp hyp.m2 -ref ref.m2
```

#### ChERRANT

Official repository: https://github.com/HillZhang1999/MuCGEC

Usage is referenced from official repository:

```shell
cd cherrant/ChERRANT
python parallel_to_m2.py -f ../hyp.txt -o hyp.m2 -g char
python compare_m2_for_evaluation.py -hyp hyp.m2 -ref ref.m2
```





