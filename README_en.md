# UnifiedGEC: Integrating Grammatical Error Correction Approaches for Multi-languages with a Unified Framework

English | [简体中文](./README.md)

This is the official repository of UnifiedGEC.

## Introduction


Complete structure of UnifiedGEC:

```
.
|-- gectoolkit  # main code of UnifiedGEC
    |-- config  # internal config and implementation of Config Class
    |-- data    # Abstract Class of Dataset and Dataloader, and implementation of GEC Dataloader
    |-- evaluate    # Abstract Class of Evaluator and implementation of GEC Evaluator
    |-- llm     # prompts for LLMs
    |-- model   # Abstract Class of Model and code of integrated models
    |-- module  # reusable modules (e.g., Transformer Layer)
    |-- properties  # detailed external config of each model
    |-- trainer # Abstract Class of Trainer and implementation of SupervisedTrainer
    |-- utils   # other tools used in our framework
    |-- quick_start.py      # code for launching the framework
|-- log         # logs of training process
|-- checkpoint  # results and checkpoints of training process
|-- dataset     # preprocessed datasets in JSON format
|-- augmentation    # data augmentation module
    |-- data    # dependencies of error patterns
    |-- noise_pattern.py    # code of error patterns
    |-- translation.py      # code of back-translation
|-- evaluation  # evaluation module
    |-- m2scorer    # M2Scorer, for NLPCC18、CoNLL14、FCE
    |-- errant      # ERRANT, for AKCES、Falko-MERLIN、Cowsl2h
    |-- cherrant    # ChERRANT, for MuCGEC
    |-- convert.py  # script for convert output JSON file into corresponding format
|-- run_gectoolkit.py       # code for launching the framework
```



## UnifiedGEC

### Installation

### Datasets

### Usage

There are 5 models and 7 datasets across different languages integrated in UnifiedGEC, and there is the best performance of implemented models on each dataset (P/R/F0.5):


| Models                     | CoNLL14 (EN)   | FCE (EN)       | NLPCC18 (ZH)   | MuCGEC (ZH)    | AKCES-GEC (CS) | Falko-MERLIN (DE) | COWSL2H (ES)   |
|----------------------------| -------------- | -------------- | -------------- | -------------- | -------------- | ----------------- | -------------- |
| **LevenshteinTransformer** | 13.5/12.6/13.3 |                | 12.6/8.5/10.7  | 6.6/6.4/6.6    |                |                   |                |
| **GECToR**                 | 52.3/21.7/40.8 | 36.0/20.7/31.3 | 30.9/20.9/28.2 | 33.5/19.1/29.1 | 46.8/8.9/25.3  | 50.8/20.5/39.2    | 24.4/12.9/20.7 |
| **Transformer**            | 24.1/15.5/21.7 | 20.8/15.9/19.6 | 22.3/20.8/22.0 | 19.7/9.2/16.0  | 44.4/23.6/37.8 | 33.1/18.7/28.7    | 11.8/15.0/12.3 |
| **T5**                     | 36.6/39.5/37.1 | 29.2/29.4/29.3 | 32.5/21.1/29.4 | 30.2/14.4/24.8 | 52.5/40.5/49.6 | 47.4/50.0/47.9    | 53.7/39.1/49.9 |
| **SynGEC**                 |                |                |                |                |                |                   |                |

Users can launch our framework through command line:

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME
```

Refer to `./gectoolkit/config/config.json` for parameters related to training process, such as number of epochs, learning rate. Refer to `./gectoolkit/properties/models/` for detailed parameters of each model.

UnifiedGEC also support adjusting parameters via command line:

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --learning_rate $LR
```