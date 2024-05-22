# UnifiedGEC: Integrating Grammatical Error Correction Approaches for Multi-languages with a Unified Framework

English | [简体中文](./README.md)

This is the official repository of UnifiedGEC.

## Introduction

UnifiedGEC is an open-source, GEC-oriented framework, which integrates 5 GEC models of different architecture and 7 GEC datasets across different languages.

Our framework is user-friendly, and users can train a model on a dataset with a single command. Moreover, users are able to deal with low-resource tasks with our proposed data augmentation module, or use given prompts to conduct experiments on LLMs.

Meanwhile, our framework is extensible, as we implement abstract classes for dataset, dataloader, evaluator, model and trainer, which enables users to implement their own classes.

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

We use Python=3.8 in our experiments. Users can use conda to setup environments for UnifiedGEC:

```shell
conda create -n gec python=3.8
conda activate gec
pip install -r requirements.txt
```

Note: Errors may occur while installing jsonnet with `pip`. Users are suggested to use `conda install jsonnet` to finish installation.

### Datasets

Datasets integrated in UnifiedGEC are in JSON format:

```json
[
    {
        "id": 0,
        "source_text": "My town is a medium size city with eighty thousand inhabitants .",
        "target_text": "My town is a medium - sized city with eighty thousand inhabitants ."
    }
]
```

Our preprocessed datasets can download [here](https://drive.google.com/file/d/1UwQQRHW7ueadlQ3Nc8hZNKpklZJLdjaW/view?usp=sharing).

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

### Data Augmentation Module

We provide users with two data augmentation methods (for Chinese and English):

- error patterns: add noises randomly to sentences
- back-translation: translate sentences into the other language, and then translate back to origin language

We conduct experiments on NLPCC18 and CoNLL14 datasets, and simulate low-resource cases by choosing 10% data from datasets (P/R/F0.5/**delta F0.5**):

| Models                     | Data Augmentation method | CoNLL14 (EN)     | NLPCC18 (ZH)        |
|----------------------------|--------------------------| ---------------- | ------------------- |
| **LevenshteinTransformer** | w/o augmentation         |                  |                     |
|                            | error patterns           |                  |                     |
|                            | back-translation         |                  |                     |
| **GECToR**                 | w/o augmentation         | 13.3/20.1/14.2/- | 17.4/17.2/17.4/-   |
|                            | error patterns           | 14.1/21.1/15.1/**0.9** | 20.2/18.6/19.9/**2.5** |
|                            | back-translation         | 15.3/26.7/16.7/**2.5** | 20.1/17.1/19.4/**2.0** |
| **Transformer**            | w/o augmentation         | 11.7/18.2/12.6/- | 11.6/5.6/9.5/-     |
|                            | error patterns           | 13.4/21.6/14.5/**1.9** | 11.6/6.3/9.9/**0.4** |
|                            | back-translation         | 15.4/24.2/16.6/**4.0** | 10.3/10.6/10.4/**0.9** |
| **T5**                     | w/o augmentation         | 31.5/32.5/31.7/- | 31.1/16.3/26.3/-    |
|                            | error patterns           | 31.5/33.8/32.0/**0.3** | 30.4/18.8/27.0/**0.7** |
|                            | back-tanslation          | 30.8/39.1/32.2/**0.5** |                     |
| **SynGEC**                 | w/o augmentation         |                  |                     |
|                            | error patterns           |                  |                     |
|                            | back-translation         |                  |                     |

Users can use `augment` in command line to use our data augmentation module, and `noise` and `translation` are available values:

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --augment noise
```

Upon first use, our framework will generate augmented data and save the datasets in the local file, and the back-translation method requires a certain amount of time. UnifiedGEC will use generated data directly while subsequent executions.

### Prompts

We also provide prompts for LLMs (for Chinese and English), including zero-shot prompts and few-shot prompts. We conduct experiments on NLPCC18 and CoNLL14 datasets (P/R/F0.5):

|           | CoNLL14 (EN)   | NLPCC18 (ZH)   |
| --------- | -------------- | -------------- |
| zero-shot | 48.8/49.1/48.8 | 24.7/38.3/26.6 |
| few-shot  | 50.4/50.2/50.4 | 24.8/39.8/26.8 |

Users can use prompts with `use_llm` in command lines，and specify the number of in-context learning examples with argument `example_num`.

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $ DATASET_NAME --use_llm --example_num $EXAMPLE_NUM
```

Model name used here should be those from huggingface, such as `Qwen/Qwen-7B-chat`.