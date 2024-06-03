# UnifiedGEC: Integrating Grammatical Error Correction Approaches for Multi-languages with a Unified Framework

English | [简体中文](./README.md)



UnifiedGEC is an open-source, GEC-oriented framework, which integrates 5 GEC models of different architecture and 7 GEC datasets across different languages. The sturcture of our framework is shown in the picture. It provides abstract classes of dataset, dataloader, evaluator, model and trainer, allowing users to implement their own modules. This ensures excellent extensibility. 

Our framework is user-friendly, and users can train a model on a dataset with a single command. Moreover, users are able to deal with low-resource tasks with our proposed data augmentation module, or use given prompts to conduct experiments on LLMs.



![](./UnifiedGEC.jpg)



## Characterisic

- **User-friendly**: UnifiedGEC provides users with a convenient way to use our framework. They can start training or inference easily with a command line specifying the model and dataset they need to use. They can also adjust parameters, or launch data augmentation or prompt modules through a single line of command.
- **Modularized and extensible**: UnifiedGEC consists of modules including dataset, dataloader, config and so on, and provides users with abstract classses of these modules. Users are allowed to implement their own modules through these classes.
- **Comprehensive**: UnifiedGEC has integrated 3 Seq2Seq models, 2 Seq2Edit models, 2 Chinese datasets, 2 English datasets and 3 datasets of other languages. We have conducted experiments on these datasets and evaluated the performance of integerated models, which provides users with a more comprehensive understanding of GEC tasks and models.



## Architecture

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

### Models

We integrated **5** GEC models in our framework, which can be divided into two categories: **Seq2Seq** models and **Seq2Edit** models, as shown in table:

<table align="center">
    <thread>
        <tr>
            <th align="center">type</th>
            <th align="center">model</th>
            <th align="center">reference</th>
        </tr>
    </thread>
    <tbody>
        <tr>
            <td rowspan="3">Seq2Seq</td>
            <td>Transformer</td>
            <td align="center"><a href="https://arxiv.org/abs/1706.03762">(Vaswani et al., 2017)</a></td>
        </tr>
        <tr>
            <td>T5</td>
            <td align="center"><a href="https://aclanthology.org/2021.naacl-main.41/">(Xue et al., 2021)</a></td>
        </tr>
        <tr>
            <td>SynGEC</td>
            <td align="center"><a href="https://arxiv.org/abs/2210.12484">(Zhang et al., 2022)</a></td>
        </tr>
        <tr>
            <td rowspan="2">Seq2Edit</td>
            <td>Levenshtein Transformer</td>
            <td align="center"><a href="https://arxiv.org/abs/1905.11006">(Gu et al., 2019)</a></td>
        </tr>
        <tr>
            <td>GECToR</td>
            <td align="center"><a href="https://aclanthology.org/2020.bea-1.16/">(Omelianchuk et al., 2020)</a></td>
        </tr>
    </tbody>
</table>



### Datasets

We integrated **7** datasets of different languages in our framework, including Chinese, English, Spanish, Czech and German:

|   dataset    | language |                          reference                           |
| :----------: | :------: | :----------------------------------------------------------: |
|     FCE      | English  | [(Yannakoudakis et al., 2011)](https://aclanthology.org/P11-1019/) |
|   CoNLL14    | English  |   [(Ng et al., 2014)](https://aclanthology.org/W14-1701/)    |
|   NLPCC18    | Chinese  | [(Zhao et al., 2018)](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_41) |
|    MuCGEC    | Chinese  | [(Zhang et al., 2022)](https://aclanthology.org/2022.naacl-main.227/) |
|   COWSL2H    | Spanish  | [(Yamada et al., 2020)](https://ricl.aelinco.es/index.php/ricl/article/view/109) |
| Falko-MERLIN |  German  | [(Boyd et al., 2014)](http://www.lrec-conf.org/proceedings/lrec2014/pdf/606_Paper.pdf) |
|  AKCES-GEC   |  Czech   |  [(Náplava et al., 2019)](https://arxiv.org/abs/1910.00353)  |

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

### 

## Quick Start

### Installation

We use Python 3.8 in our experiments. Please install allennlp 1.3.0 first, then install other dependencies:

```shell
pip install allennlp==1.3.0
pip install -r requirements.txt
```

Note: Errors may occur while installing jsonnet with `pip`. Users are suggested to use `conda install jsonnet` to finish installation.

### Usage

Please create directories for logs and checkpoints before using our framework:

```shell
mkdir log
mkdir checkpoint
```

Users can launch our framework through command line:

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME
```

Refer to `./gectoolkit/config/config.json` for parameters related to training process, such as number of epochs, learning rate. Refer to `./gectoolkit/properties/models/` for detailed parameters of each model.

Models except Transformer require pre-trained models, so please download them and store them in the corresponding model directory under `./gectoolkit/properties/models/`. We provide download links for some of pre-trained models, and users can also download them from Huggingface.

UnifiedGEC also support adjusting parameters via command line:

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --learning_rate $LR
```

### Data Augmentation Module

We provide users with two data augmentation methods (for Chinese and English):

- error patterns: add noises randomly to sentences
- back-translation: translate sentences into the other language, and then translate back to origin language

Users can use `augment` in command line to use our data augmentation module, and `noise` and `translation` are available values:

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --augment noise
```

Upon first use, our framework will generate augmented data and save the datasets in the local file, and the back-translation method requires a certain amount of time. UnifiedGEC will use generated data directly while subsequent executions.

### Prompts

We also provide prompts for LLMs (for Chinese and English), including zero-shot prompts and few-shot prompts. 

Users can use prompts with `use_llm` in command lines，and specify the number of in-context learning examples with argument `example_num`.

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $ DATASET_NAME --use_llm --example_num $EXAMPLE_NUM
```

Model name used here should be those from huggingface, such as `Qwen/Qwen-7B-chat`.



## Experiment Results

### Models

There are 5 models and 7 datasets across different languages integrated in UnifiedGEC, and there is the best performance of implemented models on each dataset (P/R/F0.5):


| Models                     | CoNLL14 (EN)   | FCE (EN)       | NLPCC18 (ZH)   | MuCGEC (ZH)    | AKCES-GEC (CS) | Falko-MERLIN (DE) | COWSL2H (ES)   |
| -------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | ----------------- | -------------- |
| **LevenshteinTransformer** | 13.5/12.6/13.3 | 6.3/6.9/6.4    | 12.6/8.5/10.7  | 6.6/6.4/6.6    | 4.4/5.0/4.5    |                   |                |
| **GECToR**                 | 52.3/21.7/40.8 | 36.0/20.7/31.3 | 30.9/20.9/28.2 | 33.5/19.1/29.1 | 46.8/8.9/25.3  | 50.8/20.5/39.2    | 24.4/12.9/20.7 |
| **Transformer**            | 24.1/15.5/21.7 | 20.8/15.9/19.6 | 22.3/20.8/22.0 | 19.7/9.2/16.0  | 44.4/23.6/37.8 | 33.1/18.7/28.7    | 11.8/15.0/12.3 |
| **T5**                     | 36.6/39.5/37.1 | 29.2/29.4/29.3 | 32.5/21.1/29.4 | 30.2/14.4/24.8 | 52.5/40.5/49.6 | 47.4/50.0/47.9    | 53.7/39.1/49.9 |
| **SynGEC**                 | 50.6/51.8/50.9 | 59.5/52.7/58.0 | 36.0/36.8/36.2 | 22.3/26.2/23.6 | 21.9/27.6/22.8 | 32.2/33.4/32.4    | 9.3/18.8/10.3  |

### Data Augmentation

We conduct experiments on NLPCC18 and CoNLL14 datasets, and simulate low-resource cases by choosing 10% data from datasets (F0.5/**delta F0.5**):

| Models                     | Data Augmentation method | CoNLL14 (EN) | NLPCC18 (ZH)  |
| -------------------------- | ------------------------ | ------------ | ------------- |
| **LevenshteinTransformer** | w/o augmentation         | 9.5/-        | 6.0/-         |
|                            | error patterns           | 6.4/**-3.1** | 4.9/**-1.1**  |
|                            | back-translation         | 12.5/**3.0** | 5.9/**-0.1**  |
| **GECToR**                 | w/o augmentation         | 14.2/-       | 17.4/-        |
|                            | error patterns           | 15.1/**0.9** | 19.9/**2.5**  |
|                            | back-translation         | 16.7/**2.5** | 19.4/**2.0**  |
| **Transformer**            | w/o augmentation         | 12.6/-       | 9.5/-         |
|                            | error patterns           | 14.5/**1.9** | 9.9/**0.4**   |
|                            | back-translation         | 16.6/**4.0** | 10.4/**0.9**  |
| **T5**                     | w/o augmentation         | 31.7/-       | 26.3/-        |
|                            | error patterns           | 32.0/**0.3** | 27.0/**0.7**  |
|                            | back-tanslation          | 32.2/**0.5** | 24.1/**-2.2** |
| ****SynGEC**               | w/o augmentation         | 47.7/-       | 32.4/-        |
|                            | error patterns           | 48.2/**0.5** | 34.9/**2.5**  |
|                            | back-translation         | 47.7/**0.0** | 34.6/**2.2**  |

### Prompts

We use `Qwen1.5-14B-chat` and `Llama2-7B-chat` and conduct experiments on NLPCC18 and CoNLL14 datasets (P/R/F0.5):

|           | CoNLL14 (EN)   | NLPCC18 (ZH)   |
| --------- | -------------- | -------------- |
| zero-shot | 48.8/49.1/48.8 | 24.7/38.3/26.6 |
| few-shot  | 50.4/50.2/50.4 | 24.8/39.8/26.8 |



## License

UnifiedGEC uses [Apache 2.0 License](./LICENSE).

