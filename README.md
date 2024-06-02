# UnifiedGEC: Integrating Grammatical Error Correction Approaches for Multi-languages with a Unified Framework

[English](./README_en.md) | 简体中文

本仓库用于存放GEC工具包UnifiedGEC相关代码。



## 简介

![](./UnifiedGEC.jpg)

UnifiedGEC是一个面向GEC设计的开源框架，集成了**5个不同架构的GEC模型和7个不同语种的GEC数据集**。

我们的框架简单易用，用户可以通过一行简单的命令，调用指定的模型在数据集上进行训练。此外，通过额外的命令行参数，用户可以使用我们提供的数据增强方法来应对低资源情况，或是调用我们给出的prompt进行LLM相关的实验。

同时，我们的框架还具有非常好的扩展性，提供了dataset、dataloader、evaluator、model、trainer等模块的抽象类，允许用户自行实现相关模块。

UnifiedGEC的完整结构如下：

```
.
|-- gectoolkit  # 框架的主要代码
    |-- config  # 全局配置及Config类
    |-- data    # Dataset和Dataloader的抽象类
    |-- evaluate    # Evaluator抽象类及GEC Evaluator
    |-- llm     # prompts for LLMs
    |-- model   # Model抽象类及已经集成的模型
    |-- module  # 可复用的组件(Transformer Layer)
    |-- properties  # 模型的详细配置
    |-- trainer # Trainer的抽象类和supervised_trainer
    |-- utils   # 使用到的其他代码
    |-- quick_start.py      # run_gectoolkit.py通过调用该代码启动框架
|-- log         # 训练日志
|-- checkpoint  # 训练结果及checkpoint
|-- dataset     # 处理后的json格式数据集
|-- augmentation    # 数据增强模块
    |-- data    # error patterns方法所需的依赖项
    |-- translation_model   # back-translation方法使用的预训练模型
    |-- noise_pattern.py    # 添加噪声的数据增强方法(error patterns)
    |-- translation.py      # 翻译成其他语言再翻译回源语言的数据增强方法(back-translation)
|-- evaluation  # 评估工具及转换脚本
    |-- m2scorer    # M2Scorer, 适用于NLPCC18、CoNLL14、FCE
    |-- errant      # ERRANT, 适用于AKCES、Falko-MERLIN、Cowsl2h
    |-- cherrant    # ChERRANT, 适用于MuCGEC
    |-- convert.py  # 将生成结果转化为待评估格式的脚本
|-- run_gectoolkit.py       # 框架的启动文件
```



## UnifiedGEC

### 环境

我们的框架使用Python 3.8，请先安装allennlp 1.3.0，再安装其他依赖：

```shell
pip install allennlp==1.3.0
pip install -r requirements.txt
```

注：在conda环境下使用pip安装allennlp时，jsonnet依赖项可能报错，可以使用`conda install jsonnet`完成安装。

### 数据集

UnifiedGEC集成的数据集均是处理完成的json格式：

```json
[
    {
        "id": 0,
        "source_text": "My town is a medium size city with eighty thousand inhabitants .",
        "target_text": "My town is a medium - sized city with eighty thousand inhabitants ."
    }
]
```

我们使用的处理后数据集可以从[此处下载](https://drive.google.com/file/d/1UwQQRHW7ueadlQ3Nc8hZNKpklZJLdjaW/view?usp=sharing)。
### 模型调用

UnifiedGEC共集成了5个模型和7个不同语言的GEC数据集，各模型在数据集上测得的最好表现如下（P/R/F0.5）：

| 模型                       | CoNLL14 (EN)   | FCE (EN)       | NLPCC18 (ZH)   | MuCGEC (ZH)    | AKCES-GEC (CS) | Falko-MERLIN (DE) | COWSL2H (ES)   |
| -------------------------- | -------------- |----------------| -------------- | -------------- |----------------| ----------------- | -------------- |
| **LevenshteinTransformer** | 13.5/12.6/13.3 | 6.3/6.9/6.4    | 12.6/8.5/10.7  | 6.6/6.4/6.6    | 4.4/5.0/4.5    |                   |                |
| **GECToR**                 | 52.3/21.7/40.8 | 36.0/20.7/31.3 | 30.9/20.9/28.2 | 33.5/19.1/29.1 | 46.8/8.9/25.3  | 50.8/20.5/39.2    | 24.4/12.9/20.7 |
| **Transformer**            | 24.1/15.5/21.7 | 20.8/15.9/19.6 | 22.3/20.8/22.0 | 19.7/9.2/16.0  | 44.4/23.6/37.8 | 33.1/18.7/28.7    | 11.8/15.0/12.3 |
| **T5**                     | 36.6/39.5/37.1 | 29.2/29.4/29.3 | 32.5/21.1/29.4 | 30.2/14.4/24.8 | 52.5/40.5/49.6 | 47.4/50.0/47.9    | 53.7/39.1/49.9 |
| **SynGEC**                 | 50.6/51.8/50.9 | 59.5/52.7/58.0 | 36.0/36.8/36.2 | 22.3/26.2/23.6 | 21.9/27.6/22.8 | 32.2/33.4/32.4    | 9.3/18.8/10.3  |

使用前，请先创建存放日志和checkpoint的目录：
```shell
mkdir log
mkdir checkpoint
```

使用时，输入指令：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME
```

训练轮数、学习率等参数配置请见`./gectoolkit/config/config.json`文件，模型的详细参数请见`./gectoolkit/properties/model/`下的对应配置。

除Transformer外的其他模型需要使用到预训练模型，请下载后存储至`./gectoolkit/properties/model/`对应的模型目录下。我们提供部分预训练模型的下载地址，用户也可以前往huggingface自行下载。

UnifiedGEC也支持通过命令行修改对应参数：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --learning_rate $LR
```

### 数据增强

我们提供了两种数据增强方法（暂时只支持中英文）：

- error patterns：以增、删、改的方式向句子中添加噪声
- back-translation：将句子翻译成另一语言，再翻译回原本的语言

我们在NLPCC18和CoNLL14上做了实验，选取10%的数据来模拟低资源任务的情况（P/R/F0.5/**delta F0.5**）：

| 模型                       | 数据增强方法     | CoNLL14 (EN)     | NLPCC18 (ZH) |
| -------------------------- | ---------------- | ---------------- | ------------------- |
| **LevenshteinTransformer** | w/o augmentation |                  |                  |
|                            | error patterns   |                  |                  |
|                            | back-translation |                  |                  |
| **GECToR**                 | w/o augmentation | 13.3/20.1/14.2/- | 17.4/17.2/17.4/- |
|                            | error patterns   | 14.1/21.1/15.1/**0.9** | 20.2/18.6/19.9/**2.5** |
|                            | back-translation | 15.3/26.7/16.7/**2.5** | 20.1/17.1/19.4/**2.0** |
| **Transformer**            | w/o augmentation | 11.7/18.2/12.6/- | 11.6/5.6/9.5/- |
|                            | error patterns   | 13.4/21.6/14.5/**1.9** | 11.6/6.3/9.9/**0.4** |
|                            | back-translation | 15.4/24.2/16.6/**4.0** | 10.3/10.6/10.4/**0.9** |
| **T5**                     | w/o augmentation | 31.5/32.5/31.7/- | 31.1/16.3/26.3/- |
|                            | error patterns   | 31.5/33.8/32.0/**0.3** | 30.4/18.8/27.0/**0.7** |
|                            | back-tanslation  | 30.8/39.1/32.2/**0.5** | 24.5/22.5/24.1/**-2.2** |
| **SynGEC**                 | w/o augmentation | 48.1/46.6/47.7/- | 32.1/33.7/32.4/- |
|                            | error patterns   | 48.3/47.9/48.2/**0.5** | 34.5/36.3/34.9/**2.5** |
|                            | back-translation | 47.1/50.1/47.7/**0.0** | 33.9/37.4/34.6/**2.2** |

使用数据增强模块时，在命令行中添加`augment`参数，可选的值分为noise和translation：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --augment noise
```

初次使用时，UnifieGEC执行数据增强方法并生成相应的文件到本地，translation方法可能需要比较久的时间来生成。再次执行时，UnifiedGEC会直接使用先前生成的增强数据。

### Prompts

我们还提供了在LLM上执行GEC任务的prompt（暂时只支持中英文），分为zero-shot与few-shot版本，在NLPCC18和CoNLL14数据集上进行了测试（P/R/F0.5）：

|           | CoNLL14 (EN)   | NLPCC18 (ZH)   |
| --------- | -------------- | -------------- |
| zero-shot | 48.8/49.1/48.8 | 24.7/38.3/26.6 |
| few-shot  | 50.4/50.2/50.4 | 24.8/39.8/26.8 |

在命令行中使用`use_llm`参数，并且通过`example_num`参数指定In-context learning的样例数量：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $ DATASET_NAME --use_llm --example_num $EXAMPLE_NUM
```

此处使用的模型名称应该是huggingface的LLM名称，如`Qwen/Qwen-7B-chat`。