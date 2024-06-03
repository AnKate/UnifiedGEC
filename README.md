# UnifiedGEC: Integrating Grammatical Error Correction Approaches for Multi-languages with a Unified Framework

[English](./README_en.md) | 简体中文



UnifiedGEC是一个面向GEC设计的开源框架，集成了**5个不同架构的GEC模型和7个不同语种的GEC数据集**。我们的框架结构如图所示，提供了dataset、dataloader、evaluator、model、trainer等模块的抽象类，允许用户自行实现相关模块，具有非常好的扩展性。

用户可以通过一行简单的命令，调用指定的模型在数据集上进行训练。此外，通过额外的命令行参数，用户可以使用我们提供的数据增强方法来应对低资源情况，或是调用我们给出的prompt进行LLM相关的实验。

![](./UnifiedGEC.jpg)



## 框架特点

- **易于使用**：UnifiedGEC为用户提供了非常方便的调用方式，只需要在命令行中输入指令，指定使用的模型、数据集，即可迅速便捷地开始训练或推理。调整参数、使用数据增强或prompt模块，也只需要一行指令即可启动。
- **模块化设计，扩展性强**：UnifiedGEC包含了Dataset、Dataloader、Config等多个模块，并提供了相应的抽象类。用户可以通过继承相应的类，轻松地实现自己的方法。
- **集成内容多，综合性强**：UnifiedGEC集成了3个Seq2Seq模型、2个Seq2Edit模型、2个中文数据集、2个英文数据集及3个小语种数据集，并且在上述的数据集上，对5个模型的性能进行了评估，为用户提供了有关GEC任务和模型更为全面的认识。



## 框架结构

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

### 模型

我们在框架中集成了**5**个GEC模型，按照架构可以分成**Seq2Seq**和**Seq2Edit**两类，如下表所示：

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

### 数据集

我们在框架中集成了**7**个GEC数据集，包含中文、英语、西语、捷克语和德语：

|   dataset    | language |                          reference                           |
| :----------: | :------: | :----------------------------------------------------------: |
|     FCE      | English  | [(Yannakoudakis et al., 2011)](https://aclanthology.org/P11-1019/) |
|   CoNLL14    | English  |   [(Ng et al., 2014)](https://aclanthology.org/W14-1701/)    |
|   NLPCC18    | Chinese  | [(Zhao et al., 2018)](https://link.springer.com/chapter/10.1007/978-3-319-99501-4_41) |
|    MuCGEC    | Chinese  | [(Zhang et al., 2022)](https://aclanthology.org/2022.naacl-main.227/) |
|   COWSL2H    | Spanish  | [(Yamada et al., 2020)](https://ricl.aelinco.es/index.php/ricl/article/view/109) |
| Falko-MERLIN |  German  | [(Boyd et al., 2014)](http://www.lrec-conf.org/proceedings/lrec2014/pdf/606_Paper.pdf) |
|  AKCES-GEC   |  Czech   |  [(Náplava et al., 2019)](https://arxiv.org/abs/1910.00353)  |

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



## Quick Start

### 环境

我们的框架使用Python 3.8，请先安装allennlp 1.3.0，再安装其他依赖：

```shell
pip install allennlp==1.3.0
pip install -r requirements.txt
```

注：在conda环境下使用pip安装allennlp时，jsonnet依赖项可能报错，可以使用`conda install jsonnet`完成安装。

### 模型调用

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

使用数据增强模块时，在命令行中添加`augment`参数，可选的值分为noise和translation：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --augment noise
```

初次使用时，UnifieGEC执行数据增强方法并生成相应的文件到本地，translation方法可能需要比较久的时间来生成。再次执行时，UnifiedGEC会直接使用先前生成的增强数据。

### Prompts

我们提供了中英文的prompt，包括zero-shot和few-shot两种设置。

调用prompt时，需要在命令行中使用`use_llm`参数，并且通过`example_num`参数指定In-context learning的样例数量：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $ DATASET_NAME --use_llm --example_num $EXAMPLE_NUM
```

此处使用的模型名称应该是huggingface的LLM名称，如`Qwen/Qwen-7B-chat`。



## 实验结果

### 模型表现

UnifiedGEC共集成了5个模型和7个不同语言的GEC数据集，各模型在数据集上测得的最好表现如下（P/R/F0.5）：

| 模型                       | CoNLL14 (EN)   | FCE (EN)       | NLPCC18 (ZH)   | MuCGEC (ZH)    | AKCES-GEC (CS) | Falko-MERLIN (DE) | COWSL2H (ES)   |
| -------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | ----------------- | -------------- |
| **LevenshteinTransformer** | 13.5/12.6/13.3 | 6.3/6.9/6.4    | 12.6/8.5/10.7  | 6.6/6.4/6.6    | 4.4/5.0/4.5    |                   |                |
| **GECToR**                 | 52.3/21.7/40.8 | 36.0/20.7/31.3 | 30.9/20.9/28.2 | 33.5/19.1/29.1 | 46.8/8.9/25.3  | 50.8/20.5/39.2    | 24.4/12.9/20.7 |
| **Transformer**            | 24.1/15.5/21.7 | 20.8/15.9/19.6 | 22.3/20.8/22.0 | 19.7/9.2/16.0  | 44.4/23.6/37.8 | 33.1/18.7/28.7    | 11.8/15.0/12.3 |
| **T5**                     | 36.6/39.5/37.1 | 29.2/29.4/29.3 | 32.5/21.1/29.4 | 30.2/14.4/24.8 | 52.5/40.5/49.6 | 47.4/50.0/47.9    | 53.7/39.1/49.9 |
| **SynGEC**                 | 50.6/51.8/50.9 | 59.5/52.7/58.0 | 36.0/36.8/36.2 | 22.3/26.2/23.6 | 21.9/27.6/22.8 | 32.2/33.4/32.4    | 9.3/18.8/10.3  |

### 数据增强

我们在NLPCC18和CoNLL14上做了实验，选取10%的数据来模拟低资源任务的情况（F0.5/**delta F0.5**）：

| 模型                       | 数据增强方法     | CoNLL14 (EN) | NLPCC18 (ZH)  |
| -------------------------- | ---------------- | ------------ | ------------- |
| **LevenshteinTransformer** | w/o augmentation | 9.5/-        | 6.0/-         |
|                            | error patterns   | 6.4/**-3.1** | 4.9/**-1.1**  |
|                            | back-translation | 12.5/**3.0** | 5.9/**-0.1**  |
| **GECToR**                 | w/o augmentation | 14.2/-       | 17.4/-        |
|                            | error patterns   | 15.1/**0.9** | 19.9/**2.5**  |
|                            | back-translation | 16.7/**2.5** | 19.4/**2.0**  |
| **Transformer**            | w/o augmentation | 12.6/-       | 9.5/-         |
|                            | error patterns   | 14.5/**1.9** | 9.9/**0.4**   |
|                            | back-translation | 16.6/**4.0** | 10.4/**0.9**  |
| **T5**                     | w/o augmentation | 31.7/-       | 26.3/-        |
|                            | error patterns   | 32.0/**0.3** | 27.0/**0.7**  |
|                            | back-tanslation  | 32.2/**0.5** | 24.1/**-2.2** |
| **SynGEC**                 | w/o augmentation | 47.7/-       | 32.4/-        |
|                            | error patterns   | 48.2/**0.5** | 34.9/**2.5**  |
|                            | back-translation | 47.7/**0.0** | 34.6/**2.2**  |

### Prompts

我们使用`Qwen1.5-14B-chat`和`Llama2-7B-chat`，在NLPCC18和CoNLL14数据集上对提供的prompt进行了测试（P/R/F0.5），得到的最好结果如下：

|           | CoNLL14 (EN)   | NLPCC18 (ZH)   |
| --------- | -------------- | -------------- |
| zero-shot | 48.8/49.1/48.8 | 24.7/38.3/26.6 |
| few-shot  | 50.4/50.2/50.4 | 24.8/39.8/26.8 |



## License

UnifiedGEC使用[Apache License](./LICENSE).

