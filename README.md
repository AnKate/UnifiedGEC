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
  	<thead>
        <tr>
            <th align="center">type</th>
            <th align="center">model</th>
            <th align="center">reference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3" align="center">Seq2Seq</td>
            <td align="center">Transformer</td>
            <td align="center"><a href="https://arxiv.org/abs/1706.03762">(Vaswani et al., 2017)</a></td>
        </tr>
        <tr>
            <td align="center">T5</td>
            <td align="center"><a href="https://aclanthology.org/2021.naacl-main.41/">(Xue et al., 2021)</a></td>
        </tr>
        <tr>
            <td align="center">SynGEC</td>
            <td align="center"><a href="https://arxiv.org/abs/2210.12484">(Zhang et al., 2022)</a></td>
        </tr>
        <tr>
            <td rowspan="2" align="center">Seq2Edit</td>
            <td align="center">Levenshtein Transformer</td>
            <td align="center"><a href="https://arxiv.org/abs/1905.11006">(Gu et al., 2019)</a></td>
        </tr>
        <tr>
            <td align="center">GECToR</td>
            <td align="center"><a href="https://aclanthology.org/2020.bea-1.16/">(Omelianchuk et al., 2020)</a></td>
        </tr>
    </tbody>
</table>

### 数据集

我们在框架中集成了**7**个GEC数据集，包含中文、英语、西语、捷克语和德语：

<table align="center">
  	<thead>
       <tr>
            <th align="center">dataset</th>
            <th align="center">language</th>
            <th align="center">reference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">FCE</td>
            <td align="center">English</td>
            <td align="center"><a href="https://aclanthology.org/P11-1019/">(Yannakoudakis et al., 2011)</a></td>
        </tr>
        <tr>
            <td align="center">CoNLL14</td>
          	<td align="center">English</td>
            <td align="center"><a href="https://aclanthology.org/W14-1701/">(Ng et al., 2014)</a></td>
        </tr>
        <tr>
            <td align="center">NLPCC18</td>
          	<td align="center">Chinese</td>
            <td align="center"><a href="https://link.springer.com/chapter/10.1007/978-3-319-99501-4_41">(Zhao et al., 2018)</a></td>
        </tr>
        <tr>
            <td align="center">MuCGEC</td>
            <td align="center">Chinese</td>
            <td align="center"><a href="https://aclanthology.org/2022.naacl-main.227/">(Zhang et al., 2022)</a></td>
        </tr>
        <tr>
            <td align="center">COWSL2H</td>
            <td align="center">Spanish</td>
            <td align="center"><a href="https://ricl.aelinco.es/index.php/ricl/article/view/109">(Yamada et al., 2020)</a></td>
        </tr>
      	<tr>
            <td align="center">Falko-MERLIN</td>
            <td align="center">German</td>
            <td align="center"><a href="http://www.lrec-conf.org/proceedings/lrec2014/pdf/606_Paper.pdf">(Boyd et al., 2014)</a></td>
        </tr>
      	<tr>
            <td align="center">AKCES-GEC</td>
            <td align="center">Czech</td>
            <td align="center"><a href="https://arxiv.org/abs/1910.00353">(Náplava et al., 2019)</a></td>
        </tr>
    </tbody>
</table>

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

### 新增数据集

我们的框架支持用户自行添加数据集，添加的数据集文件夹`dataset_name/`需要包含**训练集、验证集和测试集**三个JSON格式的文件，并且放置于`dataset/`目录下：

```
dataset
    |-- dataset_name
        |-- trainset.json
        |-- validset.json
        |-- testset.json
```

另外，还需要在`gectoolkit/properties/dataset/`目录下新增相应的数据集配置文件，内容可以参考同目录下的其他数据集配置。

完成配置后，就可以通过上述的命令行参数运行新增的数据集。

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

### Evaluation

我们在Evaluation模块中集成了GEC任务上主流的评估工具，包括M2Scorer、ERRANT与ChERRANT。此外，还提供了转换脚本及部分数据集的ground truth。UnifiedGEC框架在训练过程中计算得到的是Micro-level PRF，用户如果希望得到Macro-level PRF，可以使用本模块对生成结果进行整体评估。

首先使用脚本将json格式的输出结果转化为评估工具对应的格式：

```shell
python convert.py --predict_file $PREDICT_FILE --dataset $DATASET
```

此处的dataset参数和评估工具的对应关系：

<table align="center">
  <thead>
    <tr>
      <th align="center">数据集</th>
      <th align="center">评估工具</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">CoNLL14、FCE、NLPCC18</td>
      <td align="center">M2Scorer</td>
    </tr>
    <tr>
      <td align="center">AKCES-GEC、Falko-MERLIN、COWSL2H</td>
      <td align="center">ERRANT</td>
    </tr>
    <tr>
      <td align="center">MuCGEC</td>
      <td align="center">ChERRANT</td>
    </tr>
  </tbody>
</table>

然后进入对应文件夹进行后续处理。

#### M2Scorer

官方仓库：https://github.com/nusnlp/m2scorer

对于英语数据集 (CoNLL14、FCE)，直接使用m2scorer进行评估即可：

```shell
cd m2scorer
m2scorer/m2scorer predict.txt m2scorer/conll14.gold
```

对于中文数据集 (NLPCC18)，**需要先使用pkunlp工具进行分词**。我们同样提供了转换脚本：

```shell
cd m2scorer
python pkunlp/convert_output.py --input_file predict.txt --output_file seg_predict.txt
m2scorer/m2scorer seg_predict.txt m2scorer/nlpcc18.gold
```

#### ERRANT

官方仓库：https://github.com/chrisjbryant/errant

使用方法参考官方仓库：

```shell
cd errant
errant_parallel -orig source.txt -cor target.txt -out ref.m2
errant_parallel -orig source.txt -cor predict.txt -out hyp.m2
errant_compare -hyp hyp.m2 -ref ref.m2
```

#### ChERRANT

官方仓库：https://github.com/HillZhang1999/MuCGEC

使用方法参考官方仓库：

```shell
cd cherrant/ChERRANT
python parallel_to_m2.py -f ../hyp.txt -o hyp.m2 -g char
python compare_m2_for_evaluation.py -hyp hyp.m2 -ref ref.m2
```





## 实验结果

### 模型表现

UnifiedGEC共集成了5个模型和7个不同语言的GEC数据集，各模型在中英文的数据集上测得的最好表现如下：

<table align="center">
  <thead>
    <tr>
      <th rowspan="3" align="center">model</th>
      <th colspan="12" align="center">dataset</th>
    </tr>
    <tr>
      <th colspan="3" align="center">CoNLL14(EN)</th>
      <th colspan="3" align="center">FCE(EN)</th>
      <th colspan="3" align="center">NLPCC18(ZH)</th>
      <th colspan="3" align="center">MuCGEC(ZH)</th>
      </tr>
    <tr>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Levenshtein Transforer</td>
      <td align="center">13.5</td>
      <td align="center">12.6</td>
      <td align="center">13.3</td>
      <td align="center">6.3</td>
      <td align="center">6.9</td>
      <td align="center">6.4</td>
      <td align="center">12.6</td>
      <td align="center">8.5</td>
      <td align="center">10.7</td>
      <td align="center">6.6</td>
      <td align="center">6.4</td>
      <td align="center">6.6</td>
    </tr>
    <tr>
      <td align="center">GECToR</td>
      <td align="center">52.3</td>
      <td align="center">21.7</td>
      <td align="center">40.8</td>
      <td align="center">36.0</td>
      <td align="center">20.7</td>
      <td align="center">31.3</td>
      <td align="center">30.9</td>
      <td align="center">20.9</td>
      <td align="center">28.2</td>
      <td align="center">33.5</td>
      <td align="center">19.1</td>
      <td align="center">29.1</td>
    </tr>
    <tr>
      <td align="center">Transformer</td>
      <td align="center">24.1</td>
      <td align="center">15.5</td>
      <td align="center">21.7</td>
      <td align="center">20.8</td>
      <td align="center">15.9</td>
      <td align="center">19.6</td>
      <td align="center">22.3</td>
      <td align="center">20.8</td>
      <td align="center">22.0</td>
      <td align="center">19.7</td>
      <td align="center">9.2</td>
      <td align="center">16.0</td>
    </tr>
    <tr>
      <td align="center">T5</td>
      <td align="center">36.6</td>
      <td align="center">39.5</td>
      <td align="center">37.1</td>
      <td align="center">29.2</td>
      <td align="center">29.4</td>
      <td align="center">29.3</td>
      <td align="center">32.5</td>
      <td align="center">21.1</td>
      <td align="center">29.4</td>
      <td align="center">30.2</td>
      <td align="center">14.4</td>
      <td align="center">24.8</td>
    </tr>
    <tr>
      <td align="center">SynGEC</td>
      <td align="center">50.6</td>
      <td align="center">51.8</td>
      <td align="center">50.9</td>
      <td align="center">59.5</td>
      <td align="center">52.7</td>
      <td align="center">58.0</td>
      <td align="center">36.0</td>
      <td align="center">36.8</td>
      <td align="center">36.2</td>
      <td align="center">22.3</td>
      <td align="center">26.2</td>
      <td align="center">23.6</td>
    </tr>
  </tbody>
</table>

各模型在小语种数据集上的最好表现如下：

<table align="center">
  <thead>
    <tr>
      <th rowspan="3" align="center">model</th>
      <th colspan="9" align="center">dataset</th>
    </tr>
    <tr>
      <th colspan="3" align="center">AKCES-GEC(CS)</th>
      <th colspan="3" align="center">Falko-MERLIN(DE)</th>
      <th colspan="3" align="center">COWSL2H</th>     
    </tr>
    <tr>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Levenshtein Transforer</td>
      <td align="center">4.4</td>
      <td align="center">5.0</td>
      <td align="center">4.5</td>
      <td align="center">2.3</td>
      <td align="center">4.2</td>
      <td align="center">2.5</td>
      <td align="center">1.9</td>
      <td align="center">2.3</td>
      <td align="center">2.0</td>
    </tr>
    <tr>
      <td align="center">GECToR</td>
      <td align="center">46.8</td>
      <td align="center">8.9</td>
      <td align="center">25.3</td>
      <td align="center">50.8</td>
      <td align="center">20.5</td>
      <td align="center">39.2</td>
      <td align="center">24.4</td>
      <td align="center">12.9</td>
      <td align="center">20.7</td>
    </tr>
    <tr>
      <td align="center">Transformer</td>
      <td align="center">44.4</td>
      <td align="center">23.6</td>
      <td align="center">37.8</td>
      <td align="center">33.1</td>
      <td align="center">18.7</td>
      <td align="center">28.7</td>
      <td align="center">11.8</td>
      <td align="center">15.0</td>
      <td align="center">12.3</td>
    </tr>
    <tr>
      <td align="center">T5</td>
      <td align="center">52.5</td>
      <td align="center">40.5</td>
      <td align="center">49.6</td>
      <td align="center">47.4</td>
      <td align="center">50.0</td>
      <td align="center">47.9</td>
      <td align="center">53.7</td>
      <td align="center">39.1</td>
      <td align="center">49.9</td>
    </tr>
    <tr>
      <td align="center">SynGEC</td>
      <td align="center">21.9</td>
      <td align="center">27.6</td>
      <td align="center">22.8</td>
      <td align="center">32.2</td>
      <td align="center">33.4</td>
      <td align="center">32.4</td>
      <td align="center">9.3</td>
      <td align="center">18.8</td>
      <td align="center">10.3</td>
    </tr>
  </tbody>
</table>

### 数据增强

我们在NLPCC18和CoNLL14上做了实验，选取10%的数据来模拟低资源任务的情况：

<table align="center">
    <thead>
        <tr>
            <th rowspan="3" align="center">model</th>
            <th rowspan="3" align="center">data augmentation methods</th>
            <th colspan="4" align="center">dataset</th>
        </tr>
      	<tr>
          	<th colspan="2" align="center">CoNLL14</th>
          	<th colspan="2" align="center">NLPCC18</th>
      	</tr>
      	<tr>
          	<th align="center">F0.5</th>
          	<th align="center">delta</th>
          	<th align="center">F0.5</th>
          	<th align="center">delta</th>
      	</tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="3" align="center">Levenshtein Transformer</td>
            <td align="center">w/o augmentation</td>
            <td align="center">9.5</td>
          	<td align="center">-</td>
          	<td align="center">6.0</td>
          	<td align="center">-</td>
        </tr>
        <tr>
            <td align="center">w/ error patterns</td>
            <td align="center">6.4</td>
          	<td align="center">-3.1</td>
          	<td align="center">4.9</td>
          	<td align="center">-1.1</td>
        </tr>
        <tr>
            <td align="center">w/ back-translation</td>
            <td align="center">12.5</td>
          	<td align="center">3.0</td>
	          <td align="center">5.9</td>
          	<td align="center">-0.1</td>
        </tr>
      	<tr>
            <td rowspan="3" align="center">GECToR</td>
            <td align="center">w/o augmentation</td>
            <td align="center">14.2</td>
          	<td align="center">-</td>
          	<td align="center">17.4</td>
          	<td align="center">-</td>
        </tr>
        <tr>
            <td align="center">w/ error patterns</td>
            <td align="center">15.1</td>
          	<td align="center">0.9</td>
          	<td align="center">19.9</td>
          	<td align="center">2.5</td>
        </tr>
        <tr>
            <td align="center">w/ back-translation</td>
            <td align="center">16.7</td>
          	<td align="center">2.5</td>
	          <td align="center">19.4</td>
          	<td align="center">2.0</td>
        </tr>
      	<tr>
            <td rowspan="3" align="center">Transformer</td>
            <td align="center">w/o augmentation</td>
            <td align="center">12.6</td>
          	<td align="center">-</td>
          	<td align="center">9.5</td>
          	<td align="center">-</td>
        </tr>
        <tr>
            <td align="center">w/ error patterns</td>
            <td align="center">14.5</td>
          	<td align="center">1.9</td>
          	<td align="center">9.9</td>
          	<td align="center">0.4</td>
        </tr>
        <tr>
            <td align="center">w/ back-translation</td>
            <td align="center">16.6</td>
          	<td align="center">4.0</td>
	          <td align="center">10.4</td>
          	<td align="center">0.9</td>
        </tr>
      	<tr>
            <td rowspan="3" align="center">T5</td>
            <td align="center">w/o augmentation</td>
            <td align="center">31.7</td>
          	<td align="center">-</td>
          	<td align="center">26.3</td>
          	<td align="center">-</td>
        </tr>
        <tr>
            <td align="center">w/ error patterns</td>
            <td align="center">32.0</td>
          	<td align="center">0.3</td>
          	<td align="center">27.0</td>
          	<td align="center">0.7</td>
        </tr>
        <tr>
            <td align="center">w/ back-translation</td>
            <td align="center">32.2</td>
          	<td align="center">0.5</td>
	          <td align="center">24.1</td>
          	<td align="center">-2.2</td>
        </tr>
      	<tr>
            <td rowspan="3" align="center">SynGEC</td>
            <td align="center">w/o augmentation</td>
            <td align="center">47.7</td>
          	<td align="center">-</td>
          	<td align="center">32.4</td>
          	<td align="center">-</td>
        </tr>
        <tr>
            <td align="center">w/ error patterns</td>
            <td align="center">48.2</td>
          	<td align="center">0.5</td>
          	<td align="center">34.9</td>
          	<td align="center">2.5</td>
        </tr>
        <tr>
            <td align="center">w/ back-translation</td>
            <td align="center">47.7</td>
          	<td align="center">0.0</td>
	          <td align="center">34.6</td>
          	<td align="center">2.2</td>
        </tr>
    </tbody>
</table>

### Prompts

我们使用`Qwen1.5-14B-chat`和`Llama2-7B-chat`，在NLPCC18和CoNLL14数据集上对提供的prompt进行了测试，得到的最好结果如下：

<table align="center">
	<thead>
    <tr>
    	<th rowspan="3" align="center">Setting</th>
      <th colspan="6" align="center">Dataset</th>
    </tr>
    <tr>
      <th colspan="3" align="center">CoNLL14</th>
      <th colspan="3" align="center">NLPCC18</th>
    </tr>
    <tr>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
      <th align="center">P</th>
      <th align="center">R</th>
      <th align="center">F0.5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">zero-shot</td>
      <td align="center">48.8</td>
      <td align="center">49.1</td>
      <td align="center">48.8</td>
      <td align="center">24.7</td>
      <td align="center">38.3</td>
      <td align="center">26.6</td>
    </tr>
    <tr>
      <td align="center">few-shot</td>
      <td align="center">50.4</td>
      <td align="center">50.2</td>
      <td align="center">50.4</td>
      <td align="center">24.8</td>
      <td align="center">39.8</td>
      <td align="center">26.8</td>
    </tr>
  </tbody>
</table>



## License

UnifiedGEC使用[Apache License](./LICENSE).

