# UnifiedGEC: Integrating Grammatical Error Correction Approaches for Multi-languages with a Unified Framework

English | 简体中文

本仓库用于存放GEC工具包UnifiedGEC相关代码。



## 简介





## 框架结构
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
|-- evaluation  # 对生成结果对整体评估方法
    |-- m2scorer    # M2Scorer, 适用于NLPCC18、CoNLL14、FCE
    |-- errant      # ERRANT, 适用于AKCES、Falko-MERLIN、Cowsl2h
    |-- cherrant    # ChERRANT, 适用于MuCGEC
    |-- convert.py  # 将生成结果转化为待评估格式的脚本
|-- run_gectoolkit.py       # 框架的启动文件
```



## UnifiedGEC

### 环境

### 数据集

### 模型调用

UnifiedGEC共集成了5个模型和7个不同语言的GEC数据集，各模型在数据集上测得的最好表现如下（P/R/F0.5）：

| 模型                       | CoNLL14 (EN)   | FCE (EN)       | NLPCC18 (ZH)   | MuCGEC (ZH)    | AKCES-GEC (CS) | Falko-MERLIN (DE) | COWSL2H (ES)   |
| -------------------------- | -------------- | -------------- | -------------- | -------------- | -------------- | ----------------- | -------------- |
| **LevenshteinTransformer** | 13.5/12.6/13.3 |                | 12.6/8.5/10.7  | 6.6/6.4/6.6    |                |                   |                |
| **GECToR**                 | 52.3/21.7/40.8 | 36.0/20.7/31.3 | 30.9/20.9/28.2 | 33.5/19.1/29.1 | 46.8/8.9/25.3  | 50.8/20.5/39.2    | 24.4/12.9/20.7 |
| **Transformer**            | 24.1/15.5/21.7 | 20.8/15.9/19.6 | 22.3/20.8/22.0 | 19.7/9.2/16.0  | 44.4/23.6/37.8 | 33.1/18.7/28.7    | 11.8/15.0/12.3 |
| **T5**                     | 36.6/39.5/37.1 | 29.2/29.4/29.3 | 32.5/21.1/29.4 | 30.2/14.4/24.8 | 52.5/40.5/49.6 | 47.4/50.0/47.9    | 53.7/39.1/49.9 |
| **SynGEC**                 |                |                |                |                |                |                   |                |

使用时，输入指令：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME
```

训练轮数、学习率等参数配置请见`./gectoolkit/config/config.json`文件，模型的详细参数请见`./gectoolkit/properties/models/`下的对应配置。

UnifiedGEC也支持通过命令行修改对应参数：

```shell
python run_gectoolkit.py -m $MODEL_NAME -d $DATASET_NAME --learning_rate $LR
```

### 数据增强

### Prompts



