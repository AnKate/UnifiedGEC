# UnifiedGEC

本仓库用于存放GEC工具包UnifiedGEC相关代码。

## Repository Structure
```
.
|-- gectoolkit  # 框架的主要代码
    |-- config  # 全局配置及Config类
    |-- data    # Dataset和Dataloader的抽象类
    |-- evaluate    # Evaluator抽象类及GEC Evaluator
    |-- llm     # prompts for LLMs
    |-- model   # Model抽象类及已经集成的模型
    |-- module  # 可复用的组件
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


## 目前已整合的内容

### 模型
#### Seq2Edit
- LevenshteinTransformer
- GECToR

#### Seq2Seq
- Transformer
- SynGEC
- T5

### 数据集
#### Chinese
- NLPCC18 
- MuCGEC

#### English
- FCE
- CoNLL14

#### Low-resource
- AKCES-GEC (Czech)
- Falko-MERLIN (German)
- COWSL2H (Spanish)


### 数据增强
- error patterns
- back-translation (T5)


