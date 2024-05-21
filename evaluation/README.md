## Evaluation Module
English | 简体中文

本模块集成了GEC任务上主流的评估工具，包括M2Scorer、ERRANT与ChERRANT。此外，还提供了转换脚本及部分数据集的ground truth。

UnifiedGEC框架在训练过程中计算得到的是Micro-level PRF，用户如果希望得到Macro-level PRF，可以使用本模块对生成结果进行整体评估。

### Usage

首先使用脚本将json格式的输出结果转化为评估工具对应的格式：
```shell
python convert.py --predict_file $PREDICT_FILE --dataset $DATASET
```
此处的dataset参数和评估工具的对应关系：

| 数据集                           | 评估工具 |
| -------------------------------- | -------- |
| CoNLL14、FCE、NLPCC18            | M2Scorer |
| AKCES-GEC、Falko-MERLIN、Cowsl2h | ERRANT   |
| MuCGEC                           | ChERRANT |

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

官方仓库：https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT

使用方法参考官方仓库：

```shell
cd cherrant/ChERRANT
python parallel_to_m2.py -f ../hyp.txt -o hyp.m2 -g char
python compare_m2_for_evaluation.py -hyp hyp.m2 -ref ref.m2
```

