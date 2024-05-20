from typing import List, Tuple
# from modules.alignment import read_cilin, read_confusion, Alignment # 原始的
# from modules.merger import Merger  # 原始的
# from modules.classifier import Classifier # 原始的

from gectoolkit.evaluate.cherrant.modules.alignment import read_cilin, read_confusion, Alignment
from gectoolkit.evaluate.cherrant.modules.merger import Merger
from gectoolkit.evaluate.cherrant.modules.classifier import Classifier

class Annotator:
    def __init__(self,
                 align: Alignment,
                 merger: Merger,
                 classifier: Classifier,
                 granularity: str = "word",
                 strategy: str = "first"):
        self.align = align
        self.merger = merger
        self.classifier = classifier
        self.granularity = granularity
        self.strategy = strategy

    @classmethod
    def create_default(cls, granularity: str = "word", strategy: str = "first"):
        """
        Default parameters used in the paper
        """
        semantic_dict, semantic_class = read_cilin()   # 读取并获取语义词典和语义类信息
        confusion_dict = read_confusion()  # 读取并获取混淆集信息
        align = Alignment(semantic_dict, confusion_dict, granularity)
        merger = Merger(granularity)
        classifier = Classifier(granularity)
        return cls(align, merger, classifier, granularity, strategy)

    def __call__(self,
                 src: List[Tuple],   # 元组
                 tgt: List[Tuple],   # 元组
                 annotator_id: int = 0,  # 标注器的ID（可选，默认为0）
                 verbose: bool = False): # 是否打印详细信息（可选，默认为False）
        """
        Align sentences and annotate them with error type information
        将输入的源文本和目标文本进行对齐，并使用标注器（annotator）对其进行标注，添加错误类型信息。然后，将标注结果以特定格式返回。
        """
        src_tokens = [x[0] for x in src]  # 源文本的每个元组中提取标记
        tgt_tokens = [x[0] for x in tgt]  # 目标文本的每个元组中提取标记

        src_str = "".join(src_tokens)  # 连接为一个字符串
        tgt_str = "".join(tgt_tokens)  #连接为一个字符串

        # convert to text form 保存标注结果
        annotations_out = ["S " + " ".join(src_tokens) + "\n"]   # S 这 样 ， 你 就 会 尝 到 泰 国 人 死 爱 的 味 道 。

        if tgt_str == "没有错误" or src_str == tgt_str:             # Error Free Case
            annotations_out.append(f"T{annotator_id} 没有错误\n")   # T0 没有错误
            cors = [tgt_str]
            op, toks, inds = "noop", "-NONE-", (-1, -1)
            a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"  # A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0
            annotations_out.append(a_str)

        elif tgt_str == "无法标注":  # Not Annotatable Case
            annotations_out.append(f"T{annotator_id} 无法标注\n")
            cors = [tgt_str]
            op, toks, inds = "NA", "-NONE-", (-1, -1)
            a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"
            annotations_out.append(a_str)

        else:  # Other
            align_objs = self.align(src, tgt)    # 将源文本和目标文本进行对齐
            # print(align_objs)
            # 递归次数太多强制返回的情况, 直接当作没有修改
            if len(align_objs) == 0:
                annotations_out.append(f"T{annotator_id} 没有错误\n")
                cors = [tgt_str]
                op, toks, inds = "noop", "-NONE-", (-1, -1)
                a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"
                annotations_out.append(a_str)
            else:
                edit_objs = []  # 保存编辑对象
                align_idx = 0
                if self.strategy == "first":
                    align_objs = align_objs[:1]
                for align_obj in align_objs:
                    edits = self.merger(align_obj, src, tgt, verbose)
                    if edits not in edit_objs:
                        edit_objs.append(edits)
                        annotations_out.append(f"T{annotator_id}-A{align_idx} " + " ".join(tgt_tokens) + "\n")
                        align_idx += 1
                        cors = self.classifier(src, tgt, edits, verbose)
                        # annotations_out = []
                        for cor in cors:
                            op, toks, inds = cor.op, cor.toks, cor.inds
                            a_str = f"A {inds[0]} {inds[1]}|||{op}|||{toks}|||REQUIRED|||-NONE-|||{annotator_id}\n"
                            annotations_out.append(a_str)
        annotations_out.append("\n")
        return annotations_out, cors
