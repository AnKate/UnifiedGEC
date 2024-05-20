import os

from gectoolkit.evaluate.cherrant.modules.annotator import Annotator
from gectoolkit.evaluate.cherrant.modules.tokenizer import Tokenizer

import argparse
from collections import Counter
from tqdm import tqdm
import torch
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")


def annotate(line):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")

    source = sent_list[0]  # sent_list 中提取第一个元素，即源文本的部分

    # if args.segmented:
    #     source = source.strip()  # source 进行去除首尾空白字符的处理
    # else:
    #     source = "".join(source.strip().split())  # source 的空白字符全部去除后合并成一个字符串

    output_str = ""  # 存储处理后的输出结果。

    for idx, target in enumerate(sent_list[1:]):
        # print("source: ", source)
        # print("target: ", target)
        # print(sentence_to_tokenized)
        source = source.replace(" ", "")
        target = target.replace(" ", "")
        source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
        out, cors = annotator(source_tokenized, target_tokenized, idx)
        if idx == 0:
            output_str += "".join(out[:-1])
        else:
            output_str += "".join(out[1:-1])
        # # pdb.set_trace()
        # try:
        #     # if args.segmented:
        #     #     target = target.strip()
        #     # else:
        #     #     target = "".join(target.strip().split())
        #     # if not args.no_simplified:
        #     #     target = cc.convert(target)
        #     source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
        #     out, cors = annotator(source_tokenized, target_tokenized, idx)
        #     if idx == 0:
        #         output_str += "".join(out[:-1])
        #     else:
        #         output_str += "".join(out[1:-1])
        # except Exception:
        #     raise Exception
    return output_str


class Args:
    def __init__(self, file, output, batch_size=128, device=0, worker_num=16, granularity="char",
                 merge=False, multi_cheapest_strategy="all", segmented=False, no_simplified=False, bpe=False):
        self.file = file
        self.output = output
        self.batch_size = batch_size
        self.device = device
        self.worker_num = worker_num
        self.granularity = granularity
        self.merge = merge
        self.multi_cheapest_strategy = multi_cheapest_strategy
        self.segmented = segmented
        self.no_simplified = no_simplified
        self.bpe = bpe


def main(args):
    # 构造分词器 tokenizer = Tokenizer('char',)   ('优秀', 'a', ['you', 'xiu']),
    # pdb.set_trace()

    tokenizer = Tokenizer(args.granularity, args.device, args.segmented, args.bpe)
    global annotator, sentence_to_tokenized
    # 构造标注器
    # pdb.set_trace()

    annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)

    # format: id src tgt1 tgt2...
    # read() 方法读取整个文件的内容为一个字符串，并 .strip() 方法去除首尾的空白字符（包括换行符）。最后，.split("\n") 方法按照换行符对字符串进行分割，将文本文件分成多行，返回一个列表 lines

    lines = open(args.file, "r", encoding="utf-8").read().strip().split("\n")  # lines: ['1 hello world', '2 hi there'
    # print('lines:', lines)
    # error_types = []

    with open(args.output, "w", encoding="utf-8") as f:
        # pdb.set_trace()

        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}

        for line in lines:
            # pdb.set_trace()

            # sent_list = line.split("\t")[1:]
            sent_list = line.split("\t")
            for idx, sent in enumerate(sent_list):
                if args.segmented:
                    # print(sent)
                    sent = sent.strip()
                else:
                    sent = "".join(sent.split()).strip()
                if idx >= 1:
                    if not args.no_simplified:
                        sentence_set.add(sent)
                    else:
                        sentence_set.add(sent)
                else:
                    sentence_set.add(sent)
        batch = []
        for sent in (sentence_set):
            count += 1
            if sent:
                batch.append(sent)
            if count % args.batch_size == 0:
                results = tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.
                batch = []
        if batch:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.

        # 单进程模式
        for line in (lines):
            # pdb.set_trace()
            ret = annotate(line)
            f.write(ret)
            f.write("\n")

            # 多进程模式：仅在Linux环境下测试，建议在linux服务器上使用
        # with Pool(args.worker_num) as pool:
        #     for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
        #         if ret:
        #             f.write(ret)
        #             f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose input file to annotate")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input parallel file")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
    parser.add_argument("-d", "--device", type=int, help="The ID of GPU", default=0)
    parser.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
    parser.add_argument("-g", "--granularity", type=str, help="Choose char-level or word-level evaluation",
                        default="char")
    parser.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion",
                        action="store_true")
    parser.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
    parser.add_argument("--segmented", help="Whether tokens have been segmented",
                        action="store_true")  # 支持提前token化，用空格隔开
    parser.add_argument("--no_simplified", help="Whether simplifying chinese",
                        action="store_true")  # 将所有corrections转换为简体中文
    parser.add_argument("--bpe", help="Whether to use bpe", action="store_true")  # 支持 bpe 切分英文单词
    args = parser.parse_args()
    parser.format_help()
    main(args)