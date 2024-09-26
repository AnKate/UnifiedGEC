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

    source = sent_list[0]

    output_str = ""

    for idx, target in enumerate(sent_list[1:]):
        source = source.replace(" ", "")
        target = target.replace(" ", "")
        source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
        out, cors = annotator(source_tokenized, target_tokenized, idx)
        if idx == 0:
            output_str += "".join(out[:-1])
        else:
            output_str += "".join(out[1:-1])

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

    tokenizer = Tokenizer(args.granularity, args.device, args.segmented, args.bpe)
    global annotator, sentence_to_tokenized

    annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)


    lines = open(args.file, "r", encoding="utf-8").read().strip().split("\n")  # lines: ['1 hello world', '2 hi there'

    with open(args.output, "w", encoding="utf-8") as f:

        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}

        for line in lines:
            sent_list = line.split("\t")
            for idx, sent in enumerate(sent_list):
                if args.segmented:
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

        for line in (lines):
            # pdb.set_trace()
            ret = annotate(line)
            f.write(ret)
            f.write("\n")

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
                        action="store_true")
    parser.add_argument("--no_simplified", help="Whether simplifying chinese",
                        action="store_true")
    parser.add_argument("--bpe", help="Whether to use bpe", action="store_true")
    args = parser.parse_args()
    parser.format_help()
    main(args)