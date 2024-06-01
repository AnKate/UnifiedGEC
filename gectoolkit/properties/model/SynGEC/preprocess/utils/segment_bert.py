import os
import sys
import tokenization
from tqdm import tqdm
from multiprocessing import Pool
import time

tokenizer = tokenization.FullTokenizer(vocab_file="../../dicts/chinese_vocab.txt", do_lower_case=True)

def split(line):
    line = line.strip()
    origin_line = line
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        return ''
    tokens = tokenizer.tokenize(line)
    return ' '.join(tokens)

# 使用 Pool 类创建一个拥有 64 个工作进程的进程池。这允许并行地处理输入数据。
with Pool(128) as pool:
    for ret in pool.imap(split, tqdm(sys.stdin), chunksize=1024):
        print(ret)