#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""
import ast
import argparse
import fileinput
import logging
import math
import os
import sys
import time
from collections import namedtuple
from itertools import chain
import json
import numpy as np
from tqdm import tqdm
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 获取当前工作目录 /home/xiaoman/project/gectoolkit2/gectoolkit/model/SynGEC/src/src_syngec/fairseq2/fairseq_cli
# current_directory = os.getcwd()
# print('current directory:', current_directory)

fairseq_path = "/home/xiaoman/project/gectoolkit2/gectoolkit/model/SynGEC/src/src_syngec/fairseq2"
sys.path.append(fairseq_path)

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders, indexed_dataset, data_utils
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output




logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


Batch = namedtuple("Batch", "ids src_tokens src_lengths tgt_tokens constraints src_nt src_nt_lengths src_outcoming_arc_mask src_incoming_arc_mask src_dpd_matrix src_probs_matrix")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


def buffered_read(input, buffer_size):
    buffer = []
    # with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
    with open(input,'r',encoding='utf-8') as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)

    if args.constraints: # ----跳过----
        # Strip (tab-delimited) contraints, if present, from input lines,
        # store them in batch_constraints
        batch_constraints = [list() for _ in lines]
        for i, line in enumerate(lines):
            if "\t" in line:
                lines[i], *batch_constraints[i] = line.split("\t")

        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]

    # 读取句法信息，这里直接复用了写好的dataset
    src_conll_dataset = None
    src_dpd_dataset = None
    src_probs_dataset = None
    if args.conll_file:
        # 句法掩码矩阵(GAT&GCN)
        src_conll_dataset = []
        src_conll_paths = args.conll_file
        for src_conll_path in src_conll_paths:
            if indexed_dataset.dataset_exists(src_conll_path, impl="mmap"):
                src_conll_dataset.append(data_utils.load_indexed_dataset(
                    src_conll_path, None, "mmap"
                ))
            else:
                print(src_conll_path)
                raise FileNotFoundError
        if args.dpd_file:
            # 依存距离矩阵(DSA)
            src_dpd_dataset = []
            src_dpd_paths = args.dpd_file
            for src_dpd_path in src_dpd_paths:
                if indexed_dataset.dataset_exists(src_dpd_path, impl="mmap"):
                    src_dpd_dataset.append(data_utils.load_indexed_dataset(
                        src_dpd_path, None, "mmap"
                    ))
                else:
                    print(src_dpd_path)
                    raise FileNotFoundError
        if args.probs_file:
            # 句法概率矩阵(Soft GCN)
            src_probs_dataset = []
            src_probs_paths = args.probs_file
            for src_probs_path in src_probs_paths:
                if indexed_dataset.dataset_exists(src_probs_path, impl="mmap"):
                    src_probs_dataset.append(data_utils.load_indexed_dataset(
                        src_probs_path, None, "mmap"
                    ))
                else:
                    print(src_probs_path)
                    raise FileNotFoundError

    # 获取到test中 原始错误句子（每个字之间用空格分割），并且进行encode(list:2000)
    src_tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    src_lengths = [t.numel() for t in src_tokens]  # 获取每个lines的长度

    # 获取test中 目标正确句子（每个字之间用空格分割），并且进行encode
    tgt_char_path = '../../../../chinese_nlpcc18_test_with_syntax_bart/tgt.txt.char'
    # 打开文件并逐行读取内容，存储在列表中
    with open(tgt_char_path, 'r', encoding='utf-8') as file:
        tgt_lines = file.readlines()

    # 去除每行末尾的换行符，并存储在新的列表中tgt_lines
    tgt_lines = [line.strip() for line in tgt_lines]
    tgt_tokens = [
        task.target_dictionary.encode_line(
            encode_fn(tgt_str), add_if_not_exist=False
        ).long()
        for tgt_str in tgt_lines
    ]
    tgt_lengths = [t.numel() for t in tgt_tokens]  # 获取每个lines的长度


    if args.constraints: # none
        constraints_tensor = pack_constraints(batch_constraints) # none
    else:
        constraints_tensor = None

    src_nt = None
    src_nt_sizes = None

    if args.src_nt_file:      # TODO：跳过
        if indexed_dataset.dataset_exists(args.src_nt_file, impl="mmap"):
            src_nt = data_utils.load_indexed_dataset(
                args.src_nt_file, None, "mmap"
            )
            src_nt_sizes = src_nt.sizes
        else:
            print(args.src_nt_file)
            raise FileNotFoundError

    # args.task == 'syntax-enhanced-translation'
    inference_dataset = task.build_dataset_for_inference(
        src_tokens=src_tokens, src_lengths=src_lengths,
        tgt_tokens=tgt_tokens,tgt_lengths=tgt_lengths,
        constraints=constraints_tensor, src_nt=src_nt, src_nt_sizes=src_nt_sizes,
        src_conll_dataset=src_conll_dataset, src_dpd_dataset=src_dpd_dataset, src_probs_dataset=src_probs_dataset,
        syntax_type=args.syntax_type
    ) if args.task == "syntax-enhanced-translation" else task.build_dataset_for_inference(
        src_tokens, src_lengths,constraints=constraints_tensor
    )
    # print('inference_dataset:',inference_dataset[0])


    itr = task.get_batch_iterator(
        dataset = inference_dataset,
        max_tokens=args.max_tokens,     # 每个batch中最多有多少个tokens  none
        # max_sentences=args.batch_size,  # 每个batch中最多包含多少句子数  32
        max_sentences=1,
        max_positions=max_positions,    # 模型支持的最大句子长度
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test, # false
        type = 'test'
    ).next_epoch_itr(shuffle=False)


    for batch in itr:
        ids = batch["id"]
        src_tokens = batch["net_input"]["src_tokens"]
        tgt_tokens = batch['target']
        src_lengths = batch["net_input"]["src_lengths"]
        constraints = batch.get("constraints", None)
        src_outcoming_arc_mask, src_incoming_arc_mask, src_dpd_matrix, src_probs_matrix, src_nt, src_nt_lengths = None, None, None, None, None, None
        if "src_incoming_arc_mask" in batch["net_input"].keys():
            src_incoming_arc_mask = batch["net_input"]["src_incoming_arc_mask"]
        if "src_outcoming_arc_mask" in batch["net_input"].keys():
            src_outcoming_arc_mask = batch["net_input"]["src_outcoming_arc_mask"]
        if "src_dpd_matrix" in batch["net_input"].keys():
            src_dpd_matrix = batch["net_input"]["src_dpd_matrix"]
        if "src_probs_matrix" in batch["net_input"].keys():
            src_probs_matrix = batch["net_input"]["src_probs_matrix"]
        if "source_tokens_nt" in batch["net_input"].keys():
            src_nt = batch["net_input"]["source_tokens_nt"]
        if "source_tokens_nt_lengths" in batch["net_input"].keys():
            src_nt_lengths = batch["net_input"]["source_tokens_nt_lengths"]

        yield Batch(
            ids=ids,
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            tgt_tokens = tgt_tokens,
            src_nt=src_nt,
            src_nt_lengths=src_nt_lengths,
            constraints=constraints,
            src_incoming_arc_mask=src_incoming_arc_mask,
            src_outcoming_arc_mask=src_outcoming_arc_mask,
            src_dpd_matrix=src_dpd_matrix,
            src_probs_matrix=src_probs_matrix
        )
        # print('Batch:', Batch)
        # print()


def main(args):
    start_time = time.time()
    total_translate_time = 0

    # 导入syngec模型
    utils.import_user_module(args)

    args.max_source_positions = 512

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.batch_size is None:
        args.batch_size = 1

    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        not args.batch_size or args.batch_size <= args.buffer_size
    ), "--batch-size cannot be larger than --buffer-size"

    logger.info(args)
    o = open(args.output_file, "w", encoding="utf-8")

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        # arg_overrides=eval(args.model_overrides),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )


    if args.lm_path is not None: # 跳过
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({args.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Set dictionaries 中indices包含了字符和id的对应关系
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    # print(tgt_dict)

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Initialize generator
    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight} # {'lm_model': None, 'lm_weight': 0.0}
    generator = task.build_generator(models, args, extra_gen_cls_kwargs=extra_gen_cls_kwargs)
    
    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk) # None

    # max_positions = utils.resolve_max_positions(
    #     task.max_positions(), *[model.max_positions() for model in models]
    # )

    max_positions = (1024, 1024)

    if args.constraints: # 跳过
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if args.buffer_size > 1:  # Sentence buffer size: 10000
        logger.info("Sentence buffer size: %s", args.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info("Type the input sentence and press return:")
    start_id = 0


    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        count = 0
        # 对testset进行预测推理的步骤
        for batch in tqdm(make_batches(inputs, args, task, max_positions, encode_fn)):
            count  = count + 1
            if count == 4:
                break
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            tgt_tokens = batch.tgt_tokens
            constraints = batch.constraints
            src_incoming_arc_mask = batch.src_incoming_arc_mask
            src_outcoming_arc_mask = batch.src_outcoming_arc_mask 
            src_dpd_matrix = batch.src_dpd_matrix  
            src_probs_matrix = batch.src_probs_matrix
            src_nt = batch.src_nt
            src_nt_lengths = batch.src_nt_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()
                if src_incoming_arc_mask is not None:
                    for i in range(len(src_incoming_arc_mask)):
                        src_incoming_arc_mask[i] = src_incoming_arc_mask[i].cuda()
                if src_outcoming_arc_mask is not None:
                    for i in range(len(src_outcoming_arc_mask)):
                        src_outcoming_arc_mask[i] = src_outcoming_arc_mask[i].cuda()
                if src_dpd_matrix is not None:
                    for i in range(len(src_dpd_matrix)):
                        src_dpd_matrix[i] = src_dpd_matrix[i].cuda()
                if src_probs_matrix is not None:
                    for i in range(len(src_probs_matrix)):
                        src_probs_matrix[i] = src_probs_matrix[i].cuda()
                if src_nt is not None:
                    src_nt = src_nt.cuda()
                    src_nt_lengths = src_nt_lengths.cuda()

            sample = {
                "net_input": {
                    "src_tokens": src_tokens,
                    "src_lengths": src_lengths
                },
            }
            if src_incoming_arc_mask is not None:
                sample["net_input"]["src_incoming_arc_mask"] = src_incoming_arc_mask
            if src_outcoming_arc_mask is not None:
                sample["net_input"]["src_outcoming_arc_mask"] = src_outcoming_arc_mask
            if src_dpd_matrix is not None:
                sample["net_input"]["src_dpd_matrix"] = src_dpd_matrix
            if src_probs_matrix is not None:
                sample["net_input"]["src_probs_matrix"] = src_probs_matrix
            if src_nt is not None:
                sample["net_input"]["source_tokens_nt"] = src_nt
                sample["net_input"]["source_tokens_nt_lengths"] = src_nt_lengths

            # print(sample)
            translate_start_time = time.time()
            translations = task.inference_step(
                generator, models, sample, constraints=constraints
            )

            translate_time = time.time() - translate_start_time
            total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)] # [[]]
            if args.constraints: # test的时候跳过了
                list_constraints = [unpack_constraints(c) for c in constraints]

            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                constraints = list_constraints[i]
                results.append(
                    (
                        start_id + id,
                        src_tokens_i,  # 原始句子被encode后的结果
                        hypos, # 预测的结果
                        {
                            "constraints": constraints,
                            "time": translate_time / len(translations),
                        },
                    )
                )



        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                o.write("S-{}\t{}".format(id_, src_str) + "\n")
                o.write("W-{}\t{:.3f}\tseconds".format(id_, info["time"]) + "\n")
                for constraint in info["constraints"]:
                    o.write(
                        "C-{}\t{}".format(
                            id_, tgt_dict.string(constraint, args.remove_bpe)
                        )
                     + "\n")

            # Process top predictions 这个循环遍历了翻译结果列表 hypos 中的每一个翻译候选。
            for hypo in hypos[: min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe, # @@ 这是args里面定义的
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )

                detok_hypo_str = decode_fn(hypo_str) # 解码处理后的字符串形式
                print('------------------------- 预测后的结果：', detok_hypo_str)
                score = hypo["score"] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                o.write("H-{}\t{}\t{}".format(id_, score, hypo_str) + "\n")
                # detokenized hypothesis
                o.write("D-{}\t{}\t{}".format(id_, score, detok_hypo_str) + "\n")
                o.write(
                    "P-{}\t{}".format(
                        id_,
                        " ".join(
                            map(
                                lambda x: "{:.4f}".format(x),
                                # convert from base e to base 2
                                hypo["positional_scores"].div_(math.log(2)).tolist(),
                            )
                        ),
                    )
                 + "\n")
                if args.print_alignment: # 跳过了
                    alignment_str = " ".join(
                        ["{}-{}".format(src, tgt) for src, tgt in alignment]
                    )
                    o.write("A-{}\t{}".format(id_, alignment_str) + "\n")

        # update running id_ counter
        start_id += len(inputs)

    logger.info(
        "Total time: {:.3f} seconds; translation time: {:.3f}".format(
            time.time() - start_time, total_translate_time
        )
    )
    o.close()


def cli_main():
    # parser = options.get_interactive_generation_parser()
    # args = options.parse_args_and_arch(parser)

    # 获取当前工作目录 /home/xiaoman/project/gectoolkit2/gectoolkit/model/SynGEC/src/src_syngec/fairseq2/fairseq_cli
    # current_directory = os.getcwd()
    # print('current directory:', current_directory)

    # 从 JSON 文件中读取字典
    with open('../../../../args/args_syngec_generate.json', 'r') as json_file:
        args_dict = json.load(json_file)

    # 创建 Namespace 对象
    args = argparse.Namespace(**args_dict)
    print(args)

    distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()
