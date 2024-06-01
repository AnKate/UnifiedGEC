#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import random
import sys
import json
import pdb

import numpy as np
import torch

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# 添加 fairseq 模块所在的路径
fairseq_path = "/home/xiaoman/project/gectoolkit2/gectoolkit/model/SynGEC/src/src_syngec/fairseq2"
sys.path.append(fairseq_path)

from fairseq import (
    checkpoint_utils,
    distributed_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")


def main(args):
    utils.import_user_module(args)  # 导入用户模型的路径 args.user_dir

    # 提示用户必须使用其中一个参数来指定批处理大小
    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # 调用了faireseq.logging.mertrics中的 reset 方法，该方法用于重置一些记录指标（metrics）的内部状态，以便开始一个新的训练轮次或评估阶段。
    metrics.reset()

    np.random.seed(args.seed)
    utils.set_torch_seed(args.seed)

    # 在主进程中执行验证检查点目录的函数
    if distributed_utils.is_master(args):  #  检查分布式训练中当前进程是否为主进程。
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    logger.info(args)

    # 设置了一个任务task Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid data (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(","): # valid
        task.load_dataset(valid_sub_split, combine=False, epoch=1)  # 导入数据

    # Build model and criterion
    model = task.build_model(args)         # 导入模型
    criterion = task.build_criterion(args)  # LabelSmoothedCrossEntropyCriterion()

    logger.info(model)
    logger.info("task: {} ({})".format(args.task, task.__class__.__name__))
    logger.info("model: {} ({})".format(args.arch, model.__class__.__name__))
    logger.info(
        "criterion: {} ({})".format(args.criterion, criterion.__class__.__name__)
    )
    logger.info(
        "num. model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    # (optionally) Configure quantization
    if args.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=args.quantization_config_path,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if args.model_parallel_size == 1:
        trainer = Trainer(args, task, model, criterion, quantizer)   # 给trainer（fairseq -- trainer）
    else:
        trainer = MegatronTrainer(args, task, model, criterion)

    logger.info(
        "training on {} devices (GPUs/TPUs)".format(args.distributed_world_size)
    )
    logger.info(
        "max tokens per GPU = {} and max sentences per GPU = {}".format(
            args.max_tokens, args.batch_size
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        args,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"), # 如果任务（task）具有分片数据（sharded data）用于训练，则禁用迭代器缓存。
    )
    print('extra_state:',extra_state)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf   # 最大轮数 60
    lr = trainer.get_lr()                    # 获取当前的学习率

    train_meter = meters.StopwatchMeter()  # 记录了开始的时间
    train_meter.start()

    while lr > args.min_lr and epoch_itr.next_epoch_idx <= max_epoch:
        # train for one epoch
        valid_losses, should_stop = train(args, trainer, task, epoch_itr)
        # print('valid_losses:',valid_losses)
        # print('should_stop:',should_stop)

        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )

    train_meter.stop() # 记录结束时间
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))


def should_stop_early(args, valid_loss):
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if args.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    args.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    # 从epoch_itr中获取下一个epoch的数据迭代器。
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,         # 每个GPU上固定batch的大小
        shuffle=(epoch_itr.next_epoch_idx > args.curriculum), # 如果当前epoch的索引大于args.curriculum，则进行shuffle。
    )

    # 确定在训练过程中更新参数的频率。
    # update_freq = (
    #     #  如果当前epoch小于等于args.update_freq的长度，使用args.update_freq中的对应epoch的值。
    #     args.update_freq[epoch_itr.epoch - 1]
    #     if epoch_itr.epoch <= len(args.update_freq)
    #     else args.update_freq[-1]
    # )

    # 12.26修改不需要梯度累计
    update_freq = 1

    # 使用GroupedIterator将数据迭代器进行分组，以便按照update_freq的频率更新参数。
    itr = iterators.GroupedIterator(itr, update_freq)

    # 如果命令行参数中设置了args.tpu，则将数据迭代器转换为TPU数据加载器。（跳过）
    if getattr(args, "tpu", False):
        itr = utils.tpu_data_loader(itr)

    # 初始化一个进度条，用于在终端显示训练进度。
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,       # 使用命令行参数中指定的日志格式。
        log_interval=args.log_interval,   # 设置日志输出的时间间隔
        epoch=epoch_itr.epoch,            # 设置当前epoch的编号。
        tensorboard_logdir=(
            args.tensorboard_logdir if distributed_utils.is_master(args) else None
        ),
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )


    trainer.begin_epoch(epoch_itr.epoch)          # 通知训练器开始新的epoch
    valid_losses = [None]                         # 初始化一个列表，用于存储验证损失
    valid_subsets = args.valid_subset.split(",")  # 将命令行参数中的验证子集字符串分割为列表。
    should_stop = False                       # 初始化一个标志，用于指示是否应该停止训练。
    num_updates = trainer.get_num_updates()   # 获取当前已执行的更新步数。
    for i, samples in enumerate(progress):    # 迭代训练数据，samples是一个batch的样本。
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i      # 记录每个训练步骤的运行时间。
        ):
            log_output = trainer.train_step(samples)   # 执行一步训练，并获取训练输出。
            print('----train.py中的log_output:',log_output)
            print()
        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()    # 获取当前已执行的更新步数。
            if num_updates % args.log_interval == 0:   # 如果已执行的更新步数达到指定的日志间隔。
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")   # 重置"train_inner"度量，以便下一个日志间隔重新收集统计信息。

        # 检查是否到达epoch的末尾。
        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(  # 执行验证并保存检查点。
            args, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))  # 获取训练统计信息。
    progress.print(stats, tag="train", step=num_updates)  # 在进度条上打印统计信息。

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def validate_and_save(args, trainer, task, epoch_itr, valid_subsets, end_of_epoch):
    num_updates = trainer.get_num_updates()
    max_update = args.max_update or math.inf
    do_save = (
        (end_of_epoch and epoch_itr.epoch % args.save_interval == 0)
        or num_updates >= max_update
        or (
            args.save_interval_updates > 0
            and num_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates >= args.validate_after_updates
        )
    )
    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % args.validate_interval == 0)
        or num_updates >= max_update
        or (
            args.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % args.validate_interval_updates == 0
        )
    ) and not args.disable_validation

    # Validate
    valid_losses = [None]
    if do_validate: # do_valiate是一个bool变量
        valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

    # Stopping conditions
    should_stop = (
        should_stop_early(args, valid_losses[0])
        or num_updates >= max_update
        or (
            args.stop_time_hours > 0
            and trainer.cumulative_training_time() / (60 * 60) > args.stop_time_hours
        )
    )

    # Save checkpoint
    if do_save or should_stop:
        logger.info("begin save checkpoint")
        checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

    return valid_losses, should_stop


def get_training_stats(stats):
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(shuffle=False)
        if getattr(args, "tpu", False):
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args.tensorboard_logdir if distributed_utils.is_master(args) else None
            ),
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(args, trainer, stats):
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best, stats[args.best_checkpoint_metric]
        )
    return stats

def cli_main(modify_parser=None):

    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    #################################################################
    # # 获取当前工作目录
    # current_directory = os.getcwd()
    # print('current directory:',current_directory)    # 打印当前工作目录
    # logger.info(f"fairseq_cli的当前工作目录: {current_directory}")
    #
    # # # 将字典写入 JSON 文件
    # # with open('args/args_syngec_bart.json', 'w') as json_file:
    # #     json.dump(args_dict, json_file, indent=4)
    # #     logger.info(f"Arguments saved to 'args/args_syngec_bart.json'")
    # # 当前位置："/home/xiaoman/project/gectoolkit2/gectoolkit/model/SynGEC/src/src_syngec/fairseq-0.10.2/fairseq_cli"
    # # 从 JSON 文件中读取字典
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # # 目标目录
    # target_dir = '/home/xiaoman/project/gectoolkit2'
    # # 更改当前工作目录
    # os.chdir(target_dir)
    # print('fairseq_cli/train.py的当前目录:', os.getcwd())
    # with open('/home/xiaoman/project/gectoolkit2/gectoolkit/properties/model/Syngec/args/Chinese/Chinese_nlpcc18_translation_bart.json', 'r') as json_file:
    #     args_dict = json.load(json_file)
    #
    # # 创建 Namespace 对象
    # args = argparse.Namespace(**args_dict)
    # print(args)
    ###########################################################################################

    if args.profile:
        print('args.profile:',args.profile)
        with torch.cuda.profiler.profile():  # 使用PyTorch的CUDA性能分析器进行GPU性能分析的上下文管理器。这会记录CUDA操作的时间和内存使用情况。
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(args, main)
    else:
        distributed_utils.call_main(args, main)


if __name__ == "__main__":
    cli_main()