# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/27 10:20
# @File: quick_start.py
import logging
import os
import sys
from logging import getLogger

from gectoolkit.config.configuration import Config
from gectoolkit.data.utils import get_evaluator_module
from gectoolkit.data.utils import get_dataset_module, get_dataloader_module
from gectoolkit.utils.file_reader import get_model, init_seed
from gectoolkit.utils.utils import get_trainer
from gectoolkit.utils.logger import init_logger
from gectoolkit.llm.run_prompts import run_prompts

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))


def train_with_train_valid_test_split(temp_config):
    """
    Train GEC models with evaluation
    """
    if temp_config['training_resume'] or temp_config['resume']:
        config = Config.load_from_pretrained(temp_config['checkpoint_dir'])
    else:
        config = temp_config
    config._update(temp_config.internal_config_dict)
    device = config["device"]

    logger = getLogger()
    logger.info(config)

    dataset = get_dataset_module(config)
    dataset._load_dataset()
    dataloader = get_dataloader_module(config)(config, dataset)
    model = get_model(config["model"])(config, dataloader.pretrained_tokenizer)
    if device:
        model = model.cuda(device)

    evaluator = get_evaluator_module(config)(config, dataloader.pretrained_tokenizer)
    trainer = get_trainer(config)(config, model, dataloader, evaluator)

    if temp_config['training_resume'] or temp_config['resume']:
        trainer._load_checkpoint()

    logger.info(model)
    trainer.fit()


def test_with_train_valid_test_split(temp_config):
    """
    Evaluate existing GEC models
    """
    print(temp_config['checkpoint_dir'])
    config = Config.load_from_pretrained(temp_config['checkpoint_dir'])
    print(config)
    device = config["device"]

    logger = getLogger()
    logger.info(config)

    dataset = get_dataset_module(config)
    dataset._load_dataset()
    dataloader = get_dataloader_module(config)(config, dataset)
    model = get_model(config["model"])(config, dataloader.pretrained_tokenizer)
    if device:
        model = model.cuda(device)

    evaluator = get_evaluator_module(config)(config, dataloader.pretrained_tokenizer)
    trainer = get_trainer(config)(config, model, dataloader, evaluator)
    trainer._load_checkpoint()

    trainer.test()


def run_toolkit(model_name, dataset_name, augment_method, use_llm=False, example_num=0, config_dict=None):
    """
    Run GEC toolkit
    """
    if use_llm:
        run_prompts(model_name, dataset_name, example_num)
    else:
        if config_dict is None:
            config_dict = {}
        config = Config(model_name=model_name, dataset_name=dataset_name,
                        augment_method=augment_method, config_dict=config_dict)

        init_seed(config['random_seed'], True)

        init_logger(config)

        if config['test_only']:
            test_with_train_valid_test_split(config)
        else:
            train_with_train_valid_test_split(config)
