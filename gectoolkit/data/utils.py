# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 10:35
# @File: utils.py

from typing import Union, Type

from gectoolkit.config.configuration import Config

from gectoolkit.data.dataset.gec_dataset import GECDataset
from gectoolkit.data.dataset.syngec_dataset import SyngecDataset
from gectoolkit.data.dataloader.gec_dataloader import GECDataLoader
from gectoolkit.data.dataloader.syngec_dataloader import SyngecDataLoader
from gectoolkit.evaluate.gec_evaluator import GECEvaluator


def get_dataset_module(config: Config):
    """
    return a dataset module according to config

    :param config: An instance object of Config, used to record parameter information.

    :return: dataset module
    """
    if config["model"] == "SynGEC":
        return SyngecDataset(config)
    else:
        return GECDataset(config)


def get_dataloader_module(config: Config):
    """
    return a dataloader according to config

    :param config: An instance object of Config, used to record parameter information.

    :return: Dataloader module
    """
    if config["model"] == 'SynGEC':
        return SyngecDataLoader
    else:
        return GECDataLoader


def get_evaluator_module(config: Config):
    """
    return a evaluator module according to config

    :param config: An instance object of Config, used to record parameter information.
    :return: evaluator module
    """

    return GECEvaluator



