# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 10:35
# @File: utils.py

import importlib
import math
from typing import Union, Type

from gectoolkit.config.configuration import Config
from gectoolkit.data.dataset.gec_dataset import GECDataset
from gectoolkit.utils.enum_type import SupervisingMode
from gectoolkit.data.dataloader.gec_dataloader import GECDataLoader


def get_dataset_module(config: Config) \
        -> Type[Union[
            GECDataset]]:
    """
    return a dataset module according to config

    :param config: An instance object of Config, used to record parameter information.
    :return: dataset module
    """
    try:
        return eval('Dataset{}'.format(config['model']))
    except:
        pass

    return GECDataset
    # task_type = config['language'].lower()
    # if task_type == DatasetLanguage.en:
    #     return EnglishDataset
    # elif task_type == DatasetLanguage.zh:
    #     return ChineseDataset
    # else:
    #     return AbstractDataset


def get_dataloader_module(config: Config) \
        -> Type[Union[
            GECDataLoader]]:
    """Create dataloader according to config

        Args:
            config (gectoolkit.config.configuration.Config): An instance object of Config, used to record parameter information.

        Returns:
            Dataloader module
        """
    try:
        return eval('DataLoader{}'.format(config['model']))
    except:
        pass

    return GECDataLoader

def get_trainer(config):
    r"""Automatically select trainer class based on task type and model name

    Args:
        config (~gectoolkit.config.configuration.Config)

    Returns:
        ~gectoolkit.trainer.SupervisedTrainer: trainer class
    """
    model_name = config["model"]
    sup_mode = config["supervising_mode"]
    if sup_mode == SupervisingMode.fully_supervised:
        if config['embedding']:
            try:
                return getattr(
                    importlib.import_module('gectoolkit.trainer.supervised_trainer'),
                    'Pretrain' + model_name + 'Trainer'
                )
            except:
                if model_name.lower() in ['mathen']:
                    return getattr(
                        importlib.import_module('gectoolkit.trainer.supervised_trainer'),
                        'PretrainSeq2SeqTrainer'
                    )
                else:
                    pass
        try:
            return getattr(
                importlib.import_module('gectoolkit.trainer.supervised_trainer'),
                model_name + 'Trainer'
            )
        except AttributeError:
            return getattr(
                importlib.import_module('gectoolkit.trainer.supervised_trainer'),
                'SupervisedTrainer'
            )

    elif sup_mode in SupervisingMode.weakly_supervised:
        try:
            return getattr(
                importlib.import_module('gectoolkit.trainer.weakly_supervised_trainer'),
                model_name + 'WeakTrainer'
            )
        except AttributeError:
            return getattr(
                importlib.import_module('gectoolkit.trainer.weakly_supervised_trainer'),
                'WeaklySupervisedTrainer'
            )
    else:
        return getattr(
            importlib.import_module('gectoolkit.trainer.abstract_trainer'),
            'AbstractTrainer'
        )


def time_since(s):
    """compute time

    Args:
        s (float): the amount of time in seconds.

    Returns:
        (str) : formatting time.
    """
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)
