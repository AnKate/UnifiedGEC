# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 11:31
# @File: abstract_dataset.py


import random
import os
import copy
import re

from gectoolkit.utils.file_reader import read_json_data, write_json_data
from augmentation.noise_pattern import noise_pattern
from augmentation.translation import translation


class GECDataset(object):
    """Chinese dataset

    the base class of dataset class
    """

    def __init__(self, config):
        super().__init__()
        self.model = config["model"]
        self.dataset = config["dataset"]
        self.augment_method = config["augment"]

        self.dataset_path = config['dataset_dir'] if config['dataset_dir'] else config["dataset_path"]
        self.language_name = config["language"]

        self.validset_divide = config["validset_divide"]
        self.shuffle = config["shuffle"]

        self.device = config["device"]

        self.resume_training = config['resume_training'] if config['resume_training'] else config['resume']

        self.from_pretrained = False
        self.datas = []

        self.trainset = []
        self.validset = []
        self.testset = []
        self.validset_id = []
        self.trainset_id = []
        self.testset_id = []
        self.folds = []
        self.folds_id = []


        self._load_dataset()

    def _load_all_data(self):
        trainset_file = os.path.join(self.dataset_path, 'trainset.json')
        validset_file = os.path.join(self.dataset_path, 'validset.json')
        testset_file = os.path.join(self.dataset_path, 'testset.json')

        if os.path.isabs(trainset_file):
            trainset = read_json_data(trainset_file)
        else:
            trainset = read_json_data(os.path.join(os.getcwd(), trainset_file))
        if os.path.isabs(validset_file):
            validset = read_json_data(validset_file)
        else:
            validset = read_json_data(os.path.join(os.getcwd(), validset_file))
        if os.path.isabs(testset_file):
            testset = read_json_data(testset_file)
        else:
            testset = read_json_data(os.path.join(os.getcwd(), testset_file))

        return trainset + validset + testset

    def _load_dataset(self):
        """
        read dataset from files
        """
        if self.trainset_id and self.testset_id:
            self._init_split_from_id()
        else:
            trainset_file = os.path.join(self.dataset_path, 'trainset.json')
            validset_file = os.path.join(self.dataset_path, 'validset.json')
            testset_file = os.path.join(self.dataset_path, 'testset.json')
            if os.path.isabs(trainset_file):
                self.trainset = read_json_data(trainset_file)
            else:
                self.trainset = read_json_data(os.path.join(os.getcwd(), trainset_file))
            if os.path.isabs(validset_file):
                self.validset = read_json_data(validset_file)  # [:32]
            else:
                self.validset = read_json_data(os.path.join(os.getcwd(), validset_file))
            if os.path.isabs(testset_file):
                self.testset = read_json_data(testset_file)
            else:
                self.testset = read_json_data(os.path.join(os.getcwd(), testset_file))

            if self.validset_divide is not True:
                self.testset = self.validset + self.testset
                self.validset = []

            if self.augment_method != 'none':
                if self.language_name not in ["en", "zh"]:
                    raise SyntaxError("Invalid language for data augmentation.")
                else:
                    augmented_trainset_file = os.path.join(self.dataset_path,
                                                           'trainset_' + self.augment_method + '.json')
                    if os.path.exists(augmented_trainset_file):
                        self.trainset = read_json_data(augmented_trainset_file)
                    else:
                        if self.augment_method == 'noise':
                            self.trainset = noise_pattern(self.trainset, self.language_name)
                            write_json_data(self.trainset, augmented_trainset_file)
                        elif self.augment_method == 'translation':
                            self.trainset = translation(self.trainset, self.language_name)
                            write_json_data(self.trainset, augmented_trainset_file)
                        else:
                            raise SyntaxError("Invalid method for data augmentation.")

    def dataset_load(self):
        """"
        dataset process and build vocab.

        when running k-fold setting, this function required to call once per fold.
        """
        parameters = self._build_vocab()
        if not self.resume_training and not self.from_pretrained:
            for key, value in parameters.items():
                setattr(self, key, value)
        if self.resume_training:
            self.resume_training = False

        if self.shuffle:
            random.shuffle(self.trainset)

    def parameters_to_dict(self):
        """
        return the parameters of dataset as format of dict.
        """
        parameters_dict = {}
        for name, value in vars(self).items():
            if hasattr(eval('self.{}'.format(name)), '__call__') or re.match('__.*?__', name):
                continue
            else:
                parameters_dict[name] = copy.deepcopy(value)
        return parameters_dict

    def _preprocess(self):
        raise NotImplementedError

    def _build_vocab(self):
        raise NotImplementedError

    def _update_vocab(self, vocab_list):
        raise NotImplementedError

    def save_dataset(self, trained_dir):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        raise NotImplementedError
