# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2022/1/27 10:35
# @File: configuration.py

import copy
import sys
import os
import re
import json
import warnings
from logging import getLogger
from enum import Enum

import torch

from gectoolkit.utils.file_reader import read_json_data, write_json_data
from gectoolkit.utils.enum_type import known_args


class Config(object):
    """The class for loading pre-defined parameters.

    Config will load the parameters from internal config file, dataset config file, model config file, config dictionary and cmd line.

    The default road path of internal config file is 'gectoolkit/config/config.json', and it's not supported to change.

    The dataset config, model config and config dictionary are called the external config.

    According to specific dataset and model, this class will load the dataset config from default road path 'gectoolkit/properties/dataset/dataset_name.json'
    and model config from default road path 'gectoolkit/properties/model/model_name.json'.

    You can set the parameters 'model_config_path' and 'dataset_config_path' to load your own model and dataset config, but note that only json file can be loaded correctly.
    Config dictionary is a dict-like object. When you initialize the Config object, you can pass config dictionary through the code 'config = Config(config_dict=config_dict)'

    Cmd line requires you keep the template --param_name=param_value to set any parameter you want.

    If there are multiple values of the same parameter, the priority order is as following:

    cmd line > external config > internal config

    in external config, config dictionary > model config > dataset config.

    """

    def __init__(self, model_name=None, dataset_name=None, augment_method='none', config_dict={}):
        """
        :param model_name: the model name, default is None, if it is None, config will search the parameter 'model' from the external input as the model name.
        :param dataset_name: the dataset name, default is None, if it is None, config will search the parameter 'dataset' from the external input as the dataset name.
        :param is_augment: the flag to indicate whether to use data augmentation or not.
        :param config_dict: the external parameter dictionaries, default is None.
        """
        super().__init__()
        # internal config
        self.internal_config_dict = {}
        self.path_config_dict = {}

        # external config
        self.external_config_dict = {}
        self.model_config_dict = {}
        self.dataset_config_dict = {}

        # cmd config
        self.cmd_config_dict = {}

        # final config
        self.final_config_dict = {}

        # load internal config from file
        self._load_internal_config()

        # initialize external config
        self._init_external_config(model_name, dataset_name, augment_method, config_dict)
        # load cmd line
        self._load_cmd_line()

        self._build_path_config()
        # load model config
        self._load_model_config()
        # load dataset config
        self._load_dataset_config()

        # merge model and dataset config to external config
        self._merge_external_config_dict()

        # merge internal, external and cmd line config to final config
        self._build_final_config_dict()

        # self._init_model_path()
        self._init_device()

        self.convert_none()

    def _update(self, external_config):
        for key in external_config:
            self.final_config_dict[key] = external_config[key]

    def _load_internal_config(self):
        dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir, 'config.json')
        self.internal_config_dict = read_json_data(config_path)

    def _init_external_config(self, model_name, dataset_name, augment_method, config_dict):
        # print(config_dict)
        self.external_config_dict['model'] = model_name
        self.external_config_dict['dataset'] = dataset_name
        self.external_config_dict['augment'] = augment_method
        self.external_config_dict.update(config_dict)

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum, None)):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    elif param.lower() == "none":
                        value = None
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_cmd_line(self):
        """
        Read parameters from command line and convert it to str.
        """
        cmd_config_dict = dict()
        arg_tuple = None

        for arg in sys.argv[1:]:
            if arg.startswith("-") and arg not in known_args:
                if arg_tuple:
                    cmd_arg_name, cmd_arg_value = arg_tuple
                    cmd_arg_name = cmd_arg_name.replace("-", "")
                    if cmd_arg_value:
                        if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                            raise SyntaxError("There are duplicate command arguments '%s' with different value."
                                              % cmd_arg_name)
                        else:
                            cmd_config_dict[cmd_arg_name] = cmd_arg_value
                    else:
                        cmd_config_dict[cmd_arg_name] = True

                arg_tuple = (arg, None)
            else:
                if arg_tuple:
                    arg_name, _ = arg_tuple
                    arg_tuple = (arg_name, arg)
                else:
                    continue

        if arg_tuple:
            cmd_arg_name, cmd_arg_value = arg_tuple
            cmd_arg_name = cmd_arg_name.replace("-", "")
            if cmd_arg_value:
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate command arguments '%s' with different value."
                                      % cmd_arg_name)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
            else:
                cmd_config_dict[cmd_arg_name] = True

        cmd_config_dict = self._convert_config_dict(cmd_config_dict)

        self.cmd_config_dict.update(cmd_config_dict)

        for key, value in self.external_config_dict.items():
            try:
                self.external_config_dict[key] = self.cmd_config_dict[key]
            except KeyError:
                pass
        for key, value in self.internal_config_dict.items():
            try:
                self.internal_config_dict[key] = self.cmd_config_dict[key]
            except KeyError:
                pass
        return cmd_config_dict

    def _load_model_config(self):
        if self.internal_config_dict["load_best_config"]:
            model_config_path = self.path_config_dict["best_config_file"]
        else:
            model_config_path = self.path_config_dict["model_config_file"]
        if not os.path.isabs(model_config_path):
            model_config_path = os.path.join(os.getcwd(), model_config_path)
        try:
            self.model_config_dict = read_json_data(model_config_path)
        except FileNotFoundError:
            warnings.warn('model config file is not exist, file path : {}'.format(model_config_path))
            self.model_config_dict = {}
        for key, value in self.model_config_dict.items():
            try:
                self.model_config_dict[key] = self.external_config_dict[key]
            except KeyError:
                pass
            try:
                self.model_config_dict[key] = self.cmd_config_dict[key]
            except KeyError:
                pass

    def _load_dataset_config(self):
        dataset_config_file = self.path_config_dict["dataset_config_file"]
        if not os.path.isabs(dataset_config_file):
            dataset_config_file = os.path.join(os.getcwd(), dataset_config_file)
        try:
            self.dataset_config_dict = read_json_data(dataset_config_file)
        except FileNotFoundError:
            warnings.warn('dataset config file is not exist, file path : {}'.format(dataset_config_file))
            self.dataset_config_dict = {}
        for key, value in self.dataset_config_dict.items():
            try:
                self.dataset_config_dict[key] = self.external_config_dict[key]
            except KeyError:
                pass
            try:
                self.dataset_config_dict[key] = self.cmd_config_dict[key]
            except KeyError:
                pass
        # print('_load_dataset_config', self.dataset_config_dict)

    def _build_path_config(self):
        path_config_dict = {}
        dir = os.path.dirname(os.path.realpath(__file__))
        model_name = self.external_config_dict['model']
        dataset_name = self.external_config_dict['dataset']
        if model_name is None:
            model_name = self.cmd_config_dict["model"]
        if dataset_name is None:
            dataset_name = self.cmd_config_dict["dataset"]

        model_config_file = os.path.join(dir, "../properties/model/{}.json".format(model_name))
        best_config_file = os.path.join(dir, "../properties/best_config/{}_{}.json".format(model_name, dataset_name))
        dataset_config_file = os.path.join(dir, "../properties/dataset/{}.json".format(dataset_name))
        path_config_dict["model_config_file"] = os.path.relpath(model_config_file, os.getcwd())
        path_config_dict["best_config_file"] = os.path.relpath(best_config_file, os.getcwd())
        path_config_dict["dataset_config_file"] = os.path.relpath(dataset_config_file, os.getcwd())

        path_config_dict["dataset_dir"] = "dataset/{}".format(dataset_name)

        path_config_dict["checkpoint_file"] = 'checkpoint/' + '{}-{}.pth'.format(model_name, dataset_name)
        # path_config_dict["trained_model_dir"] = 'trained_model/' + '{}-{}'.format(model_name, dataset_name)
        path_config_dict["log_file"] = 'log/' + '{}-{}.log'.format(model_name, dataset_name)
        path_config_dict["output_dir"] = 'result/{}-{}'.format(model_name, dataset_name)
        path_config_dict["checkpoint_dir"] = 'checkpoint/' + '{}-{}'.format(model_name, dataset_name)

        self.path_config_dict = path_config_dict

        for key, value in path_config_dict.items():
            try:
                self.path_config_dict[key] = self.external_config_dict[key]
            except KeyError:
                pass
            try:
                self.path_config_dict[key] = self.cmd_config_dict[key]
            except KeyError:
                pass

        # merge path config into internal config
        self.internal_config_dict.update(self.path_config_dict)

    def _init_model_path(self):
        path_config_dict = {}
        model_name = self.final_config_dict["model"]
        dataset_name = self.final_config_dict["dataset"]
        fix = self.final_config_dict["equation_fix"]
        path_config_dict["checkpoint_file"] = 'checkpoint/' + '{}-{}-{}.pth'.format(model_name, dataset_name, fix)
        # path_config_dict["trained_model_dir"] = 'trained_model/' + '{}-{}-{}'.format(model_name, dataset_name, fix)
        path_config_dict["log_file"] = 'log/' + '{}-{}-{}.log'.format(model_name, dataset_name, fix)
        for key, value in path_config_dict.items():
            try:
                path_config_dict[key] = self.external_config_dict[key]
            except KeyError:
                pass
            try:
                path_config_dict[key] = self.cmd_config_dict[key]
            except KeyError:
                pass
        self.path_config_dict.update(path_config_dict)
        self.final_config_dict.update(path_config_dict)

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.dataset_config_dict)
        external_config_dict.update(self.model_config_dict)
        external_config_dict.update(self.external_config_dict)
        # external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _build_final_config_dict(self):
        self.final_config_dict.update(self.internal_config_dict)
        self.final_config_dict.update(self.external_config_dict)
        self.final_config_dict.update(self.cmd_config_dict)

    def _init_device(self):
        if self.final_config_dict["gpu_id"] == None:
            if torch.cuda.is_available() and self.final_config_dict["use_gpu"]:
                self.final_config_dict["gpu_id"] = "0"
            else:
                self.final_config_dict["gpu_id"] = ""
        else:
            if self.final_config_dict["use_gpu"] != True:
                self.final_config_dict["gpu_id"] = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict["gpu_id"])
        self.final_config_dict['device'] = torch.device("cuda") if torch.cuda.is_available() else 0
        self.final_config_dict["map_location"] = "cuda" if torch.cuda.is_available() else "cpu"
        self.final_config_dict['gpu_nums'] = torch.cuda.device_count()

    def _update_internal_config(self, key, value):
        if key in self.internal_config_dict:
            self.internal_config_dict[key] = value
        if key in self.path_config_dict:
            self.path_config_dict[key] = value

    def _update_external_config(self, key, value):
        if key in self.external_config_dict:
            self.external_config_dict[key] = value
        if key in self.model_config_dict:
            self.model_config_dict[key] = value
        if key in self.dataset_config_dict:
            self.dataset_config_dict[key] = value

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        config_file = os.path.join(pretrained_dir, 'config.json')
        config_dict = read_json_data(config_file)
        # print(config_dict)
        model_name = config_dict['final_config_dict']['model']
        dataset_name = config_dict['final_config_dict']['dataset']
        # task_type = config_dict['final_config_dict']['language']
        config = Config(model_name, dataset_name)
        for key, value in config_dict.items():
            setattr(config, key, value)
        config._load_cmd_line()
        config._build_path_config()
        config._build_final_config_dict()
        config._init_device()
        config.convert_none()
        return config

    def save_config(self, trained_dir):
        json_encoder = json.encoder.JSONEncoder()
        config_file = os.path.join(trained_dir, 'config.json')
        config_dict = self.to_dict()
        not_support_json = []
        for key1, value1 in config_dict.items():
            for key2, value2 in value1.items():
                try:
                    json_encoder.encode({key2: value2})
                except TypeError:
                    # del config_dict[key1][key2]
                    not_support_json.append([key1, key2])
        for keys in not_support_json:
            del config_dict[keys[0]][keys[1]]
        write_json_data(config_dict, config_file)

    def to_dict(self):
        config_dict = {}
        for name, value in vars(self).items():
            if hasattr(eval('self.{}'.format(name)), '__call__') or re.match('__.*?__', name):
                continue
            else:
                config_dict[name] = copy.deepcopy(value)
        return config_dict

    def convert_none(self):
        for name in self.final_config_dict:
            if self.final_config_dict[name] == "None":
                self.final_config_dict[name] = None

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        value = self._convert_config_dict({key: value})[key]
        self.final_config_dict[key] = value
        self._update_internal_config(key, value)
        self._update_external_config(key, value)

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __delitem__(self, key):
        del self.final_config_dict[key]
        del self.external_config_dict[key]
        del self.model_config_dict[key]
        del self.dataset_config_dict[key]
        del self.internal_config_dict[key]
        del self.path_config_dict[key]

    def __str__(self):
        args_info = ''
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info
