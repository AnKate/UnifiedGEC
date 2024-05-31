# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/30 11:19
# @File: gec_dataloader.py


import math
import torch
from typing import List
import numpy as np
import re

from gectoolkit.config import Config
from gectoolkit.data.dataset.abstract_dataset import AbstractDataset
from gectoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.utils.preprocess_data import convert_data_to_vocab

from transformers import AutoTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields import TextField, SequenceLabelField


class GECDataLoader(AbstractDataLoader):
    """
    dataloader class for deep-learning model EPT
    """

    def __init__(self, config: Config, dataset: AbstractDataset):
        super().__init__(config, dataset)

        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.max_input_len = config["max_input_len"]
        self.language_name = dataset.language_name

        self.tagging_rule = config["model"]
        self.model_path = config["pretrained_model_path"]
        if self.tagging_rule in ["GECToR", "Transformer", "LevenshteinTransformer"]:
            if self.language_name == 'zh':
                self.model_path = self.model_path + '/Chinese/'
            elif self.language_name == 'en':
                self.model_path = self.model_path + '/English/'
            else:
                self.model_path = self.model_path + '/Multilingual/'

        if config['cache_dir'] is not None:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                                      cache_dir=config['cache_dir'])
        else:
            self.pretrained_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        special_tokens = [SpecialTokens.__dict__[k] for k in SpecialTokens.__dict__ if not re.search('^\_', k)]
        special_tokens.sort()
        self.pretrained_tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

        # 当模型是GECToR时, 使用特殊的数据处理方法
        if self.tagging_rule == "GECToR":
            bert_token_indexer = PretrainedTransformerIndexer(model_name=self.model_path, namespace="bert")
            self.pretrained_tokenizer = {'bert': bert_token_indexer}
            self.vocab_path = self.model_path + "vocabulary/"
            self.min_count = config["min_count"]
            self.worker_num = config["worker_num"]
            self.tag_strategy = config["tag_strategy"]
            self.skip_complex = config["skip_complex"]
            self.save_vocab = config["save_vocab"]

        # if self.language_name == 'zh':
        #     self.pre_tokenizer = lambda x: [w for w in x.strip()]
        # else:
        #     self.pre_tokenizer = lambda x: self.pretrained_tokenizer.tokenize(x, add_special_tokens=False)

        # 处理unknown token
        self.replaced_symbols = []

        self.__init_batches()

    def __build_batch(self, batch_data):
        """load one batch

        Args:
            batch_data (list[dict])
        
        Returns:
            loaded batch data (dict)
        """
        source_list_batch = []
        target_list_batch = []
        source_batch = []
        target_batch = []

        instance_batch = []

        for data in batch_data:
            if self.tagging_rule == 'GECToR':
                if self.language_name == 'zh':
                    source = [i for i in data["source_text"].strip()][:self.max_input_len]
                    target = [i for i in data['target_text'].strip()][:self.max_input_len]
                else:
                    source = data['source_text'].split(' ')[:self.max_input_len]
                    target = data['target_text'].split(' ')[:self.max_input_len]

                sor_list = source
                tag_list = target
                instance_batch.append(data.get('instance'))
            else:
                source = data['source_text']
                target = data['target_text']

                sor_list = self.pretrained_tokenizer.encode(source, add_special_tokens=False)
                tag_list = self.pretrained_tokenizer.encode(target, add_special_tokens=False)

            source_batch.append(source)
            target_batch.append(target)
            source_list_batch.append(sor_list)
            target_list_batch.append(tag_list)

        return {
            "source_batch": source_batch,
            "target_batch": target_batch,
            "source_list_batch": source_list_batch,
            "target_list_batch": target_list_batch,
            "instance_batch": instance_batch
        }

    def __init_batches(self):
        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []
        for set_type in ['train', 'valid', 'test']:
            if set_type == 'train':
                datas = self.dataset.trainset
                batch_size = self.train_batch_size
                if self.tagging_rule == 'GECToR':
                    datas = self.GECToR_preprocess(datas, set_type)
            elif set_type == 'valid':
                datas = self.dataset.validset
                batch_size = self.test_batch_size
                if self.tagging_rule == 'GECToR':
                    datas = self.GECToR_preprocess(datas, set_type)
            elif set_type == 'test':
                datas = self.dataset.testset
                batch_size = self.test_batch_size
            else:
                raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
            num_total = len(datas)
            batch_num = math.ceil(num_total / batch_size)
            for batch_i in range(batch_num):
                start_idx = batch_i * batch_size
                end_idx = (batch_i + 1) * batch_size
                if end_idx <= num_total:
                    batch_data = datas[start_idx:end_idx]
                else:
                    batch_data = datas[start_idx:num_total]
                built_batch = self.__build_batch(batch_data)
                if set_type == 'train':
                    self.trainset_batches.append(built_batch)
                elif set_type == 'valid':
                    self.validset_batches.append(built_batch)
                elif set_type == 'test':
                    self.testset_batches.append(built_batch)
                else:
                    raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))
        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = len(self.trainset_batches)
        self.validset_batch_nums = len(self.validset_batches)
        self.testset_batch_nums = len(self.testset_batches)

    def build_batch_for_predict(self, batch_data: List[dict]):
        raise NotImplementedError

    def truncate_tensor(self, sequence, pad_token=None):
        max_len = 0
        for instance in sequence:
            max_len = max(len(instance), max_len)
        result_batch_tag_list = list()
        for instance in sequence:
            one_tag_list = []
            one_tag_list.extend(instance)
            len_diff = max_len - len(one_tag_list)
            for _ in range(len_diff):
                if pad_token is not None:
                    one_tag_list.append(self.pretrained_tokenizer.convert_tokens_to_ids(pad_token))  # for padding
                else:
                    one_tag_list.append(self.pretrained_tokenizer.convert_tokens_to_ids('<-PAD->'))
            result_batch_tag_list.append(one_tag_list)

        result_batch_tag_matrix = np.array(result_batch_tag_list)
        result_batch_tag_matrix = torch.tensor(result_batch_tag_matrix)

        return result_batch_tag_matrix

    def GECToR_preprocess(self, datas, set_type):
        vocab = Vocabulary.from_files(self.vocab_path)

        source_lines = []
        target_lines = []
        removed_sent_id = []    # 用于记录在这一步骤中被跳过的句子id
        for i in range(len(datas)):
            data = datas[i]
            if self.language_name == "zh":
                source_line = " ".join(data['source_text'])
                target_line = " ".join(data['target_text'])
            else:
                source_line = data['source_text']
                target_line = data['target_text']
            if not source_line.strip() or not target_line.strip():
                removed_sent_id.append(i)
                continue
            else:
                source_lines.append(source_line)
                target_lines.append(target_line)

        assert len(source_lines) == len(target_lines)

        # preprocess, 生成vocab.txt和label格式的数据
        if set_type == 'train' and self.save_vocab:
            save_vocab_flag = True
        else:
            save_vocab_flag = False
        tagged_text = convert_data_to_vocab(source_lines, target_lines, self.vocab_path,
                                            self.min_count, save_vocab_flag, self.worker_num)
        # exit()

        instances = []
        # 实现datareader的功能, 基于label格式的数据生成对应fields, 同时在这一步中将token转换为id
        for tagged_line in tagged_text:
            # print(tagged_line)
            tokens_and_tags = [pair.rsplit("SEPL|||SEPR") for pair in tagged_line.split(" ")]
            try:
                tokens = [Token(token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]
            except ValueError:
                tokens = [Token(token) for token, tag in tokens_and_tags]
                tags = [tag for token, tag in tokens_and_tags]

            if tokens and tokens[0] != Token("$START"):
                tokens = [Token("$START")] + tokens

            if self.max_input_len is not None:
                tokens = tokens[:self.max_input_len]
                tags = None if tags is None else tags[:self.max_input_len]

            # 将tokenize后的token序列转为instance实例对象
            fields = {}
            sequence = TextField(tokens, self.pretrained_tokenizer)  # token_indexer将tokens转为ids
            fields["tokens"] = sequence
            # 抽取编辑标签
            if tags is not None:
                labels = [x.split("SEPL__SEPR") for x in tags]
                labels_final = []

                for x in labels:
                    if len(x) == 1:
                        labels_final.append(x[0])
                    elif len(x) > 5:
                        if self.skip_complex:
                            labels_final.append("$KEEP")
                        else:
                            labels_final.append(x[1] if x[0] == "$KEEP" else x[0])
                    else:
                        labels_final.append(x[1] if x[0] == "$KEEP" else x[0])

                detect_tags = ["CORRECT" if label == "$KEEP" else "INCORRECT" for label in labels_final]

                fields["labels"] = SequenceLabelField(labels_final, sequence, label_namespace="labels")
                fields["d_tags"] = SequenceLabelField(detect_tags, sequence, label_namespace="d_tags")

                instance = Instance(fields)

                # DatasetReader的功能完成, 此处需要通过vocab完成token到index的转换
                instance.index_fields(vocab)
                # print(instance); exit()
                instances.append(instance)

        assert len(instances) == len(tagged_text)

        # return instances
        ret_datas = []
        j = 0
        for i in range(len(datas)):
            if i in removed_sent_id:
                continue
            ret_data = datas[i]
            ret_data['instance'] = instances[j]
            ret_datas.append(ret_data)
            j += 1

        # exit()
        return ret_datas
