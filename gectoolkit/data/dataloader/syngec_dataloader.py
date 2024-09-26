import os
import sys

import random
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

fairseq_path = os.path.abspath(os.path.join(os.getcwd(), "gectoolkit", "model", "SynGEC", "src",
                               "src_syngec", "fairseq2"))
sys.path.insert(0, fairseq_path)

from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils
)
from fairseq.data import encoders, iterators
from fairseq.trainer import Trainer
from gectoolkit.data.dataloader.abstract_dataloader import AbstractDataLoader
from gectoolkit.data.dataset.syngec_dataset import SyngecDataset
from gectoolkit.config import Config


class SyngecTokenizer(object):
    def __init__(self, dataset, language_name):
        if language_name == 'Chinese':
            self.tokenizer = SyngecChineseTokenizer(dataset)
        else:
            self.tokenizer = SyngecEnglishTokenizer(dataset)


class SyngecChineseTokenizer(object):
    def __init__(self,dataset):
        self.src_dict = dataset.src_dict
        self.tgt_dict = dataset.tgt_dict

    def index(self,tokens):
        ids = self.src_dict.index(tokens)
        return ids

    def convert_tokens_to_ids(self, tokens):
        ids = self.src_dict.encode_tokens(tokens)
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        remove_bpe = "@@ "
        tokens = self.tgt_dict.string(ids, remove_bpe)
        tokens = tokens.replace('<pad>', '').strip()
        tokens = tokens.split()
        return tokens

    def decode(self,ids):
        remove_bpe = "@@ "
        tokens = self.tgt_dict.string(ids, remove_bpe)
        return tokens


class SyngecEnglishTokenizer(object):
    def __init__(self, dataset):
        self.src_dict = dataset.src_dict
        self.tgt_dict = dataset.tgt_dict
        self.encoder_json = dataset.encoder_json
        self.vocab_bpe = dataset.vocab_bpe
        self.args_generate = dataset.args_generate
        self.bpe = encoders.build_bpe(self.args_generate)

    def convert_ids_to_tokens(self,ids):
        remove_bpe = "@@ "
        extra_symbols_to_ignore = {2}
        ids_str = self.tgt_dict.string(ids, remove_bpe,extra_symbols_to_ignore)

        words = ids_str.split()
        words = [word for word in words if word != '<pad>']
        ids_str_without_pad = ' '.join(words)

        text = self.bpe.decode(ids_str_without_pad)
        return text

    def convert_tokens_to_ids(self,tokens):
        tokens = self.bpe.encode(tokens)
        return tokens

    def str_decode(self, str):
        tokens = self.bpe.decode(str)
        return tokens

    def str_encode(self, str):
        ids = self.bpe.encode(str)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        text = self.convert_ids_to_tokens(ids)
        return text

    def bpe_encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def bpe_decode(self, line):
        global bpe
        tokens = line.split()
        len_1 = len(tokens)
        tokens_map = map(int, tokens)
        text = [bpe.decoder.get(token, token) for token in tokens_map]
        for idx, tok in enumerate(text):
            if idx == 0:
                continue
            if tok[0] == "Ġ":
                tok = tok[1:]
                text[idx] = tok
            else:
                text[idx - 1] = text[idx - 1] + "@@"
        len_2 = len(text)
        assert len_1 == len_2, print(str(tokens), str(text))
        text1 = " ".join(text)
        print('text:',text1)
        return text1

    def encode_lines(self, lines):
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        global bpe
        dec_lines = []
        for line in lines:
            line = line.strip()
            res = self.decode(line)
            dec_lines.append(res)
        return ["PASS", dec_lines]


class SyngecDataLoader(AbstractDataLoader):
    def __init__(self, config: Config, dataset: SyngecDataset):
        super().__init__(config, dataset)
        self.dataset = dataset
        self.trainset_nums = len(dataset.trainset)
        self.validset_nums = len(dataset.validset)
        self.testset_nums = len(dataset.testset)
        self.language_name = self.dataset.language_name  # Spanish
        self.dataset_name = self.dataset.dataset # conll14

        self.trainset_batch_nums = self.trainset_nums/self.train_batch_size
        self.validset_batch_nums = self.validset_nums/self.test_batch_size

        self.max_input_len = config["max_input_len"]

        self.pretrained_tokenizer = SyngecTokenizer(dataset,self.language_name).tokenizer

        self.replaced_symbols = []

        self.args = self.dataset.args_bart
        self.task = self.dataset.task_bart
        self.args.required_batch_size_multiple = 1
        self.args.batch_size = self.train_batch_size
        self.args.max_epoch = config["epoch_nums"]

        self.model = None
        self.criterion = None
        self.quantizer = None
        self.optimizer = None
        self.trainer = None

        self.get_trainer()

        self.valid_itr = self.build_valid_itr()
        self.test_itr = self.build_test_itr()
        self.train_itr = self.build_train_itr()

        # self.model = config["model"].lower()
        self.__init_batches()


    def get_trainer(self):
        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator

        model = self.task.build_model(self.args)
        criterion = self.task.build_criterion(self.args)
        # (optionally) Configure quantization
        if self.args.quantization_config_path is not None:  # 是否提供了量化配置文件的路径
            self.quantizer = quantization_utils.Quantizer(
                config_path=self.args.quantization_config_path,
                max_epoch=self.args.max_epoch,
                max_update=self.args.max_update,
            )
        else:
            self.quantizer = None

        self.trainer = Trainer(self.args, self.task, model, criterion, self.quantizer)

        self.model = self.trainer.model
        self.criterion = self.trainer.criterion
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler

    def build_train_itr(self):
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
            self.args,
            self.trainer,
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=self.task.has_sharded_data("train"),  # 如果任务（task）具有分片数据（sharded data）用于训练，则禁用迭代器缓存。
        )

        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=self.args.fix_batches_to_gpus,          # 每个GPU上固定batch的大小
            shuffle=(epoch_itr.next_epoch_idx > self.args.curriculum),  # 如果当前epoch的索引大于args.curriculum，则进行shuffle。
        )
        update_freq = 1                                    # 确定在训练过程中更新参数的频率。(不进行梯度累加)
        itr = iterators.GroupedIterator(itr, update_freq)  # 使用GroupedIterator将数据迭代器进行分组，以便按照update_freq的频率更新参数。
        return itr

    def build_test_itr(self):
        task = self.dataset.task_generate
        args = self.dataset.args_generate
        testset = self.dataset.testset
        max_positions = (1024, 1024)
        max_sentences = self.test_batch_size
        itr = task.get_batch_iterator(
            dataset=testset,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            type='test'
        ).next_epoch_itr(shuffle=False)
        return itr

    def build_valid_itr(self):
        task = self.dataset.task_generate
        args = self.dataset.args_generate
        validset = self.dataset.validset
        max_positions = (1024, 1024)
        max_sentences = self.test_batch_size
        itr = task.get_batch_iterator(
            dataset=validset,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            type='valid'
        ).next_epoch_itr(shuffle=False)
        return itr


    def load_data(self,type:str):
        """
                Load batches, return every batch data in a generator object.
                :param type: [train | valid | test], data type.
                :return: Generator[dict], batches
                """
        if type == "train":
            # return self.train_itr
            self.__trainset_batch_idx = -1
            if self.shuffle:
                random.shuffle(self.trainset_batches)
            for batch in self.trainset_batches:
                self.__trainset_batch_idx = (self.__trainset_batch_idx + 1) % self.trainset_batch_nums
                yield batch
        elif type == "valid":
            self.__validset_batch_idx = -1
            for batch in self.validset_batches:
                self.__validset_batch_idx = (self.__validset_batch_idx + 1) % self.validset_batch_nums
                yield batch
        elif type == "test":
            self.__testset_batch_idx = -1
            for batch in self.testset_batches:
                self.__testset_batch_idx = (self.__testset_batch_idx + 1) % self.testset_batch_nums
                yield batch
        else:
            raise ValueError("{} type not in ['train', 'valid', 'test'].".format(type))


    def __init_batches(self):
        self.trainset_batches = []
        self.validset_batches = []
        self.testset_batches = []

        for batch in self.valid_itr:
            batch['source_list_batch'] = batch['net_input']['src_tokens']
            batch['target_list_batch'] = batch['target']
            self.validset_batches.append(batch)

        for batch in self.test_itr:
            batch['source_list_batch'] = batch['net_input']['src_tokens']
            batch['target_list_batch'] = batch['target']
            self.testset_batches.append(batch)

        for batch in self.train_itr:
            batch = batch[0]
            batch['source_list_batch'] = batch['net_input']['src_tokens']
            batch['target_list_batch'] = batch['target']
            self.trainset_batches.append(batch)

        self.__trainset_batch_idx = -1
        self.__validset_batch_idx = -1
        self.__testset_batch_idx = -1
        self.trainset_batch_nums = len(self.trainset_batches)
        self.validset_batch_nums = len(self.validset_batches)
        self.testset_batch_nums = len(self.testset_batches)
