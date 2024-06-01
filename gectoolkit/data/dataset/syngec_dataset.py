import os.path
from collections import namedtuple
import json
import argparse
import sys

fairseq_path = os.path.abspath(os.path.join(os.getcwd(), "gectoolkit", "model", "SynGEC", "src",
                               "src_syngec", "fairseq2"))
sys.path.insert(0, fairseq_path)

from fairseq import tasks, utils
from fairseq.data import indexed_dataset, data_utils

Batch = namedtuple("Batch", "ids src_tokens src_lengths tgt_tokens constraints src_nt src_nt_lengths src_outcoming_arc_mask src_incoming_arc_mask src_dpd_matrix src_probs_matrix")
Translation = namedtuple("Translation", "src_str hypos pos_scores alignments")


class SyngecDataset(object):
    """Chinese data
    the base class of data class
    """

    def __init__(self, config):
        super().__init__()
        self.model = config["model"]  # Syngec
        self.dataset = config["data"] # mucgec/nlpcc18

        if config["language"] == 'zh':
            self.language_name = 'Chinese'
        elif config["language"] == 'en':
            self.language_name = 'English'
        elif config["language"] == 'cs':
            self.language_name = 'Czech'
        elif config["language"] == 'es':
            self.language_name = 'Spanish'
        elif config["language"] == 'de':
            self.language_name = 'German'

        self.dataset_path = config['dataset_dir'] if config['dataset_dir'] else config["dataset_path"] # 用不上 data/mucgec

        self.validset_divide = config["validset_divide"] # True
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

        # 中文tokenizer：关于字符-id的字典
        self.src_dict = []
        self.tgt_dict = []

        # 英文tokenizer：关于字符-id的字典
        self.encoder_json = config['encoder_json']
        self.vocab_bpe = config['vocab_bpe']

        bart_path = os.path.join(os.getcwd(), "gectoolkit", "properties", "model", "SynGEC", "args", self.language_name,
                                 self.language_name + '_' + self.dataset + '_bart.json')
        generate_path = os.path.join(os.getcwd(), "gectoolkit", "properties", "model", "SynGEC", "args", self.language_name,
                                     self.language_name + '_' + self.dataset + '_generate.json')

        # 获取bart和generate的参数
        self.args_bart = self.load_args(bart_path)
        self.args_generate = self.load_args(generate_path)

        # 根据args的json文件，创建task任务
        self.task_bart = self.load_task(self.args_bart)
        self.task_generate = self.load_task(self.args_generate)

    def load_args(self,args_path):
        with open(args_path, 'r') as json_file:
            args_dict = json.load(json_file)
        args = argparse.Namespace(**args_dict)
        return args

    def load_task(self,args):
        utils.import_user_module(args)
        task = tasks.setup_task(args)
        return task

    def _load_dataset(self):
        # Load trainset and validset
        self.trainset, self.validset = self.get_trainset_validset(self.task_bart)
        # Load testset
        self.testset = self.get_testset(self.task_generate)


    def get_trainset_validset(self,task):
        task.load_dataset('valid', combine=False, epoch=1)
        task.load_dataset('train', combine=False, epoch=1)

        trainset = task.datasets['train']
        validset = task.datasets['valid']

        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary
        return trainset, validset

    def get_testset(self,task):
        if self.language_name != 'Chinese':
            task.load_dataset('test', combine=False, epoch=1)
            testset = task.datasets['test']

        elif self.language_name == 'Chinese':
            # 读取句法信息，这里直接复用了写好的dataset
            src_conll_dataset = None
            src_dpd_dataset = None
            src_probs_dataset = None
            if task.args.conll_file:
                # 句法掩码矩阵(GAT&GCN)
                src_conll_dataset = []
                src_conll_paths = task.args.conll_file
                for src_conll_path in src_conll_paths:
                    if indexed_dataset.dataset_exists(src_conll_path, impl="mmap"):
                        src_conll_dataset.append(data_utils.load_indexed_dataset(
                            src_conll_path, None, "mmap"
                        ))
                    else:
                        print(src_conll_path)
                        raise FileNotFoundError
                if task.args.dpd_file:
                    # 依存距离矩阵(DSA)
                    src_dpd_dataset = []
                    src_dpd_paths = task.args.dpd_file
                    for src_dpd_path in src_dpd_paths:
                        if indexed_dataset.dataset_exists(src_dpd_path, impl="mmap"):
                            src_dpd_dataset.append(data_utils.load_indexed_dataset(
                                src_dpd_path, None, "mmap"
                            ))
                        else:
                            print(src_dpd_path)
                            raise FileNotFoundError
                if task.args.probs_file:
                    # 句法概率矩阵(Soft GCN)
                    src_probs_dataset = []
                    src_probs_paths = task.args.probs_file
                    for src_probs_path in src_probs_paths:
                        if indexed_dataset.dataset_exists(src_probs_path, impl="mmap"):
                            src_probs_dataset.append(data_utils.load_indexed_dataset(
                                src_probs_path, None, "mmap"
                            ))
                        else:
                            print(src_probs_path)
                            raise FileNotFoundError

                        # 获取test中 原始句子，正确句子（每个字之间用空格分割），
                        with open(task.args.src_char_path, 'r', encoding='utf-8') as file:
                            src_lines = file.readlines()
                        with open(task.args.tgt_char_path, 'r', encoding='utf-8') as file:
                            tgt_lines = file.readlines()

                        # 中文：src_lines=['冬 阴 功','不 复 杂']
                        src_lines = [line.strip() for line in src_lines]
                        tgt_lines = [line.strip() for line in tgt_lines]

                        # 进行encode(list:2000) src_tokens=[tensor([1104,..]),tensor(),...]
                        src_tokens = [
                            task.source_dictionary.encode_line(
                                src_str, add_if_not_exist=False
                            ).long()
                            for src_str in src_lines
                        ]
                        # 获取每个lines的长度 src_lengths=[45,34,..]
                        src_lengths = [t.numel() for t in src_tokens]

                        tgt_tokens = [
                            task.target_dictionary.encode_line(
                                tgt_str, add_if_not_exist=False
                            ).long()
                            for tgt_str in tgt_lines
                        ]
                        tgt_lengths = [t.numel() for t in tgt_tokens]
                        constraints_tensor = None
                        src_nt = None
                        src_nt_sizes = None
            testset = task.build_dataset_for_inference(
                src_tokens=src_tokens, src_lengths=src_lengths,
                tgt_tokens=tgt_tokens, tgt_lengths=tgt_lengths,
                constraints=constraints_tensor, src_nt=src_nt, src_nt_sizes=src_nt_sizes,
                src_conll_dataset=src_conll_dataset, src_dpd_dataset=src_dpd_dataset, src_probs_dataset=src_probs_dataset,
                syntax_type=task.args.syntax_type
            ) if task.args.task == "syntax-enhanced-translation" else task.build_dataset_for_inference(
                src_tokens, src_lengths, constraints=constraints_tensor
            )
        return testset