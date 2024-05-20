# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/10 15:22
# @File: abstract_model.py
# @Update Time: 2023/02/10 15:23

from torch import nn


class AbstractModel(nn.Module):
    def __init__(self):
        super(AbstractModel, self).__init__()

    def calculate_loss(self, batch_data: dict):
        raise NotImplementedError

    def model_test(self, batch_data: dict):
        raise NotImplementedError

    def predict(self, batch_data: dict, output_all_layers: bool = False):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        raise NotImplementedError

    def save_model(self, trained_dir):
        raise NotImplementedError

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = "\ntotal parameters : {} \ntrainable parameters : {}".format(total, trainable)
        return info + parameters
