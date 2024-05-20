# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/30 11:06
# @File: evaluator.py


import copy
import re
import threading
from typing import Type, Union

import sympy as sym

from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.config.configuration import Config

class Evaluator(object):
    def __init__(self, config, tokenizer):
        #print('self.vocab.pad_token_id', tokenizer.pad_token_id);

        special_tokens = [SpecialTokens.PAD_TOKEN, SpecialTokens.CLS_TOKEN, SpecialTokens.SEP_TOKEN, SpecialTokens.MASK_TOKEN]
        self.data_special_ids = [tokenizer.convert_tokens_to_ids(w) for w in special_tokens]
        #print('self.data_special_ids', self.data_special_ids); exit()
        self.pred_special_ids = [tokenizer.convert_tokens_to_ids(w) for w in [SpecialTokens.PAD_TOKEN]]

    def measure(self, sources, labels, predicts):
        """
        https://github.com/sunnyqiny/Confusionset-guided-Pointer-Networks-for-Chinese-Spelling-Check/blob/master/utils/evaluation_metrics.py
        """
        # print("source", sources)
        # print("labels", labels)
        # print("predicts", predicts)
        TP = 0
        FP = 0
        FN = 0

        src, tgt, predict = sources, labels, predicts
        src = [w for w in src if w not in self.data_special_ids]
        tgt = [w for w in tgt if w not in self.data_special_ids]
        predict = [w for w in predict if w not in self.pred_special_ids]

        gold_index = []
        each_true_index = []
        for i in range(len(src)):
            if i >= len(tgt) or src[i] == tgt[i]:
                continue
            else:
                gold_index.append(i)

        predict_index = []
        for i in range(len(src)):
            if i >= len(predict):
                predict_index.append(i)
                continue
            if src[i] == predict[i]:
                continue
            else:
                predict_index.append(i)
        #print('gold_index', gold_index, 'predict_index', predict_index); #exit()
        if len(gold_index) == 0 and len(predict_index) == 0:
            return 1., 1.

        for i in predict_index:
            if i in gold_index:
                TP += 1
                each_true_index.append(i)
            else:
                FP += 1
        for i in gold_index:
            if i in predict_index:
                continue
            else:
                FN += 1

        # For the detection Precision, Recall and F1
        detection_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
        detection_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
        detection_f1 = 1.25 * (detection_precision * detection_recall) / (0.25 * detection_precision + detection_recall) if (detection_precision + detection_recall) > 0 else 0
        #print("The detection result is precision={}, recall={} and F1={}".format(detection_precision, detection_recall, detection_f1))
        #exit()
        TP = 0
        FP = 0
        FN = 0

        # for i in range(len( all_predict_true_index)):
        #     # we only detect those correctly detected location, which is a different from the common metrics since
        #     # we wanna to see the precision improve by using the confusionset
        # print("src", src)
        # print("tgt", tgt)
        # print("predict", predict)
        # print(each_true_index, gold_index)

        if len(each_true_index) > 0:
            predict_words = []
            for j in each_true_index:
                if j >= len(predict):
                    FP += 1; predict_words = []
                    continue
                predict_words.append(predict[j])
                if tgt[j] == predict[j]:
                    TP += 1
                else:
                    FP += 1
            for j in gold_index:
                if tgt[j] in predict_words:
                    continue
                else:
                    FN += 1

        # For the correction Precision, Recall and F1
        correction_precision = TP / (TP + FP) if (TP+FP) > 0 else 0
        correction_recall = TP / (TP + FN) if (TP+FN) > 0 else 0
        correction_f1 = 1.25 * (correction_precision * correction_recall) / (0.25 * correction_precision + correction_recall) if (correction_precision + correction_recall) > 0 else 0
        #print("The correction  result is precision={}, recall={} and F1={}".format(correction_precision, correction_recall, correction_f1))

        return detection_f1, correction_f1
