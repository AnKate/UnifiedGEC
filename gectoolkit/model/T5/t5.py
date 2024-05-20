from transformers import T5Tokenizer, T5Model,T5ForConditionalGeneration
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
import sys
import time
from ...utils.enum_type import SpecialTokens
# from ...module.transformer import TransformerLayer, Embedding, SelfAttentionMask, LearnedPositionalEmbedding
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


from torch import nn

class T5(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.device = config["device"]
        self.num_class = num_class = len(dataset.vocab)
        self.vocab = dataset
        self.gamma = config["gamma"]
        # self.tokenizer = dataset
        # print('self.tokenzier:',self.tokenizer)
        # dataset_size = len(dataset.vocab)

        # 9.97  mT5模型的tokenzier：
        # self.model_path = "./gectoolkit/properties/model/T5large"
        self.model_path = "./gectoolkit/properties/model/T5"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        # print('tokenizer:',self.tokenizer)

        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        # print('model:',self.model)

        self.pad_token_id = self.tokenizer.pad_token_id  # 0
        # self.model.config.decoder_start_token_id = self.pad_token_id
        # print('self.pad_token_id:',self.pad_token_id)

        # the following 2 hyperparameters are task-specific
        self.max_source_length = 512
        self.max_target_length = 128

    # def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
    #     # compute cross entropy loss
    #     if gamma is None:
    #         gamma = 2
    #     p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1).long())
    #     g = (1-torch.clamp(p, min=0.01, max=0.99))**gamma
    #     #g = (1 - p) ** gamma
    #     cost = -g * torch.log(p+1e-8)
    #     cost = cost.view(y.shape)
    #     y_mask = y_mask.view(y.shape)
    #     if avg:
    #         cost = torch.sum(cost * y_mask, 1) / torch.sum(y_mask, 1)
    #     else:
    #         cost = torch.sum(cost * y_mask, 1)
    #     cost = cost.view((y.size(0), -1))
    #     return torch.mean(cost), g.view(y.shape)
    #
    # def compute_metric(self, logits, labels):
    #     # 计算预测类别
    #     y_pred = torch.argmax(logits, dim=-1)
    #     y_pred = y_pred.view(size=(-1,))
    #     print('y_pred:',y_pred) # 1440
    #
    #     # 创建掩码以标识填充部分
    #     padding_mask = (labels != -100).float()
    #     print('padding_mask:',padding_mask) # torch.Size([32, 45])
    #
    #     # 获取非填充部分的真实标签
    #     y_true = labels.view(size=(-1,))
    #     print('y_true:',y_true)
    #
    #     corr = torch.eq(y_pred, y_true)
    #     corr_float = corr.float()
    #     print('corr:',corr)
    #     print('corr.float():',corr_float)
    #     print('y_true.shape[0]:',y_true.float[0])
    #
    #     acc = torch.sum(corr.float()) / y_true.shape[0]
    #     # y_true = y_true.float()
    #     # # 将 y_pred 也转换为浮点型以匹配 y_true
    #     # y_pred = y_pred.float()
    #     #
    #     # loss_fn = nn.CrossEntropyLoss()  # reduction='none' 表示不对每个样本求平均
    #     # loss = loss_fn(y_pred, y_true)            # 注意将 target 变为 LongTensor
    #     #
    #     # # 使用填充掩码排除填充部分的损失
    #     # masked_loss = loss * padding_mask.view(-1)
    #     #
    #     # # 计算总损失
    #     # total_loss = torch.sum(masked_loss) / torch.sum(padding_mask)
    #     #
    #     # print("Total Loss:", total_loss.item())
    #     return loss

    def forward(self, batch, dataloader):
        '''
        parameters in batch.
        batch['source_batch'] is a list with the source text sentences;
        batch['target_batch'] is a list with the target text sentences;
        batch['source_list_batch'] is the indexed source sentences;
        batch['target_list_batch'] is the indexed target sentences;
        '''

        # 获取batch的source_batch、target_batch
        source_data = batch['source_batch']
        # [['从', '来', '没', '有', '学', '过', '汉', '语', '。'],[],...]

        target_data = batch['target_batch']
        # [[['之', '前', '从', '来', '没', '有', '学', '过', '汉', '语', '。']],[],...]

        # 对里面句子进行拼接
        source_data = [''.join(words) for words in source_data]
        # ['从来没有学过汉语。','可是我口语的方面还不好。', ... ]

        target_data = [''.join(words) for words in target_data]
        # ['之前从来没有学过汉语。','可是我的口语方面还不好。', ... ]

        # 对源文本进行编码
        source_encoding = self.tokenizer(
            source_data,
            padding = True,
            max_length = self.max_source_length,
            truncation = True,
            return_tensors = "pt"
        )

        # 获取source的id和attention_mask
        source_ids = source_encoding.input_ids
        source_attention_mask = source_encoding.attention_mask

        # 将 input_ids 和 attention_mask 放入gpu
        if self.device:
            source_ids = source_ids.to(self.device)
            source_attention_mask = source_attention_mask.to(self.device)

        # 对目标文本进行编码
        target_encoding = self.tokenizer(
            target_data,
            padding = True,
            max_length = self.max_target_length,
            truncation = True,
            return_tensors = "pt"
        )

        target_ids = target_encoding.input_ids

        # 将等于pad的id部分置为-100，以便计算损失是忽略
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        # 将 target_ids 放入gpu
        if self.device:
            target_ids = target_ids.to(self.device)

        # Forward pass
        outputs = self.model(input_ids = source_ids,
                             attention_mask = source_attention_mask,
                             labels = target_ids)
        loss = outputs.loss
        logits = outputs.logits
        print('-------------------------------------------------------------------loss:', loss)

        # 经过softmax分类后的id
        predicted_ids = logits.argmax(dim=-1)

        # decode的结果是predicted_ids
        self.decode_result = predicted_ids
        # print('decode_result[0]:',self.decode_result[0])

        # 对decode的结果进行解码，得到文本
        predict_decode = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        print('target_data   :', target_data)
        print('predict_decode:',predict_decode)

        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss}

        # if is_display:
        #     print(self.decode_result[0]); #exit()
        return loss_dic

    def model_test(self, batch, dataloader):
        # 获取batch，[['图', '书', '馆', '的', '对', '面', '有', '学', '生', '会', '馆', '。']]
        source_data = batch['source_batch']
        target_data = batch['target_batch']

        # 将字进行拼接，['图书馆的对面有学生会馆。']
        source_data = [''.join(words) for words in source_data]
        target_data = [''.join(words) for words in target_data]

        # 对源文本进行编码
        source_encoding = self.tokenizer(
            source_data,
            padding=True,
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt"
        )

        # 获取source的id和attention_mask
        source_ids = source_encoding.input_ids
        source_attention_mask = source_encoding.attention_mask

        # 将 input_ids 和 attention_mask 放入gpu
        if self.device:
            source_ids = source_ids.to(self.device)
            source_attention_mask = source_attention_mask.to(self.device)

        # 对目标文本进行编码
        target_encoding = self.tokenizer(
            target_data,
            padding=True,
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )

        target_ids = target_encoding.input_ids

        # 生成预测文本
        predicted_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            do_sample=False,  # disable sampling to test if batching affects output
        )

        # 对预测结果进行解码
        predict_decode = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        # 按字符进行拆分 ['图', '书', '馆', '的', '对', '面', '有', '学', '生', '会', '馆', '。']
        predict_char_text = [char for text in predict_decode for char in text]

        # 将字符转换成对应的id
        predict_char_ids = self.tokenizer.convert_tokens_to_ids(predict_char_text)

        # 将英文逗号ID=261替换为中文逗号,
        for i in range(len(predict_char_ids)):
            if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids("<unk>"):
                dataloader.replaced_symbols.append(predict_char_text[i])
            if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids(","):
                predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("，")
                dataloader.replaced_symbols.append("，")
            if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids("!"):
                predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("！")
                dataloader.replaced_symbols.append("！")
            if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids(";"):
                predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("；")
                dataloader.replaced_symbols.append("；")
            if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids(":"):
                predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("：")
                dataloader.replaced_symbols.append("：")
            if predict_char_ids[i] == self.tokenizer.convert_tokens_to_ids("?"):
                predict_char_ids[i] = self.tokenizer.convert_tokens_to_ids("？")
                dataloader.replaced_symbols.append("？")

        # print('转换中文逗号后的id：',predict_char_ids)

        result = [predict_char_ids]
        print('-----------------model_test的result:',result)

        return result,target_ids
