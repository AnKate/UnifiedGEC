from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config
import numpy as np
import torch
from transformers import MT5Tokenizer
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from torch import nn


class T5(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.device = config["device"]

        # 获取到模型的路径
        self.model_path = config["pretrained_model_path"]

        # self.dataloader.pretrained_tokenzier
        self.tokenizer = dataset

        # 导入模型
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_path)
        self.pad_token_id = self.tokenizer.pad_token_id  # pad id= 0
        self.test_id = 0

    def forward(self, batch, dataloader, is_display):
        """
        parameters in batch.
        batch['source_batch'] is a list with the source text sentences;
        batch['target_batch'] is a list with the target text sentences;
        batch['source_list_batch'] is the indexed source sentences;
        batch['target_list_batch'] is the indexed target sentences;
        """
        language = dataloader.dataset.language_name      # 中文数据集/英文数据集
        source_list_batch = batch["source_list_batch"]
        target_list_batch = batch["target_list_batch"]

        # 获取每个batch中所有句子的最大长度
        source_max_len = np.max([len(sent) for sent in source_list_batch])
        target_max_len = np.max([len(sent) for sent in target_list_batch])
        # print('source_max_len:', source_max_len)
        # print('target_max_len:', target_max_len)

        # 获取batch的source_batch、target_batch
        # 中文source： [['从', '来', '没', '有', '学', '过', '汉', '语', '。'],[],...]
        # 英文source： [['▁De', 'ar', '▁Mr', '▁...', '▁', ','],....]
        source_data = batch['source_batch']
        target_data = batch['target_batch']

        # 对句子进行拼接：
        # 中文source：['从来没有学过汉语。','可是我口语的方面还不好。', ... ]
        # 英文source：['Firstly I thought','..', ...]
        if language == 'zh':
            source_data = [''.join(words) for words in source_data]
            target_data = [''.join(words) for words in target_data]
        else:
            source_data = self.tokenizer.batch_decode(source_list_batch)
            target_data = self.tokenizer.batch_decode(target_list_batch)

        # source进行encode
        source_encoding = self.tokenizer.batch_encode_plus(source_data,
                                                           pad_to_max_length=True,
                                                           max_length=source_max_len,
                                                           truncation=True,
                                                           return_tensors="pt")

        # 获取source的id和attention_mask
        source_ids = source_encoding.input_ids                  # torch.Size([batch_size, source_max_len])
        source_attention_mask = source_encoding.attention_mask  # torch.Size([batch_size, source_max_len])

        # 将 input_ids 和 attention_mask 放入gpu
        if self.device:
            source_ids = source_ids.to(self.device)
            source_attention_mask = source_attention_mask.to(self.device)

        # 对目标文本进行编码
        target_encoding = self.tokenizer.batch_encode_plus(target_data,
                                                           pad_to_max_length=True,
                                                           max_length=target_max_len,
                                                           truncation=True,
                                                           return_tensors="pt")

        # 获取target_ids的大小torch.Size([batch_size, source_max_len])
        target_ids = target_encoding.input_ids

        # 将等于pad的id部分置为-100，以便计算损失时忽略
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        # 将 target_ids 放入gpu
        if self.device:
            target_ids = target_ids.to(self.device)

        # Forward pass
        outputs = self.model(input_ids = source_ids,
                             attention_mask = source_attention_mask,
                             labels = target_ids)
        loss = outputs.loss
        logits = outputs.logits  # torch.Size([batch_size, max_length, vocab_size])

        # 经过softmax分类后的id
        predicted_ids = logits.argmax(dim=-1)

        # 对decode的结果进行解码，得到文本
        predict_decode = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
        # print('target_data   :',target_data[0])
        # print('predict_decode:',predict_decode[0])
        # print('-----loss:', loss)

        loss_dic = {"decode_result": predicted_ids,
                    "loss": loss}

        if is_display:
            print(self.decode_result[0]); #exit()
        return loss_dic


    def model_test(self, batch, dataloader, is_display=False):
        with torch.no_grad():
            language = dataloader.dataset.language_name

            source_list_batch = batch["source_list_batch"]
            target_list_batch = batch["target_list_batch"]

            source_max_len = np.max([len(sent) for sent in source_list_batch])
            target_max_len = np.max([len(sent) for sent in target_list_batch])

            # 获取batch，[['图', '书', '馆', '的', '对', '面', '有', '学', '生', '会', '馆', '。']]
            source_data = batch['source_batch']
            target_data = batch['target_batch']

            if language == 'zh':
                source_data = [''.join(words) for words in source_data]
                target_data = [''.join(words) for words in target_data]
            else:
                source_data = self.tokenizer.batch_decode(source_list_batch)
                target_data = self.tokenizer.batch_decode(target_list_batch)

            # 源文本source进行encode
            source_encoding = self.tokenizer.batch_encode_plus(source_data,
                                                               pad_to_max_length=True,
                                                               max_length=source_max_len,
                                                               truncation=True,
                                                               return_tensors="pt")

            # 获取source的id和attention_mask
            source_ids = source_encoding.input_ids                  # torch.Size([1, source_max_len])
            source_attention_mask = source_encoding.attention_mask  # torch.Size([1, source_max_len])

            # 将 input_ids 和 attention_mask 放入gpu
            if self.device:
                source_ids = source_ids.to(self.device)
                # source_attention_mask = source_attention_mask.to(self.device)

            # 对目标文本target进行encode
            target_encoding = self.tokenizer.batch_encode_plus(target_data,
                                                               pad_to_max_length=True,
                                                               max_length=target_max_len,
                                                               truncation=True,
                                                               return_tensors="pt")
            target_ids = target_encoding.input_ids

            if language == 'zh':
                predicted_ids = self.model.generate(
                    input_ids=source_ids,
                    max_length=target_max_len ,  # disable sampling to test if batching affects output
                )
            else:
                predicted_ids = self.model.generate(
                    input_ids=source_ids,
                    max_length=target_max_len+2,  #  因为 ValueError: Input length of decoder_input_ids is 1, but `max_length` is set to 1.
                )

            # 对预测结果进行解码
            target_decode = self.tokenizer.batch_decode(target_ids, skip_special_tokens=True)
            predict_decode = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

            # 打印target和predict的长度
            # if language == 'zh':
            #     print('len(target_decode:', len(target_decode[0]))
            #     print('len(predict_decode:', len(predict_decode[0]))
            # else:
            #     print('len(target_decode:', len(target_decode[0].split()))
            #     print('len(predict_decode:', len(predict_decode[0].split()))

            # 打印decode的结果
            print('\ntarget_decode :',target_decode)
            print('predict_decode:',predict_decode)

            # mt5large存在中英文符号转换问题,将英文符号转成中文
            if language == 'zh':
                # 按字符进行拆分    ['图', '书', '馆', '的', '对', '面', '有', '学', '生', '会', '馆', '。']
                predict_char_text = [char for text in predict_decode for char in text]
                # 将字符转换成对应的id
                predict_char_ids = self.tokenizer.convert_tokens_to_ids(predict_char_text)

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
                    result = [predict_char_ids]
            else:
                result = predicted_ids.tolist()

        return result,target_ids
