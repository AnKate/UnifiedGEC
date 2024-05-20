# -*- encoding: utf-8 -*-
# @Author: Wenbiao Tao
# @Time: 2023/06/16 22:49
# @File: transformer.py

import numpy as np
import torch
import torch.nn.functional as F
import copy

from torch import nn
from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.module.transformer import TransformerLayer, Embedding, SelfAttentionMask, LearnedPositionalEmbedding


def tensor_ready(batch, tokenizer, is_train=False):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    source_max_len = np.max([len(sent) for sent in source_list_batch])
    target_max_len = np.max([len(sent) for sent in target_list_batch]) + 2

    text_list_batch = []
    tag_list_batch = []
    for idx, text_list in enumerate(source_list_batch):

        text_list = text_list
        tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])
        if is_train:
            tag_list += target_list_batch[idx]
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN]) * (
                        target_max_len - len(target_list_batch[idx]))
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN]) * (
                        source_max_len - len(source_list_batch[idx]))

        text_list_batch.append(text_list)
        tag_list_batch.append(tag_list)

    batch["ready_source_batch"] = text_list_batch
    batch["ready_target_batch"] = tag_list_batch

    return batch


class Transformer(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.embedding_size = config["embed_dim"]
        self.ff_embedding_size = config["ff_embed_dim"]
        self.num_heads = config["num_heads"]
        self.hidden_dim = config["hidden_dim"]
        self.gamma = config["gamma"]
        self.max_output_len = config["max_output_len"]
        self.layer_num = config["layers"]
        self.num_class = len(dataset.vocab)
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)

        self.tokenizer = dataset
        dataset_size = len(dataset.vocab)

        # input embedding, 对输入序列进行嵌入
        self.in_tok_embed = Embedding(dataset_size, self.embedding_size, self.padding_id)
        # positional encoding, 得到序列的位置编码
        self.positional_embed = LearnedPositionalEmbedding(self.embedding_size, device=self.device)

        # self attention mask, 得到的self mask将用于DecodeLayer
        self.self_attention_mask = SelfAttentionMask(device=self.device)
        # 由N=6个EncodeLayer堆叠成的LayerStack, 构成Transformer的encoder部分
        self.encoder_stack = nn.ModuleList(
            [TransformerLayer(self.embedding_size, self.ff_embedding_size, self.num_heads, self.dropout)
             for _ in range(self.layer_num)])
        # 由N=6个DecodeLayer堆叠成的LayerStack, 构成Transformer的decoder部分
        self.decoder_stack = nn.ModuleList(
            [TransformerLayer(self.embedding_size, self.ff_embedding_size, self.num_heads, self.dropout,
                              with_external=True) for _ in range(self.layer_num)])

        # 对decoder输出结果的后处理, Linear+Softmax得到概率分布
        self.out_tok_embed = nn.Linear(self.embedding_size, dataset_size)
        self.out_tok_embed.weight = copy.deepcopy(self.in_tok_embed.weight)

    def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
        # focal loss
        if gamma is None:
            gamma = 2
        p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1).long())
        g = (1 - torch.clamp(p, min=0.01, max=0.99)) ** gamma
        cost = -g * torch.log(p + 1e-8)
        cost = cost.view(y.shape)   # [batch_size, seq_len]
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 1) / torch.sum(y_mask, 1)
        else:
            cost = torch.sum(cost * y_mask, 1)
        cost = cost.view((y.size(0), -1))
        return torch.mean(cost), g.view(y.shape)

    def forward(self, batch, dataloader):
        """
        parameters in batch.
        batch['source_batch'] is a list with the source text sentences;
        batch['target_batch'] is a list with the target text sentences;
        batch['source_list_batch'] is the indexed source sentences;
        batch['target_list_batch'] is the indexed target sentences;
        """

        # convert indexed sentences into the required format for the model training
        # 这一步会将source和target的长度变成相同的seq_len
        batch = tensor_ready(batch, self.vocab, is_train=True)

        # truncate indexed sentences into a batch where the sentence lengths are same in the same batch
        source_data = dataloader.truncate_tensor(batch["ready_source_batch"])  # [batch_size, seq_len]
        target_data = dataloader.truncate_tensor(batch["ready_target_batch"])  # [batch_size, seq_len]

        source_data_for_position = copy.deepcopy(source_data)
        target_data_for_position = copy.deepcopy(target_data)

        # put tensor into gpu if there is one
        if self.device:
            source_data = source_data.cuda(self.device)
            target_data = target_data.cuda(self.device)

        # generate mask tensor to mask the padding position to exclude them during loss calculation
        in_source_mask_matrix = torch.eq(source_data, self.padding_id).to(torch.int)
        source_mask_matrix = in_source_mask_matrix.to(torch.bool)  # [batch_size, seq_len]

        in_target_mask_matrix = torch.eq(target_data, self.padding_id).to(torch.int)
        target_mask_matrix = in_target_mask_matrix.to(torch.bool)  # [batch_size, seq_len]

        in_mask_matrix = 1 - torch.eq(target_data, self.padding_id).to(torch.int)
        mask_matrix = in_mask_matrix.to(torch.bool)

        # convert indexed tokens into embeddings
        # [batch_size, seq_len, embedding_dim]
        source_seq_rep = self.in_tok_embed(source_data)
        target_seq_rep = self.in_tok_embed(target_data)
        batch_size, seq_len, embedding_size = target_seq_rep.size()

        # convert embeddings into position embeddings
        source_data_for_position = source_data_for_position.t()
        target_data_for_position = target_data_for_position.t()
        if self.device:
            source_data_for_position = source_data_for_position.cuda(self.device)
            target_data_for_position = target_data_for_position.cuda(self.device)

        # [batch_size, seq_len, embedding_dim]
        source_seq_position_rep = self.positional_embed(source_data_for_position).transpose(0, 1)
        target_seq_position_rep = self.positional_embed(target_data_for_position).transpose(0, 1)

        # [batch_size, seq_len, embedding_dim]
        source_seq_rep = source_seq_rep + source_seq_position_rep
        target_seq_rep = target_seq_rep + target_seq_position_rep

        encoded_rep = source_seq_rep.transpose(0, 1)    # [src_len, batch_size, embedding_dim]

        for encode_layer in self.encoder_stack:
            encoded_rep, encoded_self_attn, _ = encode_layer(encoded_rep,
                                                             self_padding_mask=source_mask_matrix.transpose(0, 1),
                                                             need_weights=True)

        attn_mask = self.self_attention_mask(seq_len).to(torch.bool)    # [tgt_len, tgt_len]
        if self.device:
            attn_mask = attn_mask.cuda(self.device)
        decoded_rep = target_seq_rep.transpose(0, 1)  # [seq_len, batch_size, embedding_dim]

        for decode_layer in self.decoder_stack:
            decoded_rep, decoded_self_attn, decoded_external_attn = decode_layer(decoded_rep,
                                                                                 self_padding_mask=target_mask_matrix.transpose(
                                                                                     0, 1),
                                                                                 self_attn_mask=attn_mask,
                                                                                 external_memories=encoded_rep,
                                                                                 external_padding_mask=source_mask_matrix.transpose(
                                                                                     0, 1),
                                                                                 need_weights=True)

        probs = torch.softmax(self.out_tok_embed(decoded_rep.transpose(0, 1)), -1)
        self.decode_result = probs.max(-1)[1]

        # compute loss
        loss_ft_fc, g = self.fc_nll_loss(probs[:, :-1, :], target_data[:, 1:], mask_matrix[:, 1:], gamma=self.gamma)
        # print(loss_ft_fc)
        loss_dic = {"decode_result": self.decode_result,
                    "loss": loss_ft_fc}

        return loss_dic

    def model_test(self, batch, dataloader):
        # convert indexed sentences into the required format for the model training
        batch = tensor_ready(batch, self.vocab)

        # truncate indexed sentences into a batch where the sentence lengths are same in the same batch
        source_data = dataloader.truncate_tensor(batch["ready_source_batch"])   # [batch_size, src_len]
        target_data = dataloader.truncate_tensor(batch["ready_target_batch"])   # [batch_size, tgt_len(=1)]

        source_data_for_position = copy.deepcopy(source_data)
        target_data_for_position = copy.deepcopy(target_data)

        # put tensor into gpu if there is one
        if self.device:
            source_data = source_data.cuda(self.device)
            target_data = target_data.cuda(self.device)

        # generate mask tensor to mask the padding position to exclude them during loss calculation
        in_source_mask_matrix = torch.eq(source_data, self.padding_id).to(torch.int)
        source_mask_matrix = in_source_mask_matrix.to(torch.bool)   # [batch_size, src_len]

        in_target_mask_matrix = torch.eq(target_data, self.padding_id).to(torch.int)
        target_mask_matrix = in_target_mask_matrix.to(torch.bool)   # [batch_size, tgt_len]

        # convert indexed tokens into embeddings
        source_seq_rep = self.in_tok_embed(source_data)     # [batch_size, src_len, embedding_dim]
        _, src_len, _ = source_seq_rep.size()
        target_seq_rep = self.in_tok_embed(target_data)
        batch_size, tgt_len, embedding_dim = target_seq_rep.size()     # [batch_size, tgt_len(=1), embedding_dim]

        # convert embeddings into position embeddings
        source_data_for_position = source_data_for_position.t()     # [src_len, batch_size, embedding_dim]
        target_data_for_position = target_data_for_position.t()     # [tgt_len(=1), batch_size, embedding_dim]
        if self.device:
            source_data_for_position = source_data_for_position.cuda(self.device)
            target_data_for_position = target_data_for_position.cuda(self.device)
        # [batch_size, src_len, embedding_dim]
        source_seq_position_rep = self.positional_embed(source_data_for_position).transpose(0, 1)
        # [batch_size, tgt_len(=1), embedding_dim]
        target_seq_position_rep = self.positional_embed(target_data_for_position).transpose(0, 1)

        source_seq_rep = source_seq_rep + source_seq_position_rep   # [batch_size, src_len, embedding_dim]
        target_seq_rep = target_seq_rep + target_seq_position_rep   # [batch_size, tgt_len(=1), embedding_dim]

        # encoder
        encoded_rep = source_seq_rep.transpose(0, 1)    # [src_len, batch_size, embedding_dim]
        for encode_layer in self.encoder_stack:
            encoded_rep, encoded_self_attn, _ = encode_layer(encoded_rep,
                                                             self_padding_mask=source_mask_matrix.transpose(0, 1),
                                                             need_weights=True)

        # get self-attention mask
        attn_mask = self.self_attention_mask(tgt_len).to(torch.bool)    # [tgt_len, tgt_len]
        if self.device:
            attn_mask = attn_mask.cuda(self.device)

        self.decode_result = []

        # decoder
        decoded_rep = target_seq_rep.transpose(0, 1)
        for i in range(self.max_output_len):
            # print(target_data)
            for decode_layer in self.decoder_stack:
                decoded_rep, decoded_self_attn, decoded_external_attn = decode_layer(decoded_rep,
                                                                                     self_padding_mask=target_mask_matrix.transpose(0, 1),
                                                                                     self_attn_mask=attn_mask,
                                                                                     external_memories=encoded_rep,
                                                                                     external_padding_mask=source_mask_matrix.transpose(0, 1),
                                                                                     need_weights=True)
            probs = torch.softmax(self.out_tok_embed(decoded_rep.transpose(0, 1)), -1)
            probs_result = probs.max(-1)[1][:, -1].view(-1, 1)  # [batch_size, 1]
            target_data = torch.cat((target_data, probs_result), -1)     # [batch_size, seq_len]

            in_target_mask_matrix = torch.eq(target_data, self.padding_id).to(torch.int)
            target_mask_matrix = in_target_mask_matrix.to(torch.bool)

            target_data_for_position = copy.deepcopy(target_data)
            target_data_for_position = target_data_for_position.t()
            target_seq_position_rep = self.positional_embed(target_data_for_position).transpose(0, 1)

            target_seq_rep = self.in_tok_embed(target_data)
            target_seq_rep = target_seq_rep + target_seq_position_rep
            decoded_rep = target_seq_rep.transpose(0, 1)

            seq_len = target_seq_rep.size()[1]
            attn_mask = self.self_attention_mask(seq_len).to(torch.bool).cuda(self.device)

            for j in range(batch_size):
                if len(self.decode_result) < batch_size:
                    self.decode_result.append([target_data[j][-1]])
                else:
                    self.decode_result[j].append(target_data[j][-1])

        self.decode_result = torch.Tensor(self.decode_result)

        self.decode_result = post_process_decode_result(self.decode_result, self.tokenizer)

        return self.decode_result, target_data


def post_process_decode_result(sentences, tokenizer):
    new_sentences = []
    for sent_idx, sent in enumerate(sentences):
        # print("sent", sent)
        # print("EOS Token", tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]))
        # print("SEP Token", tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN]))
        # exit()
        new_sent = []
        for w in sent:
            if w in tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]):
                break
            new_sent += [w]

        new_sentences += [new_sent]
    return new_sentences
