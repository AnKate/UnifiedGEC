# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/10 15:22
# @File: ttt.py

import torch
from torch import nn
import torch.nn.functional as F

from gectoolkit.module.Layer.layers import gelu, LayerNorm
from gectoolkit.module.transformer import TransformerLayer, Embedding, LearnedPositionalEmbedding
from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.model.TtT.crf_layer import DynamicCRF


def tensor_ready(batch, tokenizer):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]

    text_list_batch = []
    tag_list_batch = []
    for idx, text_list in enumerate(source_list_batch):
        text_list = tokenizer.convert_tokens_to_ids([SpecialTokens.CLS_TOKEN]) + text_list
        tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.CLS_TOKEN])
        tag_list += target_list_batch[idx]
        # tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
        '''
        if len(tag_list) > len(text_list):
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.MASK_TOKEN]) * (len(tag_list) - len(text_list))
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
        elif len(tag_list) < len(text_list):
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN]) * (len(text_list) - len(tag_list))
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
        else:
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
        assert len(text_list) == len(tag_list)
        '''
        text_list_batch.append(text_list)
        tag_list_batch.append(tag_list)

    batch["ready_source_batch"] = text_list_batch
    batch["ready_target_batch"] = tag_list_batch

    return batch


class TtT(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.bert_model = BERTLM(config, dataset)
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embed_dim"]
        self.gamma = config["gamma"]
        self.num_class = num_class = len(dataset.vocab)
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)

        self.tokenizer = dataset
        self.fc = nn.Linear(self.embedding_size, self.num_class)
        self.CRF_layer = DynamicCRF(num_class)
        self.loss_type = config["loss_type"]
        self.bert_vocab = dataset

        if self.device:
            self.bert_model = self.bert_model.cuda(self.device)

    def nll_loss(self, y_pred, y, y_mask, avg=True):
        cost = -torch.log(torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1)))
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost)

    def fc_nll_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
        if gamma is None:
            gamma = 2
        p = torch.gather(y_pred, 2, y.view(y.size(0), y.size(1), 1))
        g = (1 - torch.clamp(p, min=0.01, max=0.99)) ** gamma
        # g = (1 - p) ** gamma
        cost = -g * torch.log(p + 1e-8)
        cost = cost.view(y.shape)
        y_mask = y_mask.view(y.shape)
        if avg:
            cost = torch.sum(cost * y_mask, 0) / torch.sum(y_mask, 0)
        else:
            cost = torch.sum(cost * y_mask, 0)
        cost = cost.view((y.size(1), -1))
        return torch.mean(cost), g.view(y.shape)

    # def forward(self, text_data, in_mask_matrix, in_tag_matrix, fine_tune=False, gamma=None):
    def forward(self, batch, dataloader):
        batch = tensor_ready(batch, self.vocab)
        text_data = dataloader.truncate_tensor(batch["ready_source_batch"])
        in_tag_matrix = dataloader.truncate_tensor(batch["ready_target_batch"])
        # print('in_tag_matrix', in_tag_matrix.size())
        # in_tag_matrix = batchify(in_tag_matrix, self.vocab)

        in_mask_matrix = 1 - torch.eq(in_tag_matrix, self.padding_id).to(torch.int)
        # print('in_mask_matrix', in_mask_matrix)
        current_batch_size, seq_len = text_data.size()

        # print('in_tag_matrix', in_tag_matrix[:3, :])
        # print('in_mask_matrix', in_mask_matrix[:3, :3])
        mask_matrix = in_mask_matrix.to(torch.bool).t_()  # .contiguous()# .cuda(self.device)
        tag_matrix = torch.LongTensor(in_tag_matrix).t_()  # .contiguous()

        if self.device:
            mask_matrix = mask_matrix.cuda(self.device)
            tag_matrix = tag_matrix.cuda(self.device)  # size = [seq_len, batch_size]

        assert mask_matrix.size() == tag_matrix.size()
        assert mask_matrix.size() == torch.Size([seq_len, current_batch_size])

        # input text_data.size() = [batch_size, seq_len]
        data = text_data.t_()  # dataloader.truncate_tensor(text_data) # data.size() == [seq_len, batch_size]

        if self.device:
            data = data.cuda(self.device)

        # print('self.bert_model', data[0, :])
        # print('in_mask_matrix', mask_matrix[0, :])
        sequence_representation = self.bert_model.work(data)[
            0]  # .cuda(self.device) # [seq_len, batch_size, embedding_size]
        # print('sequence_representation', sequence_representation)
        # dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)
        sequence_representation = sequence_representation.view(current_batch_size * seq_len, self.embedding_size)
        sequence_emissions = self.fc(sequence_representation)
        # print('sequence_emissions', sequence_emissions)
        # decode_result = self.fc(sequence_representation)
        sequence_emissions = sequence_emissions.view(seq_len, current_batch_size, self.num_class)

        # bert finetune loss
        probs = torch.softmax(sequence_emissions, -1)
        # print('probs', probs); exit()
        if "FT" in self.loss_type:
            loss_ft_fc, g = self.fc_nll_loss(probs, tag_matrix, mask_matrix, gamma=self.gamma)
        else:
            loss_ft = self.nll_loss(probs, tag_matrix, mask_matrix)

        sequence_emissions = sequence_emissions.transpose(0, 1)
        tag_matrix = tag_matrix.transpose(0, 1)
        mask_matrix = mask_matrix.transpose(0, 1)

        if "FC" in self.loss_type:
            # loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask = mask_matrix, reduction='token_mean', g=g.transpose(0, 1), gamma=gamma)
            loss_crf_fc = -self.CRF_layer(sequence_emissions, tag_matrix, mask=mask_matrix, reduction='token_mean',
                                          g=None, gamma=self.gamma)
        else:
            loss_crf = -self.CRF_layer(sequence_emissions, tag_matrix, mask=mask_matrix, reduction='token_mean')
        # print('sequence_emissions', sequence_emissions.size())
        decode_result = sequence_emissions.max(-1)[1]
        # print('before decode_result', decode_result[0, :]); #exit()
        decode_result = self.CRF_layer.decode(sequence_emissions, mask=mask_matrix)
        # print('decode_result', decode_result[0].size(), decode_result[1].size())
        self.decode_scores, self.decode_result = decode_result
        self.decode_result = self.decode_result.tolist()
        # print('decode_result', data[0, :]) #, sequence_emissions[0, :3, :20], self.decode_result, tag_matrix[0], mask_matrix[0]); #exit()

        if self.loss_type == 'CRF':
            loss = loss_crf
            loss_dic = {"decode_result": self.decode_result,
                        "loss": loss,
                        "loss_ft_fc": loss_ft_fc.item(),
                        "loss_fc": 0.0}
        elif self.loss_type == 'FC':
            loss = loss_ft_fc
            loss_dic = {"decode_result": self.decode_result,
                        "loss": loss,
                        "loss_ft_fc": loss_ft_fc.item(),
                        "loss_fc": 0.0}
        elif self.loss_type == 'FT_CRF':
            loss = loss_ft + loss_crf
            loss_dic = {"decode_result": self.decode_result,
                        "loss": loss,
                        "loss_crf": loss_crf.item(),
                        "loss_ft": loss_ft.item()}
        elif self.loss_type == 'FC_FT_CRF':
            loss = loss_ft_fc + loss_crf_fc
            loss_dic = {"decode_result": self.decode_result,
                        "loss": loss,
                        "loss_crf": loss_crf_fc.item(),
                        "loss_ft": loss_ft_fc.item()}
        elif self.loss_type == 'FC_CRF':
            loss = loss_crf_fc
            loss_dic = {"decode_result": self.decode_result,
                        "loss": loss,
                        "loss_crf": loss_crf_fc.item(),
                        "loss_ft": 0.0}
        else:
            loss_dic = {"decode_result": self.decode_result,
                        "loss": 0.0,
                        "loss_crf": 0.0,
                        "loss_ft": 0.0}
        # print('loss_dic', loss_dic['loss'])
        return loss_dic

    def model_test(self, batch, dataloader):
        batch = tensor_ready(batch, self.vocab)
        text_data = dataloader.truncate_tensor(batch["ready_source_batch"])
        in_tag_matrix = dataloader.truncate_tensor(batch["ready_target_batch"])
        # print('in_tag_matrix', in_tag_matrix)
        # in_tag_matrix = batchify(in_tag_matrix, self.vocab)

        in_mask_matrix = 1 - torch.eq(in_tag_matrix, self.padding_id).to(torch.int)
        current_batch_size, seq_len = text_data.size()

        # print('in_tag_matrix', in_tag_matrix[:3, :])
        # print('in_mask_matrix', in_mask_matrix[:3, :3])
        mask_matrix = in_mask_matrix.to(torch.bool).t_()  # .contiguous()# .cuda(self.device)
        tag_matrix = torch.LongTensor(in_tag_matrix).t_()  # .contiguous()

        if self.device:
            mask_matrix = mask_matrix.cuda(self.device)
            tag_matrix = tag_matrix.cuda(self.device)  # size = [seq_len, batch_size]

        assert mask_matrix.size() == tag_matrix.size()
        assert mask_matrix.size() == torch.Size([seq_len, current_batch_size])

        # input text_data.size() = [batch_size, seq_len]
        data = text_data.t_()  # dataloader.truncate_tensor(text_data) # data.size() == [seq_len, batch_size]

        if self.device:
            data = data.cuda(self.device)

        # print('self.bert_model', data[0, :]);
        # print('tag_matrix', tag_matrix[0, :])
        sequence_representation = self.bert_model.work(data)[
            0]  # .cuda(self.device) # [seq_len, batch_size, embedding_size]
        # dropout
        sequence_representation = F.dropout(sequence_representation, p=self.dropout, training=self.training)
        sequence_representation = sequence_representation.view(current_batch_size * seq_len, self.embedding_size)
        sequence_emissions = self.fc(sequence_representation)
        # decode_result = self.fc(sequence_representation)
        sequence_emissions = sequence_emissions.view(seq_len, current_batch_size, self.num_class)

        # bert finetune loss
        probs = torch.softmax(sequence_emissions, -1)

        sequence_emissions = sequence_emissions.transpose(0, 1)
        tag_matrix = tag_matrix.transpose(0, 1)
        mask_matrix = mask_matrix.transpose(0, 1)

        # print('decode_result', decode_result[:3, :3]); exit()
        decode_result = self.CRF_layer.decode(sequence_emissions, mask=mask_matrix)
        # print('decode_result', decode_result[0].size(), decode_result[1].size())
        self.decode_scores, self.decode_result = decode_result
        self.decode_result = self.decode_result.tolist()
        self.decode_result = post_process_decode_result(self.decode_result, batch["ready_source_batch"])
        # print('decode_result', data[:, 0], sequence_emissions[0, :3, :20], self.decode_result, tag_matrix[0], mask_matrix[0]); #exit()

        return self.decode_result, tag_matrix


def post_process_decode_result(sentences, source):
    new_sentences = []
    for sent_idx, sent in enumerate(sentences):
        sour = source[sent_idx]
        sent = sent[1:]
        sent = sent[:len(sour) - 1]

        new_sentences += [sent]
    return new_sentences


class BERTLM(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.embed_dim = embed_dim = config["embed_dim"]
        self.ff_embed_dim = ff_embed_dim = config["ff_embed_dim"]
        self.num_heads = num_heads = config["num_heads"]
        self.dropout = dropout = config["dropout"]
        self.device = device = config["device"]
        self.approx = approx = config["approx"]
        self.layers = layers = config["layers"]
        self.fine_tune = config["fine_tune"]
        self.gamma = config["gamma"]
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)
        dataset_size = len(dataset.vocab)

        self.tok_embed = Embedding(dataset_size, embed_dim, self.padding_id)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=device)
        self.seg_embed = Embedding(2, embed_dim, None)

        self.out_proj_bias = nn.Parameter(torch.Tensor(dataset_size))

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))

        self.emb_layer_norm = LayerNorm(embed_dim)
        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_layer_norm = LayerNorm(embed_dim)
        self.one_more_nxt_snt = nn.Linear(embed_dim, embed_dim)
        self.nxt_snt_pred = nn.Linear(embed_dim, 1)

        if approx == "none":
            self.approx = None
        elif approx == "adaptive":
            self.approx = nn.AdaptiveLogSoftmaxWithLoss(self.embed_dim, self.vocab.size, [10000, 20000, 200000])
        else:
            raise NotImplementedError("%s has not been implemented" % approx)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.nxt_snt_pred.bias, 0.)
        nn.init.constant_(self.one_more.bias, 0.)
        nn.init.constant_(self.one_more_nxt_snt.bias, 0.)
        nn.init.normal_(self.nxt_snt_pred.weight, std=0.02)
        nn.init.normal_(self.one_more.weight, std=0.02)
        nn.init.normal_(self.one_more_nxt_snt.weight, std=0.02)

    def work(self, inp, seg=None, layers=None):
        # inp (torch.Tensor): token ids, size: (seq_len x bsz)
        # seg (torch.Tensor): segment ids, size: (seq_len x bsz), default is None, which means all zeros.
        # layers (list or None): list of layer ids or None: the list of the layers you want to return, default is None, which means only the last layer will be returned.
        # return x (torch.Tensor): token representation, size: (seq_len x bsz x embed_dim)) if layers is None else (len(layers) x seq_len x bsz x embed_dim)
        # return z (torch.Tensor): sequence representation, size: (bsz x embed_dim) if layers is None else (len(layers) x bsz x embed_dim)
        if layers is not None:
            tot_layers = len(self.layers)
            for x in layers:
                if not (-tot_layers <= x < tot_layers):
                    raise ValueError('layer %d out of range ' % x)
            layers = [(x + tot_layers if x < 0 else x) for x in layers]
            max_layer_id = max(layers)

        seq_len, bsz = inp.size()
        if seg is None:
            seg = torch.zeros_like(inp)

        # print(inp, seg)
        # print(self.tok_embed.weight)
        x = self.tok_embed(inp) + self.seg_embed(seg) + self.pos_embed(inp)
        # print('before dropout', x)
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(inp, self.padding_id)
        if not padding_mask.any():
            padding_mask = None
        # print('begin', x)

        xs = []
        for layer_id, layer in enumerate(self.layers):
            x, _, _ = layer(x, self_padding_mask=padding_mask)
            # print(layer_id, x)
            xs.append(x)
            if layers is not None and layer_id >= max_layer_id:
                break

        if layers is not None:
            x = torch.stack([xs[i] for i in layers])
            z = torch.tanh(self.one_more_nxt_snt(x[:, 0, :, :]))
        else:
            z = torch.tanh(self.one_more_nxt_snt(x[0]))
        return x, z

    # def forward(self, truth, inp, seg, msk, nxt_snt_flag):
    def forward(self, batch, tokenizer):
        truth, seg = batch['text_list'], batch['tag_list_matrix']
        # print('seg', seg)
        msk = self.fine_tune
        nxt_snt_flag = self.gamma

        seq_len, bsz = inp.size()
        x = self.seg_embed(seg) + self.tok_embed(inp) + self.pos_embed(inp)  #
        x = self.emb_layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = torch.eq(truth, self.padding_id)
        if not padding_mask.any():
            padding_mask = None
        for layer in self.layers:
            x, _, _ = layer(x, self_padding_mask=padding_mask)

        masked_x = x.masked_select(msk.unsqueeze(-1))
        masked_x = masked_x.view(-1, self.embed_dim)
        gold = truth.masked_select(msk)

        y = self.one_more_layer_norm(gelu(self.one_more(masked_x)))
        out_proj_weight = self.tok_embed.weight

        if self.approx is None:
            log_probs = torch.log_softmax(F.linear(y, out_proj_weight, self.out_proj_bias), -1)
        else:
            log_probs = self.approx.log_prob(y)

        loss = F.nll_loss(log_probs, gold, reduction='mean')

        z = torch.tanh(self.one_more_nxt_snt(x[0]))
        nxt_snt_pred = torch.sigmoid(self.nxt_snt_pred(z).squeeze(1))
        nxt_snt_acc = torch.eq(torch.gt(nxt_snt_pred, 0.5), nxt_snt_flag).float().sum().item()
        nxt_snt_loss = F.binary_cross_entropy(nxt_snt_pred, nxt_snt_flag.float(), reduction='mean')

        tot_loss = loss + nxt_snt_loss

        _, pred = log_probs.max(-1)
        tot_tokens = msk.float().sum().item()
        acc = torch.eq(pred, gold).float().sum().item()

        return (pred, gold), tot_loss, acc, tot_tokens, nxt_snt_acc, bsz
