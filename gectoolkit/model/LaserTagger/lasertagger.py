import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import re
import numpy as np
from transformers import BertTokenizer

from gectoolkit.module.Layer.layers import gelu, LayerNorm
from gectoolkit.module.transformer import TransformerLayer, Embedding, LearnedPositionalEmbedding
from gectoolkit.model.LaserTagger import transformer_decoder
from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.model.LaserTagger.utils import utils
from gectoolkit.model.LaserTagger import tagging_converter
from gectoolkit.model.LaserTagger import tagging
from gectoolkit.model.LaserTagger.curLine_file import curLine


# def split_to_wordpieces(tokens, labels):
#     """Splits tokens (and the labels accordingly) to WordPieces.
#
#     Args:
#       tokens: Tokens to be split.
#       labels: Labels (one per token) to be split.
#
#     Returns:
#       3-tuple with the split tokens, split labels, and the indices of the
#       WordPieces that start a token.
#     """
#     bert_tokens = []  # Original tokens split into wordpieces.
#     bert_labels = []  # Label for each wordpiece.
#     # Index of each wordpiece that starts a new token.
#     token_start_indices = []
#     # print('tokens:',tokens)
#     for i, token in enumerate(tokens):
#         # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
#         token_start_indices.append(len(bert_tokens) + 1)
#         # vocab_file = 'gectoolkit/properties/model/LaserTagger/vocab.txt'
#         vocab_file = 'gectoolkit/properties/model/LaserTagger'
#         # ----------add
#         special_tokens = [SpecialTokens.__dict__[k] for k in SpecialTokens.__dict__ if not re.search('^\_', k)]
#         special_tokens.sort()
#         # print('special_tokens:', special_tokens)
#         pretrained_tokenzier = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
#         pretrained_tokenzier.add_special_tokens({'additional_special_tokens': special_tokens})
#         # print("token:", token)
#         pieces = pretrained_tokenzier.tokenize(token)
#         # pieces = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case).tokenize(token)
#         bert_tokens.extend(pieces)
#         bert_labels.extend([labels[i]] * len(pieces))
#     return bert_tokens, bert_labels, token_start_indices

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def truncate_list(x, max_seq_length):
    """Returns truncated version of x according to the self._max_seq_length."""
    # Save two slots for the first [CLS] token and the last [SEP] token.
    return x[:max_seq_length - 2]


def list2Str(list):
    ret_str = ''
    for i in range(len(list)):
        ret_str += list[i]
    return ret_str


def targetTrans(sources, target, max_seq_length, enable_swap_tag, do_lower_case, label_map_file,
                use_arbitrary_target_ids_for_infeasible_examples, location=None):
    # print("source:", sources.encode('utf-8').decode('utf-8'))
    label_map = utils.read_label_map(label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        enable_swap_tag)
    # print("sources:", sources)
    task = tagging.EditingTask(sources, location=location)
    if target is not None:
        tags = converter.compute_tags(task, target)
        # print("tags:", tags)
        if not tags:  # 不可转化，取决于　use_arbitrary_target_ids_for_infeasible_examples
            if use_arbitrary_target_ids_for_infeasible_examples:
                # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
                # unlikely to be predicted by chance.
                tags = [tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                        for i, _ in enumerate(task.source_tokens)]
            else:
                return None
    else:
        # If target is not provided, we set all target labels to KEEP.
        tags = [tagging.Tag('KEEP') for _ in task.source_tokens]
    labels = [label_map[str(tag)] for tag in tags]

    tokens = task.source_tokens
    # tokens, labels, token_start_indices = split_to_wordpieces(task.source_tokens, labels)  # wordpiece： tag是以word为单位的，组成word的piece的标注与这个word相同

    if len(tokens) > max_seq_length - 2:
        print(curLine(), "%d tokens is to long," % len(task.source_tokens), "truncate task.source_tokens:",
              task.source_tokens)
        #  截断到self._max_seq_length - 2
        tokens = truncate_list(tokens, max_seq_length)
        labels = truncate_list(labels, max_seq_length)

    input_tokens = ['[CLS]'] + tokens + ['[SEP]']
    labels_mask = [0] + [1] * len(labels) + [0]
    labels = [0] + labels + [0]

    vocab_file = 'gectoolkit/properties/model/LaserTagger'
    special_tokens = [SpecialTokens.__dict__[k] for k in SpecialTokens.__dict__ if not re.search('^\_', k)]
    special_tokens.sort()
    # print('special_tokens:', special_tokens)
    pretrained_tokenzier = AutoTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)
    pretrained_tokenzier.add_special_tokens({'additional_special_tokens': special_tokens})
    tokenizer = pretrained_tokenzier
    # tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    # print("labels:", len(labels))
    return input_ids, labels, segment_ids, input_mask, labels_mask


def tensor_ready(batch, tokenizer, max_seq_length, enable_swap_tag, do_lower_case,
                 use_arbitrary_target_ids_for_infeasible_examples, label_map_file):
    source_batch = batch["source_batch"]
    target_batch = batch["target_batch"]

    input_ids_batch, labels_batch, segment_ids_batch, input_mask_batch, labels_mask_batch = [], [], [], [], []
    for idx, text in enumerate(source_batch):
        tag = target_batch[idx]
        # print("tag:", tag)
        input_ids, labels, segment_ids, input_mask, labels_mask = targetTrans(list2Str(text), list2Str(tag),
                                                                              max_seq_length, enable_swap_tag,
                                                                              do_lower_case, label_map_file,
                                                                              use_arbitrary_target_ids_for_infeasible_examples,
                                                                              location=None)
        # print("----------input--label:", len(input_ids), len(labels))

        input_ids_batch.append(input_ids)
        labels_batch.append(labels)
        segment_ids_batch.append(segment_ids)
        input_mask_batch.append(input_mask)
        labels_mask_batch.append(labels_mask)
    # batch["ready_source_batch"] = input_ids_batch
    batch["ready_labels_batch"] = labels_batch  # 对应tag_id
    batch["ready_input_mask_batch"] = input_mask_batch
    batch["ready_segment_ids_batch"] = segment_ids_batch
    batch["ready_labels_mask_batch"] = labels_mask_batch

    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    tag_list_batch, text_list_batch = [], []
    for idx, text_list in enumerate(source_list_batch):
        # 截断
        tag_list = target_list_batch[idx]
        if len(text_list) > max_seq_length - 2:
            text_list = truncate_list(text_list, max_seq_length)
            tag_list = truncate_list(target_list_batch[idx], max_seq_length)

        text_list = tokenizer.convert_tokens_to_ids(
            [SpecialTokens.CLS_TOKEN]) + text_list + tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
        tag_list = tokenizer.convert_tokens_to_ids(
            [SpecialTokens.CLS_TOKEN]) + tag_list + tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])
        # tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.CLS_TOKEN]) + target_list_batch[
        #     idx] + tokenizer.convert_tokens_to_ids([SpecialTokens.SEP_TOKEN])

        text_list_batch.append(text_list)
        # print("-----text_list:", len(text_list))
        tag_list_batch.append(tag_list)

    batch["ready_source_batch"] = text_list_batch  # 对应bert词表
    batch["ready_target_batch"] = tag_list_batch
    return batch


class LaserTagger(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        """
        Args:
          use_t2t_decoder: Whether to use the Transformer decoder (i.e.
            LaserTagger_AR). If False, the remaining args do not affect anything and
            can be set to default values.
          decoder_num_hidden_layers: Number of hidden decoder layers.
          decoder_hidden_size: Decoder hidden size.
          decoder_num_attention_heads: Number of decoder attention heads.
          decoder_filter_size: Decoder filter size.
          use_full_attention: Whether to use full encoder-decoder attention.
        """
        self.tokenizer = dataset
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.use_t2t_decoder = config["use_t2t_decoder"]
        self.decoder_num_hidden_layers = config["decoder_num_hidden_layers"]
        self.decoder_hidden_size = config["decoder_hidden_size"]
        self.decoder_num_attention_heads = config["decoder_num_attention_heads"]
        self.decoder_filter_size = config["decoder_filter_size"]
        self.use_full_attention = config["use_full_attention"]
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.max_seq_length = config["max_seq_length"]
        self.vocab_size = config["vocab_size"]
        self.use_one_hot_embeddings = config["use_one_hot_embeddings"]
        self.do_lower_case = config["do_lower_case"]
        self.enable_swap_tag = config["enable_swap_tag"]
        self.use_arbitrary_target_ids_for_infeasible_examples = config[
            "use_arbitrary_target_ids_for_infeasible_examples"]
        self.label_map_file = config["label_map_file"]

        self.bert_model = BERTLM(config, dataset)
        self.bert_vocab = dataset
        if self.device:
            self.bert_model = self.bert_model.cuda(self.device)

        self.num_tags = len(utils.read_label_map(self.label_map_file))  # 1003
        if self.use_t2t_decoder:
            output_vocab_size = self.num_tags + 2  # Account for the begin and end tokens used by Transformer.
            self.vocab_size = output_vocab_size
            config["vocab_size"] = output_vocab_size
            self.decoder = transformer_decoder.TransformerDecoder(config, True)
            # self.decoder = TransformerDecoder(config, output_vocab_size, self.hidden_size, True)
            # self.decoder = TransformerDecoder(config, self.use_one_hot_embeddings, output_vocab_size)
        else:
            self.classifier = nn.Linear(self.hidden_size, self.num_tags)
        # self.classifier = nn.Linear(self.hidden_size, self.num_tags)

    def forward(self, batch, dataloader):
        batch = tensor_ready(batch, self.tokenizer, self.max_seq_length, self.enable_swap_tag, self.do_lower_case,
                             self.use_arbitrary_target_ids_for_infeasible_examples, self.label_map_file)
        src_tokens = dataloader.truncate_tensor(batch["ready_source_batch"])  # [batch_size, seq_length]
        tgt_tokens = dataloader.truncate_tensor(batch["ready_labels_batch"])
        tgt_tokens = np.array(tgt_tokens)
        tmp_idx = (tgt_tokens == 16543)
        tgt_tokens[tmp_idx] = 0
        tgt_tokens = torch.from_numpy(tgt_tokens)

        input_mask = dataloader.truncate_tensor(batch["ready_input_mask_batch"])
        labels_mask = dataloader.truncate_tensor(batch["ready_labels_mask_batch"])

        if self.device:
            src_tokens = src_tokens.cuda(self.device)
            tgt_tokens = tgt_tokens.cuda(self.device)
            input_mask = input_mask.cuda(self.device)
            labels_mask = labels_mask.cuda(self.device)

        final_hidden, _ = self.bert_model.work(
            src_tokens.transpose(0, 1))  # [layers, seq_length, batch_size, hidden_size]
        final_hidden = torch.tensor([item.cpu().detach().numpy() for item in final_hidden]).cuda()
        final_hidden = final_hidden[-1].transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        # print("final_hidden:", final_hidden.shape)

        # 使用更好的transformer_decoder解码层
        if self.use_t2t_decoder:
            logits = self.decoder.forward(src_tokens, final_hidden,
                                          tgt_tokens + 2)  # labels is the id of operation, shift 2 for begin and end tokens
            # print("logits:", logits.shape, logits)  # [batch_size, seq_length, output_vocab_size]   grad_fn=<ReshapeAliasBackward0>
        else:
            # 使用普通的dense出概率
            final_hidden = F.dropout(final_hidden, p=self.dropout)
            # print('final_hidden:', final_hidden)
            logits = self.classifier(final_hidden)  # [batch_size, seq_length, num_tags]  grad_fn=<AddBackward0>
        # final_hidden = F.dropout(final_hidden, p=self.dropout)
        # logits = self.classifier(final_hidden)  # [batch_size, seq_length, num_tags]
        # print("logits:", logits.shape)  # [batch_size, seq_length, num_tags]

        loss = None
        if tgt_tokens is not None:
            # 训练阶段，计算 softmax_cross_entropy 损失
            criterion = nn.CrossEntropyLoss(ignore_index=-1)
            # print('tgt_tokens:', tgt_tokens.shape)
            # print("logits.view(-1, self.num_tags):", logits.view(-1, self.num_tags+2).shape)
            # print("tgt_tokens.view(-1):", tgt_tokens.view(-1).shape)
            if self.use_t2t_decoder:
                # print("0000000000000000")
                loss = criterion(logits.view(-1, self.num_tags + 2), tgt_tokens.view(-1))
                print("loss:", loss)  # grad_fn=<NllLossBackward0>
            else:
                loss = criterion(logits.view(-1, self.num_tags),
                                 tgt_tokens.view(-1))  # tensor(6.1017, device='cuda:0', grad_fn=<NllLossBackward0>)
            pred = torch.argmax(logits, dim=-1)
            # print('pred:', pred)
            # print('loss:', loss)

        loss_dic = {"decode_result": pred,
                    "loss": loss}
        return loss_dic

    # def model_test(self, batch, dataloader):
    #     self.eval()
    #     predictions = []
    #     with torch.no_grad():
    #         # for batch in dataloader:
    #         batch = tensor_ready(batch, self.tokenizer, self.max_seq_length)
    #         src_tokens = dataloader.truncate_tensor(batch["ready_source_batch"])  # [batch_size, seq_length]
    #         input_mask = dataloader.truncate_tensor(batch["ready_input_mask_batch"])
    #
    #         if self.device:
    #             src_tokens = src_tokens.cuda(self.device)
    #             input_mask = input_mask.cuda(self.device)
    #
    #         final_hidden, _ = self.bert_model.work(
    #             src_tokens.transpose(0, 1))  # [layers, seq_length, batch_size, hidden_size]
    #         final_hidden = torch.tensor([item.cpu().detach().numpy() for item in final_hidden]).cuda()
    #         final_hidden = final_hidden[-1].transpose(0, 1)  # [batch_size, seq_len, hidden_size]
    #
    #         final_hidden = F.dropout(final_hidden, p=self.dropout)
    #         logits = self.classifier(final_hidden)  # [batch_size, seq_length, num_tags]
    #
    #         pred = torch.argmax(logits, dim=-1)
    #         predictions.append(pred.cpu().numpy())
    #         print("pred:", pred)
    #
    #     return np.concatenate(predictions, axis=0)

    def model_test(self, batch, dataloader):
        batch = tensor_ready(batch, self.tokenizer, self.max_seq_length, self.enable_swap_tag, self.do_lower_case,
                             self.use_arbitrary_target_ids_for_infeasible_examples, self.label_map_file)
        src_tokens = dataloader.truncate_tensor(batch["ready_source_batch"])  # [batch_size, seq_length]
        # print("----------src_tokens:", src_tokens.shape)
        tgt_tokens = dataloader.truncate_tensor(batch["ready_labels_batch"])
        tgt_tokens = np.array(tgt_tokens)
        tmp_idx = (tgt_tokens == 16543)
        tgt_tokens[tmp_idx] = 0
        tgt_tokens = torch.from_numpy(tgt_tokens)

        if self.device:
            src_tokens = src_tokens.cuda(self.device)
            tgt_tokens = tgt_tokens.cuda(self.device)

        final_hidden, _ = self.bert_model.work(
            src_tokens.transpose(0, 1))  # [layers, seq_length, batch_size, hidden_size]
        final_hidden = torch.tensor([item.cpu().detach().numpy() for item in final_hidden]).cuda()
        final_hidden = final_hidden[-1].transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        # print("final_hidden:", final_hidden.shape)

        if self.use_t2t_decoder:
            logits = self.decoder.forward(src_tokens, final_hidden,
                                          tgt_tokens + 2)  # labels is the id of operation, shift 2 for begin and end tokens
        else:
            # 使用普通的dense出概率
            final_hidden = F.dropout(final_hidden, p=self.dropout)
            logits = self.classifier(final_hidden)  # [batch_size, seq_length, num_tags]

        if self.use_t2t_decoder:
            # print("logits:", logits)
            # pred_ids = logits["outputs"]
            pred_ids = logits.argmax(dim=-1).tolist()
            print("pred_ids:", pred_ids)
            # Transformer decoder reserves the first two IDs to the begin and the
            # end token so we shift the IDs back.
            # pred_ids -= 2
        else:
            pred_ids = logits.argmax(dim=-1).tolist()

        # pred_ids = logits.argmax(dim=-1).tolist()
        # print("pred_ids:", pred_ids)
        pred_mask = [0] + [1] * (len(pred_ids) - 2) + [0]
        label_map = utils.read_label_map(self.label_map_file)
        id_2_tag = {tag_id: tagging.Tag(tag) for tag, tag_id in label_map.items()}
        # print("id_2_tag:", id_2_tag[5], id_2_tag[4], id_2_tag[3])
        pred_tokens, output_tokens = [], []
        test_batch_num = len(pred_ids)
        for i in range(test_batch_num):
            pred_tag = []
            if self.use_t2t_decoder:
                # print("pred_ids[i]:", pred_ids[i])
                for label_id in pred_ids[i]:
                    if (label_id > 1):
                        pred_tag.append(id_2_tag[label_id - 2])
                    else:
                        pred_tag.append(id_2_tag[label_id])
                # pred_tag = [id_2_tag[label_id - 2] for label_id in pred_ids[i]]
            else:
                pred_tag = [id_2_tag[label_id] for label_id in pred_ids[i]]
            # print("pred_tag:", pred_tag)
            pred = []
            del_idx = []
            j = 0
            # print("src_tokens:", src_tokens[i])
            # print("tgt_tokens:", tgt_tokens[i])
            while j < len(src_tokens[i]):
                # for j in range(len(src_tokens[i])):  # 暂时没有SWAP操作
                pred_tag[j] = str(pred_tag[j])
                # print("pred_tag[j]:", j, str(pred_tag[j]))
                if (pred_tag[j] == 'KEEP'):
                    pred.append(src_tokens[i][j].tolist())
                    j += 1
                    continue
                elif (pred_tag[j] == 'DELETE'):
                    j += 1
                    del_idx.append(j)
                # elif (pred_tag[j] == 'SWAP'):

                else:
                    tmp_tag = pred_tag[j].split('|')
                    tag_num = len(tmp_tag[1])
                    if (tmp_tag[0] == 'DELETE'):
                        for n in range(tag_num):
                            del_idx.append(j + n)
                    else:
                        # print("tag_num:", tag_num)
                        # print("src_tokens:", len(src_tokens[i]))
                        for n in range(tag_num):
                            if (j + n < len(src_tokens[i])):
                                pred.append(src_tokens[i][j + n].tolist())

                    j += tag_num
            output_tokens.append(pred)
            pred = pred[1:]
            for j in range(len(pred) - 1, 0, -1):
                if (pred[j] == 16543):
                    pred.pop(j)
                else:
                    break
            # print("pred:", pred)
            pred_tokens.append(pred[:-1])

        self.output_result = pred_tokens
        return self.output_result, output_tokens  # test_out, target


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
            # print(layer_id, x.size())
            xs.append(x)
            if layers is not None and layer_id >= max_layer_id:
                break
        # print("--------xs:", len(xs[0]))

        if layers is not None:
            x = torch.stack([xs[i] for i in layers])
            z = torch.tanh(self.one_more_nxt_snt(x[:, 0, :, :]))
        else:
            z = torch.tanh(self.one_more_nxt_snt(x[0]))
        return xs, z

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
