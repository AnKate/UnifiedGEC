# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/02/13 22:10
# @File: levenstein_transformer.py

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import os
import copy
import random

from gectoolkit.module.Layer.layers import gelu, LayerNorm
from gectoolkit.module.transformer import TransformerLayer, Embedding, LearnedPositionalEmbedding
from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.model.LevenshteinTransformer.utils import new_arange
from gectoolkit.model.LevenshteinTransformer.transformer import TransformerDecoder


def tensor_ready(batch, tokenizer, is_train=True):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    max_len = np.max([len(sent) for sent in source_list_batch])
    max_len = np.max([len(sent) for sent in target_list_batch] + [max_len]) + 3

    text_list_batch = []
    tag_list_batch = []
    for idx, text_list in enumerate(source_list_batch):
        text_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN]) + text_list
        text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.MASK_TOKEN])
        text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])
        tag_list = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])
        tag_list += target_list_batch[idx]
        tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.MASK_TOKEN])
        tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])

        if is_train:
            tag_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN]) * (
                        max_len - len(target_list_batch[idx]))
            text_list += tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN]) * (
                        max_len - len(source_list_batch[idx]))

        text_list_batch.append(text_list)
        tag_list_batch.append(tag_list)

    batch["ready_source_batch"] = text_list_batch
    batch["ready_target_batch"] = tag_list_batch

    return batch


def inject_noise(target_tokens, tokenizer, noise):
    def _random_delete(target_tokens, tokenizer):
        pad = tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])[0]
        bos = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])[0]
        eos = tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])[0]

        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(target_tokens.eq(bos) | target_tokens.eq(eos), 0.0)
        target_score.masked_fill_(target_mask, 1)
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(1, keepdim=True)
        target_length = torch.clamp(target_length + 1, 1, 3)

        # do not delete <bos> and <eos> (we assign 0 score for them)
        target_cutoff = 2 + ((target_length - 2) * target_score.new_zeros(
            target_score.size(0), 1).uniform_()).long()
        # print('target_cutoff', target_score, target_cutoff)
        # target_cutoff = target_score.sort(1)[1] >= target_cutoff
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        prev_target_tokens = target_tokens.gather(
            1, target_rank).masked_fill_(target_cutoff, pad).gather(
            1,
            target_rank.masked_fill_(target_cutoff,
                                     max_len).sort(1)[1])
        prev_target_tokens = prev_target_tokens[:, :prev_target_tokens.
            ne(pad).sum(1).max()]

        return prev_target_tokens

    def _random_mask(target_tokens, tokenizer):
        pad = tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])[0]
        bos = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])[0]
        eos = tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])[0]
        unk = tokenizer.convert_tokens_to_ids([SpecialTokens.UNK_TOKEN])[0]

        target_masks = target_tokens.ne(pad) & \
                       target_tokens.ne(bos) & \
                       target_tokens.ne(eos)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), unk)
        # print('prev_target_tokens', target_tokens[0], prev_target_tokens[0])
        # exit()
        return prev_target_tokens

    def _random_insert(target_tokens, tokenizer):
        pad = tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])[0]
        bos = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])[0]
        eos = tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])[0]
        unk = tokenizer.convert_tokens_to_ids([SpecialTokens.UNK_TOKEN])[0]
        # random_words = tokenizer.random()[0] #torch.LongTensor([tokenizer.random()[0] for _ in range(target_tokens.size(0))])
        # print(tokenizer.vocab.values())
        random_words = random.sample(set(tokenizer.vocab.values()), k=1)[0]
        # print('random_words', random_words)

        target_masks = target_tokens.ne(pad) & \
                       target_tokens.ne(bos) & \
                       target_tokens.ne(eos)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = torch.clamp(target_length + 1, 1,
                                    3)  # torch.max([target_length + 1, 3])  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), random_words)
        # print('prev_target_tokens', target_cutoff, target_tokens[0], prev_target_tokens[0])
        # exit()
        return prev_target_tokens

    def _full_mask(target_tokens, tokenizer):
        pad = tokenizer.convert_tokens_to_ids([SpecialTokens.PAD_TOKEN])[0]
        bos = tokenizer.convert_tokens_to_ids([SpecialTokens.SOS_TOKEN])[0]
        eos = tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN])[0]
        unk = tokenizer.convert_tokens_to_ids([SpecialTokens.UNK_TOKEN])[0]

        target_mask = target_tokens.eq(bos) | target_tokens.eq(
            eos) | target_tokens.eq(pad)
        return target_tokens.masked_fill(~target_mask, unk)

    if noise == 'random_delete':
        return _random_delete(target_tokens, tokenizer)
    elif noise == 'random_mask':
        return _random_mask(target_tokens, tokenizer)
    elif noise == 'full_mask':
        return _full_mask(target_tokens, tokenizer)
    elif noise == 'random_insert':
        return _random_insert(target_tokens, tokenizer)
    elif noise == 'no_noise':
        return target_tokens
    else:
        raise NotImplementedError


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    try:
        from gectoolkit.model.LevenshteinTransformer import libnat
    except ImportError as e:
        import sys
        sys.stderr.write('ERROR: missing libnat. run `pip install --editable .`\n')
        raise e
    in_seq_len = in_tokens.size(1)
    out_seq_len = out_tokens.size(1)

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]
    # print('in_tokens_list', in_tokens_list, out_tokens_list)
    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    word_del_targets = [b[-1] for b in full_labels]
    word_del_targets = [
        labels + [0 for _ in range(out_seq_len - len(labels))]
        for labels in word_del_targets
    ]

    # transform to tensor
    word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)[:, :in_seq_len]
    return word_del_targets


def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    try:
        from gectoolkit.model.LevenshteinTransformer import libnat
    except ImportError as e:
        import sys
        sys.stderr.write('ERROR: missing libnat. run `pip install --editable .`\n')
        raise e
    in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

    with torch.cuda.device_of(in_tokens):
        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

    full_labels = libnat.suggested_ed2_path(
        in_tokens_list, out_tokens_list, padding_idx
    )
    mask_inputs = [
        [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
    ]

    # generate labels
    masked_tgt_masks = []
    for mask_input in mask_inputs:
        mask_label = []
        for beam_size in mask_input[1:-1]:  # HACK 1:-1
            mask_label += [0] + [1 for _ in range(beam_size)]
        masked_tgt_masks.append(
            (mask_label + [0 for _ in range(out_seq_len - len(mask_label))])[:out_seq_len]
        )
    mask_ins_targets = [
        mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
        for mask_input in mask_inputs
    ]

    # transform to tensor
    masked_tgt_masks = torch.tensor(
        masked_tgt_masks, device=out_tokens.device
    ).bool()
    mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
    masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
    return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets


def _apply_my_del_words(in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx, unk_idx,
                        in_const_del=None, in_const_ins=None):
    # apply deletion to a tensor
    in_masks = in_tokens.eq(padding_idx)

    max_len = in_tokens.size(1)
    # print('_apply_my_del_words before ', word_del_pred[0])
    word_del_pred.masked_fill_(in_masks, 0)
    # print('_apply_my_del_words after ', word_del_pred[0])
    if in_const_del is None:
        bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)
        word_del_pred.masked_fill_(bos_eos_masks, 0)
    else:
        word_del_pred.masked_fill_(in_const_del, 0)
    # print('word_del_pred', word_del_pred)

    out_tokens = in_tokens.masked_fill(word_del_pred, unk_idx)
    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0)

    return out_tokens, out_scores


def _apply_del_words(
        in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx,
        in_const_del=None, in_const_ins=None,
):
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    if in_const_del is None:
        bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)
        word_del_pred.masked_fill_(bos_eos_masks, 0)
    else:
        word_del_pred.masked_fill_(in_const_del, 0)

    reordering = (
        new_arange(in_tokens)
            .masked_fill_(word_del_pred, max_len)
            .sort(1)[1]
    )

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.).gather(1, _reordering)

    # update constraint mask
    out_const_del = None
    if in_const_del is not None:
        out_const_del = in_const_del.gather(1, reordering)

    out_const_ins = None
    if in_const_ins is not None:
        out_const_ins = in_const_ins.gather(1, reordering)

    return out_tokens, out_scores, out_attn, out_const_del, out_const_ins


def _apply_ins_masks(
        in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx,
        in_const_del=None, in_const_ins=None
):
    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)
    if in_const_ins is not None:
        mask_ins_pred.masked_fill_(~in_const_ins[:, :-1], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = (
            new_arange(out_lengths, out_max_len)[None, :]
            < out_lengths[:, None]
    )

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    # print('reordering', reordering)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
            .fill_(padding_idx)
            .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])
    # print('out_scores', out_scores)

    # update constraint mask
    out_const_del = None
    if in_const_del is not None:
        out_const_del = in_const_del.new_zeros(*out_tokens.size())
        out_const_del[:, 0] = in_const_del[:, 0]
        out_const_del.scatter_(1, reordering, in_const_del[:, 1:])

    out_const_ins = None
    if in_const_ins is not None:
        # default value is 1 to allow insertion outside constraint tokens
        out_const_ins = in_const_ins.new_ones(*out_tokens.size())
        out_const_ins[:, 0] = in_const_ins[:, 0]
        out_const_ins.scatter_(1, reordering, in_const_ins[:, 1:])

    return out_tokens, out_scores, out_const_del, out_const_ins


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


class LevenshteinTransformer(nn.Module):
    # def __init__(self, bert_model, num_class, embedding_size, batch_size, dropout, device, tokenizer, loss_type='FC_FT_CRF'):
    def __init__(self, config, dataset):
        super().__init__()
        self.tokenizer = tokenizer = dataset
        self.encoder_embed_dim = encoder_embed_dim = config["encoder_embed_dim"]
        self.checkpoint_dir = config["pretrained_model_path"]
        self.dropout = config["dropout"]
        self.iter_decode_max_iter = config["iter_decode_max_iter"]
        self.pad_id = dataset.convert_tokens_to_ids(SpecialTokens.PAD_TOKEN)
        self.unk_id = dataset.convert_tokens_to_ids(SpecialTokens.UNK_TOKEN)
        self.bos_id = dataset.convert_tokens_to_ids(SpecialTokens.SOS_TOKEN)
        self.eos_id = dataset.convert_tokens_to_ids(SpecialTokens.EOS_TOKEN)
        self.mask_id = dataset.convert_tokens_to_ids(SpecialTokens.MASK_TOKEN)
        self.pred_special_ids = [self.pad_id, self.unk_id, self.bos_id, self.eos_id, self.mask_id]
        self.max_mask = config["max_mask"]
        self.device = config["device"]
        self.num_class = num_class = len(dataset.vocab)

        self.encoder = BERTLM(config, dataset)
        embed_tokens = self.encoder.tok_embed
        # print('embed_tokens', embed_tokens.weight.size())
        self.decoder = LevenshteinTransformerDecoder(config, tokenizer.vocab, copy.deepcopy(embed_tokens),
                                                     # \embed_tokens, #
                                                     no_encoder_attn=True)
        self.fc = nn.Linear(self.encoder_embed_dim, self.num_class)
        self.del_layer = nn.Linear(self.encoder_embed_dim, 2)

        self.initial_parameters()

    def initial_parameters(self):
        checkpoint_dir = self.checkpoint_dir
        pretrained_model = torch.load(os.path.join(checkpoint_dir, 'pre-train.ckpt'))
        new_pretrained_model = {}
        for k in pretrained_model['model'].keys():
            v = pretrained_model['model'][k]
            if k == 'tok_embed.weight':
                new_pretrained_model[k] = torch.cat([v, v[:10]], 0)
            elif k == 'out_proj_bias':
                new_pretrained_model[k] = torch.cat([v, v[:10]], 0)
            else:
                new_pretrained_model[k] = v
            # print("encoder.tok_embed.weight", new_pretrained_model['encoder.tok_embed.weight'].size())
        self.encoder.load_state_dict(new_pretrained_model, strict=False)
        # print(pretrained_model.keys());
        # print(pretrained_model['model'].keys()); exit()

    def ins_loss(self, y_pred, y, y_mask, gamma=None, avg=True):
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

    def del_loss(self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=10.0):
        def mean_ds(x, dim=None):
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        # print('before mask outputs size', outputs.size(), masks.size())
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
        else:
            vocab_size = outputs.size(-1)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)
        # print('targets', outputs.size(), outputs[0], targets[0])
        # print ('after mask outputs size', outputs.size(), masks.size())
        # if name == 'w_ins-loss':
        #     print ('targets', outputs.size(), targets.size())
        _, class_num = outputs.size()
        logits = F.log_softmax(outputs, dim=1)
        # print ('logits', logits.size())
        # print(logits.max(-1)[1])
        if name == "del_loss":
            losses = F.nll_loss(logits, targets, weight=torch.tensor([0.2, 0.8], device=logits.device),
                                reduction="none")
        elif name == "mask_loss":
            class_weight = [0.4] + [1] * (class_num - 1)
            # print('class_weight', class_weight); exit()
            losses = F.nll_loss(logits, targets, weight=torch.tensor(class_weight, device=logits.device),
                                reduction="none")
        elif targets.dim() == 1:  #
            losses = F.nll_loss(logits, targets, reduction="none")
        else:
            losses = F.kl_div(logits, targets, reduction="none")
            losses = losses.float().sum(-1).type_as(losses)

        if losses.size(0) == 0: losses = torch.tensor([0.], device=losses.device)
        nll_loss = mean_ds(losses)
        if False:  # label_smoothing > 0:
            loss = nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
        else:
            loss = nll_loss

        loss = loss * factor
        # print('losses', loss)
        return loss

    def post_process_decode_result(self, sentences):
        new_sentences = []
        for sent_idx, sent in enumerate(sentences):
            sent = sent[1:]
            sent = [w for w in sent if w not in self.pred_special_ids]

            new_sentences += [sent]
        return new_sentences

    def forward(self, batch, dataloader, is_display=False):
        batch = tensor_ready(batch, self.tokenizer)
        src_tokens = dataloader.truncate_tensor(batch["ready_source_batch"])
        tgt_tokens = dataloader.truncate_tensor(batch["ready_target_batch"])
        current_batch_size, seq_len = src_tokens.size()

        if np.random.random() < 0.5:
            src_tokens = inject_noise(tgt_tokens, self.tokenizer, 'random_insert')

        if self.device:
            src_tokens = src_tokens.cuda(self.device)
            tgt_tokens = tgt_tokens.cuda(self.device)

        # print("src_tokens and tgt_tokens", src_tokens.size(), tgt_tokens.size())
        src_masks = src_tokens.eq(self.pad_id)
        raw_encoder_out, _ = self.encoder.work(src_tokens.transpose(0, 1))
        encoder_out = {"encoder_out": raw_encoder_out[-1].transpose(0, 1), "encoder_padding_mask": src_masks}
        # print('encoder_out', encoder_out.size())

        word_del_targets = _get_del_targets(src_tokens, tgt_tokens, self.pad_id)  # [:, :seq_len]
        word_del_out, _ = self.decoder.forward_word_del(
            src_tokens, encoder_out=encoder_out
        )
        word_del_pred = word_del_out.max(-1)[1].bool()
        del_loss = self.del_loss(word_del_out, word_del_targets, tgt_tokens.ne(self.pad_id), name='del_loss')

        src_tokens, _, _, _, _ = _apply_del_words(
            src_tokens, None, None, word_del_targets.bool(),
            self.pad_id, self.bos_id, self.eos_id
        )
        # print('word_del_pred', word_del_pred[0])

        masked_tgt_masks, masked_tgt_tokens, masked_ins_targets = _get_ins_targets(
            src_tokens, tgt_tokens, self.pad_id, self.unk_id
        )
        masked_ins_masks = src_tokens[:, 1:].ne(self.pad_id)
        # print("masked_tgt_masks", masked_tgt_masks[0], masked_tgt_tokens[0], masked_ins_targets[0])
        # print("masked_tgt_masks size", masked_tgt_masks.size(), masked_tgt_tokens.size(), masked_ins_targets.size())
        mask_ins_out, _ = self.decoder.forward_mask_ins(
            src_tokens, encoder_out=encoder_out
        )
        mask_loss = self.del_loss(mask_ins_out, masked_ins_targets, masked_ins_masks, name='mask_loss')
        mask_ins_pred = mask_ins_out.max(-1)[1]  # .bool()[:, 1:]
        # print('mask_ins_pred', mask_ins_pred[0]); # exit()

        word_ins_out, _ = self.decoder.forward_word_ins(
            masked_tgt_tokens, encoder_out=encoder_out  # I changed this line !
        )
        self.decode_result = F.softmax(word_ins_out, dim=-1).max(2)[1]
        ins_loss = self.del_loss(word_ins_out, tgt_tokens, tgt_tokens.ne(self.pad_id))  # masked_tgt_masks
        if False:  # is_display:
            print('self.decode_result', self.decode_result[0], tgt_tokens[0], ins_loss)
        # exit()

        loss_dic = {"decode_result": self.decode_result,
                    "loss": ins_loss + del_loss + mask_loss,
                    "loss_del": del_loss,
                    "loss_mask": mask_loss,
                    "loss_ins": ins_loss}
        return loss_dic

    def model_test(self, batch, dataloader):
        batch = tensor_ready(batch, self.tokenizer)
        src_tokens = dataloader.truncate_tensor(batch["ready_source_batch"])
        tgt_tokens = dataloader.truncate_tensor(batch["ready_target_batch"])
        current_batch_size, seq_len = src_tokens.size()

        if self.device:
            src_tokens = src_tokens.cuda(self.device)
            tgt_tokens = tgt_tokens.cuda(self.device)
        # print('self.iter_decode_max_iter', self.iter_decode_max_iter) self.iter_decode_max_iter
        # print('src_tokens', src_tokens.size())
        for _ in range(7):
            src_masks = src_tokens.eq(self.pad_id)
            raw_encoder_out, _ = self.encoder.work(src_tokens.transpose(0, 1))
            encoder_out = {"encoder_out": raw_encoder_out[-1].transpose(0, 1), "encoder_padding_mask": src_masks}

            # word_del_out, _ = self.decoder.forward_word_ins(src_tokens)
            word_del_out, _ = self.decoder.forward_word_del(src_tokens, encoder_out)
            # print('src_tokens', src_tokens[0])
            word_del_score = F.log_softmax(word_del_out, 2)
            word_del_pred = word_del_score.max(-1)[1].bool()
            # print('word_del_pred', word_del_pred[0])

            output_tokens, _, _, _, _ = _apply_del_words(
                src_tokens, None, None, word_del_pred,
                self.pad_id, self.bos_id, self.eos_id
            )
            # print('after del', output_tokens[0])
            mask_ins_out, _ = self.decoder.forward_mask_ins(
                output_tokens, encoder_out)
            mask_ins_score = F.log_softmax(mask_ins_out, 2)
            mask_ins_pred = mask_ins_score.max(-1)[1]
            output_tokens, _, _, _ = _apply_ins_masks(
                output_tokens, None, mask_ins_pred,
                self.pad_id, self.unk_id, self.eos_id
            )

            word_ins_out, _ = self.decoder.forward_word_ins(
                output_tokens, encoder_out=encoder_out
            )
            # word_ins_out = self.fc(raw_encoder_out[11].transpose(0, 1))
            word_ins_score, word_ins_pred = F.log_softmax(word_ins_out, 2).max(-1)

            output_tokens, _ = _apply_ins_words(
                output_tokens, None, word_ins_pred,
                None, self.unk_id
            )
            src_tokens = output_tokens
        # print('after insertion', output_tokens[0]); #exit()
        self.decoder_result = self.post_process_decode_result(output_tokens.tolist())
        # print(self.decoder_result)

        return self.decoder_result, output_tokens


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


class LevenshteinTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary[SpecialTokens.SOS_TOKEN]
        self.unk = dictionary[SpecialTokens.UNK_TOKEN]
        self.eos = dictionary[SpecialTokens.EOS_TOKEN]
        self.device = embed_tokens.weight.device
        self.early_exit = args["early_exit"]

        self.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
        self.embed_mask_ins = Embedding(256, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in self.early_exit.split(',')]
        assert len(self.early_exit) == 3

        # copy layers for mask-predict/deletion
        self.layers_msk = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_msk = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.early_exit[1])
            ])
        # print(self.layers_msk); exit()
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.early_exit[0])
            ])
        self.embed_dim = embed_tokens.weight.size(1)

    def init_hidden(self, batch_size):
        return (torch.randn(2, batch_size, self.embed_dim // 2).to(self.embed_mask_ins.weight.device),
                torch.randn(2, batch_size, self.embed_dim // 2).to(self.embed_mask_ins.weight.device))

    def extract_features(
            self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        # print('x', x[0, :3, :3])
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        # print('x', x[0, :3, :3])

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        first_attn = None
        # print('layers', layers); exit()
        # print('x', x[0, :3, :3], encoder_out["encoder_out"][0, :3, :3])
        for _, layer in enumerate(layers[: early_exit]):
            x, attn = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"] if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask,
            )
            if first_attn is None:
                first_attn = attn
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": first_attn, "inner_states": inner_states}

    def forward_mask_ins(self, prev_output_tokens, encoder_out=None, **unused):
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[1], layers=self.layers_msk, **unused
        )
        features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
        return F.linear(features_cat, self.embed_mask_ins.weight), extra['attn']

    def forward_word_ins(self, prev_output_tokens, encoder_out=None, **unused):
        # print('prev_output_tokens', prev_output_tokens[0])
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[2], layers=self.layers, **unused
        )
        features = self.output_layer(features)
        # print('decode_result', decode_result.size())
        return features, extra['attn']

    def forward_word_del(self, prev_output_tokens, encoder_out=None, **unused):
        # print('encoder_out', encoder_out)
        features, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, early_exit=self.early_exit[0], layers=self.layers_del, **unused
        )
        # self.hidden = self.init_hidden(features.size(0))
        # features, _ = self.LSTM(features, self.hidden)
        return F.linear(features, self.embed_word_del.weight), extra['attn']



