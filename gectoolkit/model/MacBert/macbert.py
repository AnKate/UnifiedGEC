import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertForMaskedLM


PAD_TOKEN = '[PAD]'
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'


def tensor_ready(batch, tokenizer, is_train=False):
    source_list_batch = batch["source_list_batch"]
    target_list_batch = batch["target_list_batch"]
    source_max_len = np.max([len(sent) for sent in source_list_batch])
    target_max_len = np.max([len(sent) for sent in target_list_batch]) + 2

    source_inp_batch = []
    target_inp_batch = []
    for idx, source_inp in enumerate(source_list_batch):

        source_inp = source_inp
        target_inp = target_list_batch[idx]
        source_inp_batch.append(
            tokenizer.convert_tokens_to_ids([CLS_TOKEN])
            + source_inp
            + tokenizer.convert_tokens_to_ids([SEP_TOKEN])
            + tokenizer.convert_tokens_to_ids([PAD_TOKEN]) * (source_max_len - len(source_list_batch[idx]))
        )
        target_inp_batch.append(
            tokenizer.convert_tokens_to_ids([CLS_TOKEN])
            + target_inp
            + tokenizer.convert_tokens_to_ids([SEP_TOKEN])
            + tokenizer.convert_tokens_to_ids([PAD_TOKEN]) * (target_max_len - len(target_list_batch[idx]))
        )

    batch["ready_source_batch"] = source_inp_batch
    batch["ready_target_batch"] = target_inp_batch

    return batch


class MacBert(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.dropout = config["dropout"]
        self.device = config["device"]
        self.batch_size = config["batch_size"]
        self.embedding_size = config["embed_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.gamma = config["gamma"]
        self.num_class = len(dataset.vocab)
        self.vocab = dataset
        self.padding_id = dataset.convert_tokens_to_ids(PAD_TOKEN)
        self.tokenizer = dataset

        bert_config = BertConfig.from_json_file('%s/config.json' % config["pretrained_bert_path"])
        self.encoder = BertForMaskedLM.from_pretrained('%s/pytorch_model.bin' % config["pretrained_bert_path"],
                                                       config=bert_config)

        self.ce_loss = nn.CrossEntropyLoss()

    def get_loss(self, y_pred, y):
        y_pred_shape = y_pred.size()
        print(y.size())
        print(y_pred_shape); exit()
        loss = self.ce_loss(y_pred.reshape(y_pred_shape[0] * y_pred_shape[1], y_pred_shape[2]),
                            y.reshape(y_pred_shape[0] * y_pred_shape[1]))
        return loss

    def forward(self, batch, dataloader):
        '''
        parameters in batch.
        batch['source_batch'] is a list with the source text sentences;
        batch['target_batch'] is a list with the target text sentences;
        batch['source_list_batch'] is the indexed source sentences;
        batch['target_list_batch'] is the indexed target sentences;
        '''

        # print(batch['source_batch'])
        # convert indexed sentences into the required format for the model training
        batch = tensor_ready(batch, self.vocab, is_train=True)

        # truncate indexed sentences into a batch where the sentence lengths are same in the same batch
        source_data = dataloader.truncate_tensor(batch["ready_source_batch"], PAD_TOKEN)
        target_data = dataloader.truncate_tensor(batch["ready_target_batch"], PAD_TOKEN)

        print(source_data.size())
        print(target_data.size())
        exit()

        # put tensor into gpu if there is one
        if self.device:
            source_data = source_data.cuda(self.device)
            target_data = target_data.cuda(self.device)

        logits = self.encoder(source_data).logits
        preds = torch.argmax(logits, dim=-1)

        # compute loss
        loss_ft_fc = self.get_loss(logits, target_data)
        loss_dic = {"decode_result": preds, "loss": loss_ft_fc}

        return loss_dic

    def model_test(self, batch, dataloader):
        # convert indexed sentences into the required format for the model testing
        batch = tensor_ready(batch, self.vocab)
        source_data = dataloader.truncate_tensor(batch["ready_source_batch"], PAD_TOKEN)
        target_data = dataloader.truncate_tensor(batch["ready_target_batch"], PAD_TOKEN)

        # put tensor into gpu if there is one
        if self.device:
            source_data = source_data.cuda(self.device)
            target_data = target_data.cuda(self.device)

        logits = self.encoder(source_data).logits
        preds = torch.argmax(logits, dim=-1)
        preds = preds.detach().cpu().numpy()
        self.decode_result = post_process_decode_result(preds, self.tokenizer)
        return self.decode_result, target_data


def post_process_decode_result(sentences, tokenizer):
    # print(tokenizer.convert_tokens_to_ids([SpecialTokens.EOS_TOKEN]))
    new_sentences = []
    for sent_idx, sent in enumerate(sentences):
        new_sent = []
        sent = sent[1:]
        for w in sent:
            if w in tokenizer.convert_tokens_to_ids([SEP_TOKEN, CLS_TOKEN]):
                break
            new_sent += [w]

        # print(tokenizer.convert_ids_to_tokens(new_sent))
        new_sentences += [new_sent]
    # print(sentences, new_sentences)
    return new_sentences
