import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.modules import TimeDistributed
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from torch.nn.modules.linear import Linear
from gectoolkit.utils.helpers import get_target_sent_by_edits
from gectoolkit.model.GECToR.segment.segment_bert import segment


class GECToR(nn.Module):
    def __init__(self, config, dataset):
        super(GECToR, self).__init__()

        self.language = config["language"]
        if self.language == "zh":
            self.model_path = config["pretrained_model_path"] + "/Chinese"
        elif self.language == "en":
            self.model_path = config["pretrained_model_path"] + "/English"
        else:
            self.model_path = config["pretrained_model_path"] + "/Multilingual"
        self.tune_bert = config["tune_bert"]
        bert_token_embedder = PretrainedTransformerEmbedder(model_name=self.model_path, last_layer_only=True,
                                                            train_parameters=self.tune_bert)
        token_embedders = {'bert': bert_token_embedder}
        self.encoder = BasicTextFieldEmbedder(token_embedders=token_embedders)

        self.indexer = dataset
        self.cuda_device = torch.device(
            "cuda:" + str(config["cuda_device"]) if int(config["cuda_device"]) >= 0 else "cpu")
        self.label_namespaces = [config["labels_namespace"], config["detect_namespace"]]
        self.vocab_path = self.model_path + "/vocabulary"
        self.vocab = Vocabulary.from_files(self.vocab_path)
        self.num_labels_classes = self.vocab.get_vocab_size(config["labels_namespace"])
        self.num_detect_classes = self.vocab.get_vocab_size(config["detect_namespace"])
        self.incorrect_index = self.vocab.get_token_index("INCORRECT", namespace=config["detect_namespace"])

        self.hidden_layers = config["hidden_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config["dropout"]))
        self.tag_labels_hidden_layers = []
        self.tag_detect_hidden_layers = []
        input_dim = self.encoder.get_output_dim()

        if self.hidden_layers > 0:
            self.tag_labels_hidden_layers.append(TimeDistributed(
                Linear(input_dim, self.hidden_dim)).cuda(self.cuda_device))
            self.tag_detect_hidden_layers.append(TimeDistributed(
                Linear(input_dim, self.hidden_dim)).cuda(self.cuda_device))
            for _ in range(self.hidden_layers - 1):
                self.tag_labels_hidden_layers.append(TimeDistributed(
                    Linear(self.hidden_dim, self.hidden_dim)).cuda(self.cuda_device))
                self.tag_detect_hidden_layers.append(TimeDistributed(
                    Linear(self.hidden_dim, self.hidden_dim)).cuda(self.cuda_device))

            self.tag_labels_projection_layer = TimeDistributed(Linear(
                self.hidden_dim, self.num_labels_classes)).cuda(self.cuda_device)
            self.tag_detect_projection_layer = TimeDistributed(Linear(
                self.hidden_dim, self.num_detect_classes)).cuda(self.cuda_device)

        else:
            self.tag_labels_projection_layer = TimeDistributed(Linear(
                input_dim, self.num_labels_classes)).to(self.cuda_device)
            self.tag_detect_projection_layer = TimeDistributed(Linear(
                input_dim, self.num_detect_classes)).to(self.cuda_device)


        initializer = InitializerApplicator()
        initializer(self)

        self.min_len = config["min_len"]
        self.max_len = config["max_len"]
        self.iterations = config["iterations"]
        self.min_error_probability = config["min_error_probability"]
        self.confidence = config["confidence"]

    def forward(self, batch, dataloader):
        batch = Batch(batch['instance_batch'])
        batch = util.move_to_device(batch.as_tensor_dict(), self.cuda_device if torch.cuda.is_available() else -1)
        tokens = batch['tokens']
        labels = batch['labels'].to(self.cuda_device)
        d_tags = batch['d_tags'].to(self.cuda_device)

        encoded_text = self.encoder(tokens)
        batch_size, seq_len, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)

        ret_train = self.decode(encoded_text, batch_size, seq_len, mask, labels, d_tags)
        _loss = ret_train["loss"]
        output_dict = {"loss": _loss}
        return output_dict

    def decode(self, encoded_text: torch.LongTensor = None,
               batch_size: int = 0,
               seq_len: int = 0,
               mask: torch.LongTensor = None,
               labels: torch.LongTensor = None,
               d_tags: torch.LongTensor = None):

        if self.tag_labels_hidden_layers:
            encoded_text_labels = encoded_text.clone().to(self.device)
            for layer in self.tag_labels_hidden_layers:
                encoded_text_labels = layer(encoded_text_labels)
            logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text_labels))
            for layer in self.tag_detect_hidden_layers:
                encoded_text = layer(encoded_text)
            logits_d = self.tag_detect_projection_layer(self.predictor_dropout(encoded_text))
        else:
            logits_labels = self.tag_labels_projection_layer(self.predictor_dropout(encoded_text))
            logits_d = self.tag_detect_projection_layer(self.predictor_dropout(encoded_text))

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, seq_len, self.num_labels_classes])
        class_probabilities_d = F.softmax(logits_d, dim=-1).view([batch_size, seq_len, self.num_detect_classes])

        error_probs = class_probabilities_d[:, :, self.incorrect_index] * mask
        incorrect_prob = torch.max(error_probs, dim=-1)[0]

        if self.confidence > 0:
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            offset = torch.FloatTensor(probability_change).repeat((batch_size, seq_len, 1)).to(self.cuda_device)
            class_probabilities_labels += util.move_to_device(offset, self.cuda_device)

        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorrect_prob}

        if labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            output_dict["loss"] = loss_d + loss_labels

        return output_dict

    def model_test(self, batch, dataloader):
        if self.language == "zh":
            test_inputs = [''.join(i) for i in batch['source_batch']]
            sents = segment(test_inputs)
        else:
            sents = batch['source_batch']

        final_batch = sents[:]
        prev_pred_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        short_ids = [i for i in range(len(sents)) if len(sents[i]) < self.min_len]
        pred_ids = [i for i in range(len(sents)) if i not in short_ids]

        for n_iter in range(self.iterations):
            origin_batch = [final_batch[i] for i in pred_ids]

            with torch.cuda.device(self.cuda_device):
                seq_lens = [len(seq) for seq in origin_batch if seq]
                if not seq_lens:
                    break
                max_len = min(max(seq_lens), self.max_len)
                batch = []
                for seq in origin_batch:
                    tokens = seq[:max_len]
                    tokens = [Token(token) for token in ['$START'] + tokens]
                    # print(tokens)
                    batch.append(Instance({'tokens': TextField(tokens, self.indexer)}))
                batch = Batch(batch)
                batch.index_instances(self.vocab)
                # print(batch.as_tensor_dict()); exit()
                batch = util.move_to_device(batch.as_tensor_dict(),
                                            self.cuda_device if torch.cuda.is_available() else -1)
                tokens = batch['tokens']

            with torch.no_grad():
                encoded_text = self.encoder(tokens)
                batch_size, seq_len, _ = encoded_text.size()
                mask = get_text_field_mask(tokens)
                # training_mode = self.training
                # print(training_mode)
                self.eval()
                prediction = self.decode(encoded_text, batch_size, seq_len, mask, None, None)
                # self.train(training_mode)

            all_class_prob = prediction['class_probabilities_labels']
            d_tags_class_prob = prediction['class_probabilities_d_tags']
            error_prob = prediction['max_error_probability'].tolist()

            d_tags_idx = torch.max(d_tags_class_prob, dim=-1)[1].tolist()
            max_vals = torch.max(all_class_prob, dim=-1)
            probs = max_vals[0].tolist()
            idx = max_vals[1].tolist()

            results = []
            noop_index = self.vocab.get_token_index("$KEEP", "labels")
            for tokens, probabilities, indexes, error_probs, d_tags_indexes in zip(origin_batch,
                                                                                   probs,
                                                                                   idx,
                                                                                   error_prob,
                                                                                   d_tags_idx):
                length = min(len(tokens), max_len)
                edits = []

                if max(indexes) == 0:
                    results.append(tokens)
                    continue

                if error_probs < self.min_error_probability:
                    results.append(tokens)
                    continue

                for i in range(length + 1):
                    if indexes[i] == noop_index:
                        continue
                    suggested_action = self.vocab.get_token_from_index(indexes[i], namespace='labels')
                    start_pos = 0
                    end_pos = 0
                    if probabilities[i] < self.min_error_probability or suggested_action in ["@@UNKNOWN@@",
                                                                                             "@@PADDING@@", "$KEEP"]:
                        action = None
                    else:
                        if suggested_action.startswith('$REPLACE_') or suggested_action.startswith(
                                '$TRANSFORM_') or suggested_action == "$DELETE":
                            start_pos = i
                            end_pos = i + 1
                        elif suggested_action.startswith("$APPEND_") or suggested_action.startswith("$MERGE_"):
                            start_pos = i + 1
                            end_pos = i + 1

                        if suggested_action == "$DELETE":
                            suggested_action_clear = ""
                        elif suggested_action.startswith("$TRANSFORM_") or suggested_action.startswith("$MERGE_"):
                            suggested_action_clear = suggested_action[:]
                        else:
                            suggested_action_clear = suggested_action[suggested_action.index('_') + 1:]

                        action = start_pos - 1, end_pos - 1, suggested_action_clear, probabilities[i]
                    if not action:
                        continue
                    edits.append(action)

                results.append(get_target_sent_by_edits(tokens, edits))

            new_pred_ids = []
            for i, origin_id in enumerate(pred_ids):
                origin = final_batch[origin_id]
                pred = results[i]
                prev_pred = prev_pred_dict[origin_id]
                if origin != pred and pred not in prev_pred:
                    final_batch[origin_id] = pred
                    new_pred_ids.append(origin_id)
                    prev_pred_dict[origin_id].append(pred)
                elif origin != pred and pred in prev_pred:
                    final_batch[origin_id] = pred
                else:
                    continue
            if not new_pred_ids:
                break
            pred_ids = new_pred_ids
        return final_batch, None
