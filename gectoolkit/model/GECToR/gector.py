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


# 基本模型架构seq2label
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

        # GECToR的tokenizer在模型训练过程中并未用到, 但是在推断阶段会需要用到
        self.indexer = dataset
        self.cuda_device = torch.device(
            "cuda:" + str(config["cuda_device"]) if int(config["cuda_device"]) >= 0 else "cpu")
        # print(self.cuda_device); exit()
        self.label_namespaces = [config["labels_namespace"], config["detect_namespace"]]
        self.vocab_path = self.model_path + "/vocabulary"  # 词表路径
        self.vocab = Vocabulary.from_files(self.vocab_path)
        self.num_labels_classes = self.vocab.get_vocab_size(config["labels_namespace"])
        self.num_detect_classes = self.vocab.get_vocab_size(config["detect_namespace"])
        self.incorrect_index = self.vocab.get_token_index("INCORRECT", namespace=config["detect_namespace"])

        # 网络结构
        self.hidden_layers = config["hidden_layers"]
        self.hidden_dim = config["hidden_dim"]
        # 使用TimeDistributed对数据进行降维, 便于并行解码
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(config["dropout"]))  # dropout层
        self.tag_labels_hidden_layers = []  # 检测编辑标签的网络层
        self.tag_detect_hidden_layers = []  # 检测预测结果的网络层
        input_dim = self.encoder.get_output_dim()
        # 增添隐藏层
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
            # 投影层
            self.tag_labels_projection_layer = TimeDistributed(Linear(
                self.hidden_dim, self.num_labels_classes)).cuda(self.cuda_device)
            self.tag_detect_projection_layer = TimeDistributed(Linear(
                self.hidden_dim, self.num_detect_classes)).cuda(self.cuda_device)
        # 没有隐藏层则直接添加投影层
        else:
            self.tag_labels_projection_layer = TimeDistributed(Linear(
                input_dim, self.num_labels_classes)).to(self.cuda_device)
            self.tag_detect_projection_layer = TimeDistributed(Linear(
                input_dim, self.num_detect_classes)).to(self.cuda_device)

        # 初始化器, 将模型参数进行初始化
        initializer = InitializerApplicator()
        initializer(self)

        # 测试阶段的相关参数
        self.min_len = config["min_len"]  # 参与预测的句子的最小长度与最大长度
        self.max_len = config["max_len"]
        self.iterations = config["iterations"]
        self.min_error_probability = config["min_error_probability"]
        self.confidence = config["confidence"]

    def forward(self, batch, dataloader):
        # tensor_ready的功能在此处实现
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

        # 经过若干个隐藏层后, 使用全连接层计算得分
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

        # 对得分使用softmax得到概率
        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, seq_len, self.num_labels_classes])
        class_probabilities_d = F.softmax(logits_d, dim=-1).view([batch_size, seq_len, self.num_detect_classes])

        # 每个句子的每个token的错误概率
        error_probs = class_probabilities_d[:, :, self.incorrect_index] * mask
        # 将错误概率的最大值视作本句句子的错误概率
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

        # 有labels与d_tags时即是训练阶段, 此时需要计算loss以便优化参数
        if labels is not None and d_tags is not None:
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            output_dict["loss"] = loss_d + loss_labels

        return output_dict

    def model_test(self, batch, dataloader):
        # 测试(推理)阶段
        # 实质上是对GECToR predict过程的重写
        if self.language == "zh":
            test_inputs = [''.join(i) for i in batch['source_batch']]
            sents = segment(test_inputs)
        else:
            sents = batch['source_batch']

        # 原模型类中的handle_batch方法
        final_batch = sents[:]
        # id -> sentence的映射, 存放每句句子的原始状态
        prev_pred_dict = {i: [final_batch[i]] for i in range(len(final_batch))}
        # 比最短长度还要短的句子不参与推理
        short_ids = [i for i in range(len(sents)) if len(sents[i]) < self.min_len]
        # 其余句子进入推理阶段
        pred_ids = [i for i in range(len(sents)) if i not in short_ids]

        # 迭代推理
        for n_iter in range(self.iterations):
            origin_batch = [final_batch[i] for i in pred_ids]

            # preprocess阶段
            # 对原始的句子进行预处理, 将token组成的list转化为instance类型
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

            # predict阶段
            with torch.no_grad():
                encoded_text = self.encoder(tokens)
                batch_size, seq_len, _ = encoded_text.size()
                mask = get_text_field_mask(tokens)
                # training_mode = self.training
                # print(training_mode)
                self.eval()
                prediction = self.decode(encoded_text, batch_size, seq_len, mask, None, None)
                # self.train(training_mode)

            # convert阶段, 提取prediction中的输出
            # 每条句子中的每个词是某label的概率, [batch_size, seq_len, label_num]
            all_class_prob = prediction['class_probabilities_labels']
            # 每条句子中每个词的错误概率, [batch_size, seq_len, label_num]
            d_tags_class_prob = prediction['class_probabilities_d_tags']
            # 每条句子的错误概率
            error_prob = prediction['max_error_probability'].tolist()

            d_tags_idx = torch.max(d_tags_class_prob, dim=-1)[1].tolist()
            max_vals = torch.max(all_class_prob, dim=-1)
            probs = max_vals[0].tolist()
            idx = max_vals[1].tolist()

            # postprocess阶段
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
                    # get_token_action阶段, 基于编辑标签提取相应操作
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
