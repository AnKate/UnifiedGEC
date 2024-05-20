# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Transformer decoder."""
from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import torch
import torch.nn as nn

from gectoolkit.model.LaserTagger.official_transformer import attention_layer
from gectoolkit.model.LaserTagger.official_transformer import embedding_layer
from gectoolkit.model.LaserTagger.official_transformer import ffn_layer, model_utils
from gectoolkit.model.LaserTagger.official_transformer import transformer


class TransformerDecoder(transformer.Transformer):
    """Transformer decoder.

    Attributes:
      train: Whether the model is in training mode.
      config: Model hyperparameters.
    """

    def __init__(self, config, train):
        """Initializes layers to build Transformer model.

        Args:
          config: hyperparameter object defining layer sizes, dropout values, etc.
          train: boolean indicating whether the model is in training mode. Used to
            determine if dropout layers should be added.
        """
        self.train = train
        self.config = config
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(config)
        # self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(config["vocab_size"],
        #                                                                       config["hidden_size"])
        # override self.decoder_stack
        if self.config["use_full_attention"]:
            self.decoder_stack = transformer.DecoderStack(config, train)
        else:
            self.decoder_stack = DecoderStack(config, train)

    def forward(self, inputs, encoder_outputs, targets=None):
        attention_bias = model_utils.get_padding_bias(inputs)  # [batch_size, 1, 1, seq_len]
        # print("attention_bias:", attention_bias)
        # print("attention_bias.shape:", attention_bias.shape)
        if targets is None:
            return self.predict(encoder_outputs, attention_bias)
        else:
            # print("decode:", self.decode)
            # print("encoder_outputs:", encoder_outputs.shape)
            logits = self.decode(targets, encoder_outputs, attention_bias)  # [batch_size, seq_len, tags]
            # print("logets111:", logits.shape)
            return logits

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1, self.config["hidden_size"])
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:].unsqueeze(1)
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[i:i + 1]
            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            if self.config["use_full_attention"]:
                encoder_outputs = cache.get("encoder_outputs")
            else:
                encoder_outputs = cache.get("encoder_outputs")[:, i:i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, encoder_outputs, self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = torch.squeeze(logits, dim=1)
            return logits, cache

        return symbols_to_logits_fn


class DecoderStack(nn.Module):
    """Modified Transformer decoder stack.

    Like the standard Transformer decoder stack but:
      1. Removes the encoder-decoder attention layer, and
      2. Adds a layer to project the concatenated [encoder activations, hidden
         state] to the hidden size.
    """

    def __init__(self, config, train):
        super(DecoderStack, self).__init__()
        self.layers = nn.ModuleList()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _ in range(config["num_hidden_layers"]):
            self_attention_layer = attention_layer.SelfAttention(
                config["hidden_size"], config["num_heads"],
                config["attention_dropout"], train).to(device)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                config["hidden_size"], config["decoder_filter_size"],
                config["relu_dropout"], train, config["allow_ffn_pad"]).to(device)
            proj_layer = nn.Linear(2 * config["hidden_size"], config["hidden_size"]).to(device)

            layer = nn.ModuleList([self_attention_layer, feed_forward_network, proj_layer])
            self.layers.append(layer)

        self.output_normalization = transformer.LayerNormalization(config["hidden_size"])

    def forward(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias=None, cache=None):
        for n, layer in enumerate(self.layers):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self_attention_layer = layer[0].to(device)
            feed_forward_network = layer[1].to(device)
            proj_layer = layer[2]

            decoder_inputs = torch.Tensor(decoder_inputs).to(device)
            encoder_outputs = torch.Tensor(encoder_outputs).to(device)

            # print("decoder_inputs:", decoder_inputs.shape)
            decoder_inputs = torch.cat([decoder_inputs, encoder_outputs], dim=-1)
            # print("decoder_inputs:", decoder_inputs.shape, encoder_outputs.shape)
            decoder_inputs = proj_layer(decoder_inputs)

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            decoder_inputs = self_attention_layer(
                decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
            decoder_inputs = feed_forward_network(decoder_inputs)

            return self.output_normalization(decoder_inputs)
