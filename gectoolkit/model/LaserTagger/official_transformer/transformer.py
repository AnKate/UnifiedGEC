# coding=utf-8
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Defines the Transformer model, and its encoder and decoder stacks.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from gectoolkit.model.LaserTagger.official_transformer import attention_layer
from gectoolkit.model.LaserTagger.official_transformer import embedding_layer
from gectoolkit.model.LaserTagger.official_transformer import ffn_layer, model_utils, beam_search

EOS_ID = 1
_NEG_INF = -1e9


class Transformer(object):
    """Transformer model for sequence to sequence data.

    Implemented as described in: https://arxiv.org/pdf/1706.03762.pdf

    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """

    def __init__(self, config, train):
        """Initialize layers to build Transformer model.

        Args:
          config: hyperparameter object defining layer sizes, dropout values, etc.
          train: boolean indicating whether the model is in training mode. Used to
            determine if dropout layers should be added.
        """
        self.train = train
        self.config = config

        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(config["vocab_size"],
                                                                              config["hidden_size"])
        self.encoder_stack = EncoderStack(config, train)
        self.decoder_stack = DecoderStack(config, train)

    def forward(self, inputs, targets=None):
        """Calculate target logits or inferred target sequences.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          targets: None or int tensor with shape [batch_size, target_length].

        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
            returns a dictionary {
              output: [batch_size, decoded length]
              score: [batch_size, float]}
        """
        # Variance scaling is used here because it seems to work in many problems.
        # Other reasonable initializers may also work just as well.
        initializer = nn.init.xavier_uniform_
        with torch.no_grad():
            # Calculate attention bias for encoder self-attention and decoder
            # multi-headed attention layers.
            attention_bias = model_utils.get_padding_bias(inputs)
        # with torch.no_grad():
        #     self.apply(initializer)
        # with torch.no_grad():
        #     self.embedding_softmax_layer.apply(initializer)
        #
        # # Calculate attention bias for encoder self-attention and decoder
        # # multi-headed attention layers.
        # attention_bias = model_utils.get_padding_bias(inputs)

        # Run the inputs through the encoder layer to map the symbol
        # representations to continuous representations.
        encoder_outputs = self.encode(inputs, attention_bias)

        # Generate output sequence if targets is None, or return logits if target
        # sequence is known.
        if targets is None:
            # print("11111111111target None")
            return self.predict(encoder_outputs, attention_bias)
        else:
            # print("22222222222target not None")
            # print("shape:", targets.shape, encoder_outputs.shape, attention_bias.shape)
            logits = self.decode(targets, encoder_outputs, attention_bias)
            return logits

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.

        Args:
          inputs: int tensor with shape [batch_size, input_length].
          attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
          float tensor with shape [batch_size, input_length, hidden_size]
        """
        # with torch.no_grad():
        # Prepare inputs to the layer stack by adding positional encodings and
        # applying dropout.
        embedded_inputs = self.embedding_softmax_layer(inputs)
        inputs_padding = (inputs == 0)

            # with torch.no_grad():
        # Create position encoding matrix
        length = embedded_inputs.size(1)
        pos_encoding = model_utils.get_position_encoding(
            length, self.config["hidden_size"])
        pos_encoding = torch.FloatTensor(pos_encoding).to(inputs.device)

        encoder_inputs = embedded_inputs + pos_encoding
        if self.train:
            encoder_inputs = F.dropout(
                encoder_inputs, p=1 - self.config["layer_postprocess_dropout"], training=True)

            return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):
        """Generate logits for each value in the target sequence.

        Args:
            targets: target values for the output sequence.
                int tensor with shape [batch_size, target_length]
            encoder_outputs: continuous representation of input sequence.
                float tensor with shape [batch_size, input_length, hidden_size]
            attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
            float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        # with torch.no_grad():
        # Prepare inputs to decoder layers by shifting targets, adding positional
        # encoding and applying dropout.
        decoder_inputs = self.embedding_softmax_layer(targets)
        decoder_inputs = F.pad(decoder_inputs, (0, 0, 1, 0, 0, 0))[:, :-1, :]
        length = decoder_inputs.size(1)
        decoder_inputs += model_utils.get_position_encoding(
            length, self.config["hidden_size"]).to(decoder_inputs.device)
        if self.train:
            decoder_inputs = F.dropout(
                decoder_inputs, p=1 - self.config["layer_postprocess_dropout"], training=True)

        # Run values
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length).to(encoder_outputs.device)
        outputs = self.decoder_stack(
            decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias)
        logits = self.embedding_softmax_layer.linear(outputs)
        return logits


    # def decode(self, targets, encoder_outputs, attention_bias):
    #     """Generate logits for each value in the target sequence.
    #
    #     Args:
    #       targets: target values for the output sequence.
    #         int tensor with shape [batch_size, target_length]
    #       encoder_outputs: continuous representation of input sequence.
    #         float tensor with shape [batch_size, input_length, hidden_size]
    #       attention_bias: float tensor with shape [batch_size, 1, 1, input_length]
    #
    #     Returns:
    #       float32 tensor with shape [batch_size, target_length, vocab_size]
    #     """
    #     with torch.no_grad():
    #         # with torch.cuda.amp.autocast(enabled=self.params["use_float16"]):
    #         # Prepare inputs to decoder layers by shifting targets, adding positional
    #         # encoding and applying dropout.
    #         decoder_inputs = self.embedding_softmax_layer(targets)
    #         with torch.no_grad():
    #             # with torch.cuda.amp.autocast(enabled=self.params["use_float16"]):
    #             with torch.no_grad():
    #                 # Shift targets to the right, and remove the last element
    #                 decoder_inputs = torch.nn.functional.pad(
    #                     decoder_inputs, [0, 0, 1, 0, 0, 0])[:, :-1, :]
    #             with torch.no_grad():
    #                 length = decoder_inputs.size(1)
    #                 pos_encoding = model_utils.get_position_encoding(
    #                     length, self.config["hidden_size"])
    #                 decoder_inputs += pos_encoding.to(decoder_inputs.device)
    #             if self.train:
    #                 decoder_inputs = F.dropout(
    #                     decoder_inputs, p=1 - self.config["layer_postprocess_dropout"], training=True)
    #
    #             decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length)
    #             outputs = self.decoder_stack(decoder_inputs, encoder_outputs, decoder_self_attention_bias,
    #                                          attention_bias)
    #
    #             logits = self.embedding_softmax_layer.linear(outputs)
    #             return logits

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""

        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1, self.config["hidden_size"]).unsqueeze(0)
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            max_decode_length).unsqueeze(0)

        def symbols_to_logits_fn(ids, i, cache):
            """Generate logits for next potential IDs.

            Args:
              ids: Current decoded sequences.
                int tensor with shape [batch_size * beam_size, i + 1]
              i: Loop index
              cache: dictionary of values storing the encoder output, encoder-decoder
                attention bias, and previous decoder attention values.

            Returns:
              Tuple of
                (logits with shape [batch_size * beam_size, vocab_size],
                 updated cache values)
            """
            # Set decoder input to the last generated IDs
            decoder_input = ids[:, -1:]

            # Preprocess decoder input by getting embeddings and adding timing signal.
            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input += timing_signal[:, i:i + 1, :]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)
            logits = logits.squeeze(1)
            return logits, cache

        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        batch_size = encoder_outputs.size(0)
        input_length = encoder_outputs.size(1)
        max_decode_length = input_length + self.config["extra_decode_length"]

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        # Create initial set of IDs that will be passed into symbols_to_logits_fn.
        initial_ids = torch.zeros(batch_size, dtype=torch.int64, device=encoder_outputs.device)

        # Create cache storing decoder attention values for each layer.
        cache = {
            "layer_%d" % layer: {
                "k": torch.zeros(batch_size, 0, self.config["hidden_size"], dtype=torch.float32,
                                 device=encoder_outputs.device),
                "v": torch.zeros(batch_size, 0, self.config["hidden_size"], dtype=torch.float32,
                                 device=encoder_outputs.device),
            } for layer in range(self.config["num_hidden_layers"])}

        # Add encoder output and attention bias to the cache.
        cache["encoder_outputs"] = encoder_outputs
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

        # Use beam search to find the top beam_size sequences and scores.
        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.config["vocab_size"],
            beam_size=self.config["beam_size"],
            alpha=self.config["alpha"],
            max_decode_length=max_decode_length,
            eos_id=EOS_ID)

        # Get the top sequence for each batch element
        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}


class LayerNormalization(nn.Module):
    """Applies layer normalization."""

    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size
        self.scale = nn.Parameter(torch.ones(hidden_size)).to(device)
        self.bias = nn.Parameter(torch.zeros(hidden_size)).to(device)
        self.epsilon = 1e-6

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        norm_x = (x - mean) * torch.rsqrt(variance + self.epsilon)
        return norm_x * self.scale + self.bias


class PrePostProcessingWrapper(nn.Module):
    """Wrapper class that applies layer pre-processing and post-processing."""

    def init(self, layer, config, train):
        super(PrePostProcessingWrapper, self).init()

        self.layer = layer
        self.postprocess_dropout = config["layer_postprocess_dropout"]
        self.train = train

        # Create normalization layer
        self.layer_norm = LayerNormalization(config["hidden_size"])

    def forward(self, x, *args, **kwargs):
        # Preprocessing: apply layer normalization
        y = self.layer_norm(x)

        # Get layer output
        y = self.layer(y, *args, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = nn.Dropout(1 - self.postprocess_dropout)(y)
        return x + y


class EncoderStack(nn.Module):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, config, train):
        super(EncoderStack, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(config["num_hidden_layers"]):
            # Create sublayers for each layer.
            self_attention_layer = attention_layer.SelfAttention(
                config["hidden_size"], config["num_heads"],
                config["attention_dropout"], train)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                config["hidden_size"], config["filter_size"],
                config["relu_dropout"], train, config["allow_ffn_pad"])

            self.layers.append(nn.ModuleList([
                PrePostProcessingWrapper(self_attention_layer, config, train),
                PrePostProcessingWrapper(feed_forward_network, config, train)]))

        # Create final layer normalization layer.
        self.output_normalization = LayerNormalization(config["hidden_size"])

    def forward(self, encoder_inputs, attention_bias, inputs_padding):
        """Return the output of the encoder layer stacks.

        Args:
          encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
          attention_bias: bias for the encoder self-attention layer.
            [batch_size, 1, 1, input_length]
          inputs_padding: P

        Returns:
          Output of encoder layer stack.
          float32 tensor with shape [batch_size, input_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            with torch.no_grad():
                encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)

            with torch.no_grad():
                encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

            # with torch.no_grad():
            #     with torch.cuda.amp.autocast(enabled=False):
            #         with torch.cuda.amp.autocast(enabled=use_amp):
            #             with torch.autograd.profiler.record_function("self_attention"):
            #                 encoder_inputs = self_attention_layer(encoder_inputs, attention_bias)
            #     with torch.cuda.amp.autocast(enabled=use_amp):
            #         with torch.autograd.profiler.record_function("ffn"):
            #             encoder_inputs = feed_forward_network(encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


class DecoderStack(nn.Module):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """

    def __init__(self, config, train):
        super(DecoderStack, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(config["num_hidden_layers"]):
            self_attention_layer = attention_layer.SelfAttention(
                config["hidden_size"], config["num_heads"],
                config["attention_dropout"], train)
            enc_dec_attention_layer = attention_layer.Attention(
                config["hidden_size"], config["num_heads"],
                config["attention_dropout"], train)
            feed_forward_network = ffn_layer.FeedFowardNetwork(
                config["hidden_size"], config["filter_size"],
                config["relu_dropout"], train, config["allow_ffn_pad"])

            self.layers.append(nn.ModuleList([
                PrePostProcessingWrapper(self_attention_layer, config, train),
                PrePostProcessingWrapper(enc_dec_attention_layer, config, train),
                PrePostProcessingWrapper(feed_forward_network, config, train)]))

        self.output_normalization = LayerNormalization(config["hidden_size"])

    def forward(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
                attention_bias, cache=None):
        """Return the output of the decoder layer stacks.

        Args:
          decoder_inputs: tensor with shape [batch_size, target_length, hidden_size]
          encoder_outputs: tensor with shape [batch_size, input_length, hidden_size]
          decoder_self_attention_bias: bias for decoder self-attention layer.
            [1, target_length, target_length]
          attention_bias: bias for encoder-decoder attention layer.
            [batch_size, 1, input_length, 1]
          cache: (Used for fast decoding) A nested dictionary storing previous
            decoder self-attention values. The items are:
              {layer_n: {"k": tensor with shape [batch_size, i, key_channels],
                         "v": tensor with shape [batch_size, i, value_channels]},
               ...}

        Returns:
          Output of decoder layer stack.
          float32 tensor with shape [batch_size, target_length, hidden_size]
        """
        for n, layer in enumerate(self.layers):
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            # Run inputs through the sublayers.
            layer_name = "layer_%d" % n
            layer_cache = cache[layer_name] if cache is not None else None
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    with torch.cuda.amp.autocast(enabled=True):
                        decoder_inputs = self_attention_layer(
                            decoder_inputs, decoder_self_attention_bias, cache=layer_cache)
                    with torch.cuda.amp.autocast(enabled=True):
                        decoder_inputs = enc_dec_attention_layer(
                            decoder_inputs, encoder_outputs, attention_bias)
                    with torch.cuda.amp.autocast(enabled=True):
                        decoder_inputs = feed_forward_network(decoder_inputs)

        return self.output_normalization(decoder_inputs)
