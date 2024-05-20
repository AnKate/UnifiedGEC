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
"""Transformer model helper methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch

# Very low numbers to represent -infinity. We do not actually use -Inf, since we
# want to be able to multiply these values by zero to get zero. (-Inf * 0 = NaN)
_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = -6e4
# _NEG_INF_FP16 = np.finfo(np.float16).min


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e2):
    position = torch.arange(length, dtype=torch.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    return signal


def get_decoder_self_attention_bias(length, dtype=torch.float32):
    """Calculate bias for decoder that maintains model's autoregressive property.

    Creates a tensor that masks out locations that correspond to illegal
    connections, so prediction at position i cannot draw information from future
    positions.

    Args:
      length: int length of sequences in batch.
      dtype: The dtype of the return value.

    Returns:
      float tensor of shape [1, 1, length, length]
    """
    neg_inf = _NEG_INF_FP16 if dtype == torch.float16 else _NEG_INF_FP32
    with torch.no_grad():
        valid_locs = torch.tril(torch.ones(length, length, dtype=dtype),
                                diagonal=0)  # Lower triangular part is 1, other is 0
        decoder_bias = neg_inf * (1.0 - valid_locs.unsqueeze(0).unsqueeze(0))
    return decoder_bias


def get_padding(x, padding_value=0, dtype=torch.float32):
    """Return float tensor representing the padding values in x.

    Args:
      x: int tensor with any shape
      padding_value: int value that
      dtype: The dtype of the return value.

    Returns:
      float tensor with same shape as x containing values 0 or 1.
        0 -> non-padding, 1 -> padding
    """
    with torch.no_grad():
        return torch.eq(x, padding_value).type(dtype)


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
      x: int tensor with shape [batch_size, length]

    Returns:
      Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with torch.no_grad():
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF_FP32
        attention_bias = attention_bias.unsqueeze(1).unsqueeze(1)
    return attention_bias
