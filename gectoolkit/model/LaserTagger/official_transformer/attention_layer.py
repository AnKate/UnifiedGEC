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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train=True):
        if hidden_size % num_heads != 0:
            raise ValueError(
                "Hidden size must be evenly divisible by the number of heads."
            )

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)

        self.output_dense_layer = nn.Linear(hidden_size, hidden_size, bias=False)

    def split_heads(self, x):
        batch_size, length, _ = x.size()
        depth = self.hidden_size // self.num_heads
        x = x.view(batch_size, length, self.num_heads, depth)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, length, _ = x.size()
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, length, self.hidden_size)

    def forward(self, x, y, bias, cache=None):
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        if cache is not None:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
            cache["k"] = k
            cache["v"] = v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = self.hidden_size // self.num_heads
        q *= depth ** -0.5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logits = torch.matmul(q, k.transpose(-1, -2))
        logits += bias.to(device)
        weights = torch.softmax(logits, dim=-1)
        if self.train:
            weights = nn.functional.dropout(weights, self.attention_dropout)

        attention_output = torch.matmul(weights, v)
        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def forward(self, x, bias, cache=None):
        return super(SelfAttention, self).forward(x, x, bias, cache)
