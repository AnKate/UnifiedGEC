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
"""Implementation of embedding layer with shared weights."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class EmbeddingSharedWeights(nn.Module):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    # def __init__(self, vocab_size, hidden_size):
    def __init__(self, config):
        """Specify characteristic parameters of embedding layer.

        Args:
          vocab_size: Number of tokens in the embedding. (Typically ~32,000)
          hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__()
        # self.vocab_size = vocab_size
        # self.hidden_size = hidden_size
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]

        # with torch.no_grad():
        # print("type:", type(self.vocab_size), type(self.hidden_size))
        self.shared_weights = nn.Parameter(
            torch.randn((self.vocab_size, self.hidden_size), requires_grad=True))  # [vocab_size, hidden_size]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.shared_weights = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=0).to(device)
        # nn.init.normal_(self.shared_weights.weight, mean=0.0, std=self.hidden_size ** -0.5)

    def forward(self, x):
        """Get token embeddings of x.

        Args:
          x: An int64 tensor with shape [batch_size, length]
        Returns:
          embeddings: float32 tensor with shape [batch_size, length, embedding_size]
          padding: float32 tensor with shape [batch_size, length] indicating the
            locations of the padding tokens in x.
        """
        # mask = (x != self.pad_index).float()
        #
        # embeddings = self.shared_weights(x)
        # embeddings *= mask.unsqueeze(-1)
        #
        # embeddings *= self.hidden_size ** 0.5
        #
        # return embeddings

        # Create binary mask of size [batch_size, length]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        mask = torch.unsqueeze(torch.where(x != 0, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device)),
                               dim=-1).float()  # [batch_size, length, 1]

        shared_weights = torch.Tensor(self.shared_weights).to(device)
        # print(shared_weights.shape, x.shape)
        batch_size, seq_length = x.shape
        embeddings = shared_weights.index_select(0, x.view(-1)).view(batch_size, seq_length, self.hidden_size)
        # print("embeddings", embeddings.shape, mask.shape)

        embeddings *= mask
        # Scale embedding by the sqrt of the hidden size
        embeddings *= self.hidden_size ** 0.5

        return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
          x: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        # with torch.name_scope("presoftmax_linear"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = x.shape[0]
        length = x.shape[1]

        x = x.reshape([-1, self.hidden_size])
        logits = torch.matmul(x, self.shared_weights.T.to(device))

        return logits.reshape([batch_size, length, self.vocab_size])
