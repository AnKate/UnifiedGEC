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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class FeedFowardNetwork(nn.Module):
    """Fully connected feedforward network."""

    def __init__(self, hidden_size, filter_size, relu_dropout, train, allow_pad):
        super(FeedFowardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train
        self.allow_pad = allow_pad

        # print("type:", type(hidden_size), type(filter_size))
        self.filter_dense_layer = nn.Linear(hidden_size, filter_size)
        self.output_dense_layer = nn.Linear(filter_size, hidden_size)

    def forward(self, x, padding=None):
        """Return outputs of the feedforward network.

        Args:
          x: tensor with shape [batch_size, length, hidden_size]
          padding: (optional) If set, the padding values are temporarily removed
            from x (provided self.allow_pad is set). The padding values are placed
            back in the output tensor in the same locations.
            shape [batch_size, length]

        Returns:
          Output of the feedforward network.
          tensor with shape [batch_size, length, hidden_size]
        """
        padding = None if not self.allow_pad else padding

        # Retrieve dynamically known shapes
        batch_size = x.size(0)
        length = x.size(1)

        if padding is not None:
            with torch.no_grad():
                # Flatten padding to [batch_size*length]
                pad_mask = padding.view(-1)
                nonpad_ids = pad_mask.nonzero().squeeze()

                # Reshape x to [batch_size*length, hidden_size] to remove padding
                x = x.reshape(-1, self.hidden_size)
                x = torch.index_select(x, 0, nonpad_ids)

                # Reshape x from 2 dimensions to 3 dimensions.
                x = x.unsqueeze(0)
                x.requires_grad = True

        output = self.filter_dense_layer(x)
        if self.train:
            output = nn.functional.dropout(output, p=self.relu_dropout)
        output = self.output_dense_layer(output)

        if padding is not None:
            with torch.no_grad():
                output = output.squeeze(0)
                output_shape = [batch_size * length, self.hidden_size]
                output = torch.zeros(output_shape, device=output.device).index_add(
                    dim=0, index=nonpad_ids, source=output.squeeze(0)
                )
                output = output.view(batch_size, length, self.hidden_size)

        return output
