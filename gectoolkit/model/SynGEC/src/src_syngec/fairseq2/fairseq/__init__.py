# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

__all__ = ["pdb"]
__version__ = "0.10.2"

import sys
import os
# print(os.getcwd()) # /home/xiaoman/project/gectoolkit2

# 获取 fairseq 所在的绝对路径
fairseq_path = os.path.abspath(os.path.join(os.getcwd(), "gectoolkit/model/SynGEC/src/src_syngec/fairseq2"))

# 将路径添加到 sys.path
sys.path.insert(0, fairseq_path)

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from fairseq.logging import meters, metrics, progress_bar  # noqa

sys.modules["fairseq.meters"] = meters
sys.modules["fairseq.metrics"] = metrics
sys.modules["fairseq.progress_bar"] = progress_bar

import fairseq.criterions  # noqa
import fairseq.models  # noqa
import fairseq.modules  # noqa
import fairseq.optim  # noqa
import fairseq.optim.lr_scheduler  # noqa
import fairseq.pdb  # noqa
import fairseq.scoring  # noqa
import fairseq.tasks  # noqa
import fairseq.token_generation_constraints  # noqa

import fairseq.benchmark  # noqa
import fairseq.model_parallel  # noqa
