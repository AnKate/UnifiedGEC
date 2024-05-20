#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension
import sys


if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required for fairseq.')



extensions = [
    cpp_extension.CppExtension(
        'libnat',
        sources=[
            'clib/libnat/edit_dist.cpp',
        ],
    )
]
# gectoolkit/model/LevenshteinTransformer/

setup(
    name = 'extension',
    ext_modules=extensions,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)