# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import, arguments-differ
r"""Module for pre-defined neural network models.

This module contains definitions for the following model architectures:
-  `ResNet V1`_
-  `ResNet V2`_

.. _ResNet V1: https://arxiv.org/abs/1512.03385
.. _ResNet V2: https://arxiv.org/abs/1603.05027
"""

from networks.resnet import *
from networks.resnext import *

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    classes : int
        Number of classes for the output layer.

    Returns
    -------
    HybridBlock
        The model.
    """
    models = {'resnet18_v1': resnet18_v1,
              'resnet34_v1': resnet34_v1,
              'resnet50_v1': resnet50_v1,
              'resnet101_v1': resnet101_v1,
              'resnet152_v1': resnet152_v1,
              'resnet18_v1b': resnet18_v1b,
              'resnet34_v1b': resnet34_v1b,
              'resnet50_v1b': resnet50_v1b,
              'resnet101_v1b': resnet101_v1b,
              'resnet152_v1b': resnet152_v1b,
              'resnet18_v2': resnet18_v2,
              'resnet34_v2': resnet34_v2,
              'resnet50_v2': resnet50_v2,
              'resnet101_v2': resnet101_v2,
              'resnet152_v2': resnet152_v2,
              'resnext50_32x4d': resnext50_32x4d,
              'resnext101_32x4d': resnext101_32x4d,
              'resnext101_64x4d': resnext101_64x4d,
              'se_resnext50_32x4d': se_resnext50_32x4d,
              'se_resnext101_32x4d': se_resnext101_32x4d,
              'se_resnext101_64x4d': se_resnext101_64x4d,
             }
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s' % (
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)
