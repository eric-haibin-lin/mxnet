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
# pylint: disable= arguments-differ
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNetV1', 'ResNetV2',
           'BasicBlockV1', 'BasicBlockV2',
           'BottleneckV1', 'BottleneckV2',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b', 'resnet101_v1b', 'resnet152_v1b',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'get_resnet']

import os
import sys

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import base

sys.path.insert(0, "..")
from common.helper import *

# Helpers
def _conv3x3(channels, stride, in_channels, layout):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels, layout=layout)


# Blocks
class BasicBlockV1(HybridBlock):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 conv_layout='NCHW', bn_layout='NCHW', variant='a', **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels, layout=conv_layout))
        self.body.add(batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout))
        self.body.add(_conv3x3(channels, 1, channels, layout=conv_layout))
        self.body.add(batchnorm(io_layout=conv_layout, bn_layout=bn_layout))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels,
                                          layout=conv_layout))
            self.downsample.add(batchnorm(io_layout=conv_layout, bn_layout=bn_layout))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x


class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 conv_layout='NCHW', bn_layout='NCHW', variant='a', **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        stride1 = stride if variant == 'a' else 1
        stride2 = 1 if variant == 'a' else stride
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=stride1, use_bias=False, layout=conv_layout))
        self.body.add(batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout))
        self.body.add(_conv3x3(channels//4, stride2, channels//4, layout=conv_layout))
        self.body.add(batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False, layout=conv_layout))
        self.body.add(batchnorm(io_layout=conv_layout, bn_layout=bn_layout))
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels,
                                          layout=conv_layout))
            self.downsample.add(batchnorm(io_layout=conv_layout, bn_layout=bn_layout))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


class BasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 conv_layout='NCHW', bn_layout='NCHW', **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout)
        self.conv1 = _conv3x3(channels, stride, in_channels, layout=conv_layout)
        self.bn2 = batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout)
        self.conv2 = _conv3x3(channels, 1, channels, layout=conv_layout)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels, layout=conv_layout)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.conv2(x)

        return x + residual


class BottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 conv_layout='NCHW', bn_layout='NCHW', **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout)
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False, layout=conv_layout)
        self.bn2 = batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout)
        self.conv2 = _conv3x3(channels//4, stride, channels//4, layout=conv_layout)
        self.bn3 = batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout)
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False, layout=conv_layout)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels, layout=conv_layout)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.conv3(x)

        return x + residual


# Nets
class ResNetV1(HybridBlock):
    r"""ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, input_layout,
                 conv_layout, bn_layout, pooling_layout,
                 classes=1000, thumbnail=False, resnet_variant='a', **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        self.variant = resnet_variant
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(Transpose(input_layout, conv_layout))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0, layout=conv_layout))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, layout=conv_layout))
                self.features.add(batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout))
                self.features.add(max_pool(3, 2, 1, io_layout=conv_layout, pooling_layout=pooling_layout))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i],
                                                   conv_layout=conv_layout, bn_layout=bn_layout))
            self.features.add(global_avg_pool(io_layout=conv_layout, pooling_layout=pooling_layout))

            self.output = nn.Dense(classes, in_units=channels[-1])

    def _make_layer(self, block, layers, channels, stride, stage_index,
                    in_channels=0, conv_layout='NCHW', bn_layout='NCHW'):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            conv_layout=conv_layout, bn_layout=bn_layout,
                            variant=self.variant, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                conv_layout=conv_layout, bn_layout=bn_layout,
                                variant=self.variant, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


class ResNetV2(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, input_layout,
                 conv_layout, bn_layout, pooling_layout, classes=1000, thumbnail=False, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(Transpose(input_layout, conv_layout))
            self.features.add(batchnorm(io_layout=conv_layout, bn_layout=bn_layout, scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 0, conv_layout))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, layout=conv_layout))
                self.features.add(batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout))
                self.features.add(max_pool(3, 2, 1, io_layout=conv_layout,
                                           pooling_layout=pooling_layout))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels,
                                                   conv_layout=conv_layout,
                                                   bn_layout=bn_layout))
                in_channels = channels[i+1]
            self.features.add(batchnorm_relu(io_layout=conv_layout, bn_layout=bn_layout))
            self.features.add(global_avg_pool(io_layout=conv_layout, pooling_layout=pooling_layout))
            self.features.add(nn.Flatten())

            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0, conv_layout='NCHW', bn_layout='NCHW'):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            conv_layout=conv_layout, bn_layout=bn_layout, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                conv_layout=conv_layout, bn_layout=bn_layout, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Specification
resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


# Constructor
def get_resnet(version, num_layers, input_layout='NCHW', conv_layout='NCHW',
               bn_layout='NCHW', pooling_layout='NCHW', **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    input_layout : str, default 'NCHW'
        Dimension ordering of the input data
    conv_layout : str, default 'NCHW'
        Dimension ordering of convolution layers
    bn_layout : str, default 'NCHW'
        Dimension ordering of the BatchNorm layers
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert version >= 1 and version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2."%version
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, input_layout,
                       conv_layout, bn_layout, pooling_layout, **kwargs)
    return net

def resnet18_v1(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(1, 18, **kwargs)

def resnet34_v1(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(1, 34, **kwargs)

def resnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(1, 50, **kwargs)

def resnet101_v1(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(1, 101, **kwargs)

def resnet152_v1(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(1, 152, **kwargs)

def resnet18_v1b(**kwargs):
    r"""ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    kwargs['resnet_variant'] = 'b'
    return get_resnet(1, 18, **kwargs)

def resnet34_v1b(**kwargs):
    r"""ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    kwargs['resnet_variant'] = 'b'
    return get_resnet(1, 34, **kwargs)

def resnet50_v1b(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    kwargs['resnet_variant'] = 'b'
    return get_resnet(1, 50, **kwargs)

def resnet101_v1b(**kwargs):
    r"""ResNet-101 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    kwargs['resnet_variant'] = 'b'
    return get_resnet(1, 101, **kwargs)

def resnet152_v1b(**kwargs):
    r"""ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    """
    kwargs['resnet_variant'] = 'b'
    return get_resnet(1, 152, **kwargs)

def resnet18_v2(**kwargs):
    r"""ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(2, 18, **kwargs)

def resnet34_v2(**kwargs):
    r"""ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(2, 34, **kwargs)

def resnet50_v2(**kwargs):
    r"""ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(2, 50, **kwargs)

def resnet101_v2(**kwargs):
    r"""ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(2, 101, **kwargs)

def resnet152_v2(**kwargs):
    r"""ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    """
    return get_resnet(2, 152, **kwargs)
