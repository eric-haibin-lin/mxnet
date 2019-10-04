import mxnet as mx
import numpy as np
from mxnet.gluon import nn, HybridBlock

_CONFIG_DTYPE='float32'

class Transpose(HybridBlock):
    def __init__(self,from_layout, to_layout, **kwargs):
        super(Transpose, self).__init__(**kwargs)
        supported_layouts = ['NCHW', 'NHWC']
        if from_layout not in supported_layouts:
            raise ValueError('Not prepared to handle layout: {}'.format(from_layout))
        if to_layout not in supported_layouts:
            raise ValueError('Not prepared to handle layout: {}'.format(to_layout))
        self.from_layout = from_layout
        self.to_layout = to_layout

    def hybrid_forward(self, F, x):
        # Insert transpose if from_layout and to_layout don't match
        if self.from_layout == 'NCHW' and self.to_layout == 'NHWC':
            return F.transpose(x, axes=(0, 2, 3, 1))
        elif self.from_layout == 'NHWC' and self.to_layout == 'NCHW':
            return F.transpose(x, axes=(0, 3, 1, 2))
        else:
            return x

    def __repr__(self):
        s = '{name}({content})'
        if self.from_layout == self.to_layout:
            content = 'passthrough ' + self.from_layout
        else:
            content = self.from_layout + ' -> ' + self.to_layout
        return s.format(name=self.__class__.__name__,
                        content=content)

class LayoutWrapper(HybridBlock):
    def __init__(self, op, io_layout, op_layout, **kwargs):
        super(LayoutWrapper, self).__init__(**kwargs)
        with self.name_scope():
            self.net = nn.HybridSequential()
            self.net.add(Transpose(io_layout, op_layout))
            self.net.add(op)
            self.net.add(Transpose(op_layout, io_layout))

    def hybrid_forward(self, F, x):
        return self.net(x)

# A BatchNorm wrapper that responds to the input layout
def batchnorm(*args, io_layout='NCHW', bn_layout='NCHW', **kwargs):
    bn_axis = 3 if bn_layout == 'NHWC' else 1
    kwargs['axis'] = bn_axis
    return LayoutWrapper(nn.BatchNorm(*args, **kwargs), io_layout, bn_layout)

# A BatchNorm+ReLU wrapper that responds to the input layout
def batchnorm_relu(*args, io_layout='NCHW', bn_layout='NCHW', **kwargs):
    bn_axis = 3 if bn_layout == 'NHWC' else 1
    kwargs['axis'] = bn_axis
    if bn_layout == 'NHWC' and _CONFIG_DTYPE == 'float16':
        kwargs['act_type'] = 'relu'
        op = nn.BatchNorm(*args, **kwargs)
    else:
        op = nn.HybridSequential()
        op.add(nn.BatchNorm(*args, **kwargs))
        op.add(nn.Activation('relu'))
    return LayoutWrapper(op, io_layout, bn_layout)

# A BatchNorm+ReLU wrapper that responds to the input layout
def batchnorm_relu(*args, io_layout='NCHW', bn_layout='NCHW', **kwargs):
    bn_axis = 3 if bn_layout == 'NHWC' else 1
    kwargs['axis'] = bn_axis
    if bn_layout == 'NHWC' and _CONFIG_DTYPE == 'float16':
        kwargs['act_type'] = 'relu'
        op = nn.BatchNorm(*args, **kwargs)
    else:
        op = nn.HybridSequential()
        op.add(nn.BatchNorm(*args, **kwargs))
        op.add(nn.Activation('relu'))
    return LayoutWrapper(op, io_layout, bn_layout)

# A Pooling wrapper that responds to the input layout
def max_pool(*args, io_layout='NCHW', pooling_layout='NCHW', **kwargs):
    kwargs['layout'] = pooling_layout
    return LayoutWrapper(nn.MaxPool2D(*args, **kwargs), io_layout, pooling_layout)

def global_avg_pool(*args, io_layout='NCHW', pooling_layout='NCHW', **kwargs):
    kwargs['layout'] = pooling_layout
    return LayoutWrapper(nn.GlobalAvgPool2D(*args, **kwargs), io_layout, pooling_layout)

def LRSchedule(base_lr, step, epoch_size, batch_size, warmup_length, steps, step_size):
    steps_per_epoch = epoch_size // batch_size
    if step <= warmup_length * steps_per_epoch:
        return base_lr * step / (warmup_length * steps_per_epoch)
    else:
        epoch = step / steps_per_epoch
        return base_lr * (step_size ** int(np.sum(np.array(steps) <= epoch)))
