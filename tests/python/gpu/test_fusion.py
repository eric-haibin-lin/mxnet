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

import os
import random
import mxnet as mx
import numpy as np
from mxnet.test_utils import *
from mxnet.cuda_utils import get_sm_arch

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed

def check_fused_symbol(sym, **kwargs):
    inputs = sym.list_inputs()
    shapes = {inp : kwargs[inp].shape for inp in inputs}
    # Double identity so that there is always something to fuse
    test_sym = mx.sym.Group([mx.sym.identity(mx.sym.identity(s)) for s in sym])
    rtol = {'float16' : 1e-2,
            'float32' : 1.5e-6,
            'float64' : 1.5e-6,
            }
    atol = {'float16' : 1e-3,
            'float32' : 1e-7,
            'float64' : 1e-7,
            }
    for dtype in ['float16', 'float32', 'float64']:
        data = {inp : kwargs[inp].astype(dtype) for inp in inputs}
        for grad_req in ['write', 'add']:
            type_dict = {inp : dtype for inp in inputs}
            os.environ["MXNET_USE_FUSION"] = "0"
            orig_exec = test_sym.simple_bind(ctx=mx.gpu(0), grad_req=grad_req, type_dict=type_dict, **shapes)
            os.environ["MXNET_USE_FUSION"] = "1"
            fused_exec = test_sym.simple_bind(ctx=mx.gpu(0), grad_req=grad_req, type_dict=type_dict, **shapes)
            fwd_orig = orig_exec.forward(is_train=True, **data)
            out_grads = [mx.nd.ones_like(arr) for arr in fwd_orig]
            orig_exec.backward(out_grads=out_grads)
            fwd_fused = fused_exec.forward(is_train=True, **data)
            fused_exec.backward(out_grads=out_grads)
            for orig, fused in zip(fwd_orig, fwd_fused):
                assert_allclose(orig, fused, rtol=rtol[dtype], atol=atol[dtype])
            for orig, fused in zip(orig_exec.grad_arrays, fused_exec.grad_arrays):
                if orig is None and fused is None:
                    continue
                assert orig is not None
                assert fused is not None
                assert_allclose(orig, fused, rtol=rtol[dtype], atol=atol[dtype])

def check_unary_ops():
    unary_ops = [
            'relu',
            'sigmoid',
            'softsign',
            'exp',
            'expm1',
            'log',
            'log10',
            'log2',
            'log1p',
            'degrees',
            'radians',
            'sin',
            'cos',
            'tan',
            'arcsin',
            'arccos',
            'arctan',
            'sinh',
            'cosh',
            'tanh',
            'arcsinh',
            'arctanh',
            'sqrt',
            'rsqrt',
            'cbrt',
            'rcbrt',
            'square',
            'squeeze',
            'zeros_like',
            'ones_like',
            'flatten',
            'round',
            'rint',
            'fix',
            'floor',
            'ceil',
            'trunc',
            'sign',
            'reciprocal',
            'abs',
            'gamma',
            'gammaln',
            'erf',
            'negative',
            ]

    def announce_check(op_name):
        print("Checking fusion of " + op_name)

    arr = mx.random.uniform(shape=rand_shape_2d())
    a = mx.sym.Variable('a')
    for op_name in unary_ops:
        announce_check(op_name)
        op = getattr(mx.sym, op_name)
        sym = op(a)
        check_fused_symbol(sym, a=arr)

    # unary ops requiring special treatment

    # arccosh needs input to be >= 1
    arr2 = arr + 1
    announce_check('arccosh')
    check_fused_symbol(mx.sym.arccosh(a), a=arr2)

    # erfinv needs -1 < input < 1, but we avoid the limits of this range where the slope nears +inf.
    arr2 = (arr - 0.5) * 1.99
    announce_check('erfinv')
    check_fused_symbol(mx.sym.erfinv(a), a=arr2)

    # Activation requires act_type attribute
    for act_type in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
        announce_check("Activation(act_type='{}')".format(act_type))
        check_fused_symbol(mx.sym.Activation(a, act_type=act_type), a=arr)

    # Cast requires dtype
    for dtype in ['float16', 'float32', 'float64', 'int32']:
        announce_check("Cast(dtype='{}')".format(dtype))
        check_fused_symbol(mx.sym.Cast(a, dtype=dtype), a=arr)

    # reshape requires shape
    announce_check('reshape')
    check_fused_symbol(mx.sym.reshape(a, shape=(-1,)), a=arr)

    # expand_dims requires axis
    announce_check('expand_dims')
    check_fused_symbol(mx.sym.expand_dims(a, axis=1), a=arr)

    # clip requires a_min, a_max
    announce_check('clip')
    check_fused_symbol(mx.sym.clip(a, a_min=0.3, a_max=0.7), a=arr)

    # smooth_l1 requires a scalar
    announce_check('smooth_l1')
    check_fused_symbol(mx.sym.smooth_l1(a, scalar=0.3), a=arr)

def check_binary_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    check_fused_symbol(a+b, a=arr1, b=arr2)
    check_fused_symbol(a+3, a=arr1)
    check_fused_symbol(a-b, a=arr1, b=arr2)
    check_fused_symbol(a-3, a=arr1)
    check_fused_symbol(3-a, a=arr1)
    check_fused_symbol(a*b, a=arr1, b=arr2)
    check_fused_symbol(a*3, a=arr1)
    check_fused_symbol(a/b, a=arr1, b=arr2)
    check_fused_symbol(a/3, a=arr1)
    check_fused_symbol(3/a, a=arr1)
    check_fused_symbol(a**b, a=arr1, b=arr2)
    check_fused_symbol(a**3, a=arr1)
    check_fused_symbol(mx.sym.pow(3,a), a=arr1)
    check_fused_symbol(mx.sym.maximum(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.minimum(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.hypot(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.hypot(a,3), a=arr1)

def check_other_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = mx.sym.Variable('c')
    shape = rand_shape_2d()
    shape = (5,) + shape
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)
    arr3 = mx.random.uniform(shape=shape)

    check_fused_symbol(mx.sym.add_n(a,b,c), a=arr1, b=arr2, c=arr3)

    check_fused_symbol(mx.sym.slice_axis(a, axis=0, begin=1, end=4), a=arr1)

    begin = (random.randint(0, shape[0]-1),
             random.randint(0, shape[1]-1),
             random.randint(0, shape[2]-1))
    end = (random.randint(begin[0]+1, shape[0]),
           random.randint(begin[1]+1, shape[1]),
           random.randint(begin[2]+1, shape[2]))
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

    arr1 = mx.random.uniform(shape=(2,3,4,5))
    arr2 = mx.random.uniform(shape=(1,2,3))
    check_fused_symbol(mx.sym.slice_like(a,b, axes=[-2, 0]), a=arr1, b=arr2)

    arr1 = mx.random.uniform(shape=(1,1,2,3))
    arr2 = mx.random.uniform(shape=(2,2,2,3))
    check_fused_symbol(mx.sym.broadcast_like(a, b, lhs_axes=[0], rhs_axes=[0]), a=arr1, b=arr2)


def set_back_env_var(var_name, old_env_var):
    if old_env_var is None:
        os.environ.pop(var_name)
    else:
        os.environ[var_name] = old_env_var


def check_batch_norm_activ():
    old_env_var = os.environ.get('MXNET_DISABLE_BNACTIV_FUSION', None)
    bn_name = "batchnorm"
    data = mx.sym.Variable('data')
    rtol = 1e-2
    atol = 1e-3
    for axis in [1, 3]:
        for act_type in ['relu', 'tanh']:
            for ctx in [mx.gpu(0), mx.cpu(0)]:
                for channel_size in [11, 12]:
                    for dtype in ['float16', 'float32']:
                        print("axis: {} | act_type: {} | ctx: {} | channel_size: {} | dtype: {}".format(
                              axis, act_type, ctx, channel_size, dtype))
                        input_shape = [10, 5, 5, 5]
                        input_shape[axis] = channel_size
                        input_shape = tuple(input_shape)
                        bn = mx.sym.BatchNorm(data, axis=axis, name=bn_name)
                        act = mx.sym.Activation(bn, act_type=act_type)
                        os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '0'
                        executor = act.simple_bind(ctx, data=input_shape, grad_req='null', force_rebind=True,
                                                   type_dict={'data': dtype})
                        if (axis == 3 and act_type == 'relu' and ctx == mx.gpu(0) and
                            channel_size % 4 == 0 and dtype == 'float16'):
                            assert executor.get_optimized_symbol().name == bn_name + "_activ"
                            arg_params = {'data': mx.random.uniform(shape=input_shape, dtype=dtype),
                                          'batchnorm_gamma': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_beta': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            aux_params = {'batchnorm_moving_mean': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_moving_var': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            fused_out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            fused_out_train = executor.outputs[0].asnumpy()
                            os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '1'
                            executor = act.simple_bind(ctx, data=input_shape,
                                                       grad_req='null', force_rebind=True,
                                                       type_dict={'data': dtype})
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()))
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            out_train = executor.outputs[0].asnumpy()
                            assert_allclose(out_predict, fused_out_predict, rtol, atol)
                            assert_allclose(out_train, fused_out_train, rtol, atol)
                        else:
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()))
    os.environ['MXNET_USE_FUSION'] = '0'
    os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '0'
    bn = mx.sym.BatchNorm(data, axis=3, name=bn_name)
    act = mx.sym.Activation(bn, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), data=(10, 5, 5, 12),
                               grad_req='null', force_rebind=True,
                               type_dict={'data': dtype})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()))

    os.environ['MXNET_USE_FUSION'] = '1'
    os.environ['MXNET_DISABLE_BNACTIV_FUSION'] = '1'
    bn = mx.sym.BatchNorm(data, axis=3, name=bn_name)
    act = mx.sym.Activation(bn, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), data=(10, 5, 5, 12),
                               grad_req='null', force_rebind=True,
                               type_dict={'data': dtype})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()))

    set_back_env_var('MXNET_DISABLE_BNACTIV_FUSION', old_env_var)

def check_batch_norm_add_relu():
    old_env_var = os.environ.get('MXNET_DISABLE_BNADDRELU_FUSION', None)
    bn_name = "batchnorm"
    lhs = mx.sym.Variable('lhs')
    rhs = mx.sym.Variable('rhs')
    rtol = 1e-2
    atol = 1e-3
    for axis in [1, 3]:
        for act_type in ['relu', 'tanh']:
            for ctx in [mx.gpu(0), mx.cpu(0)]:
                for channel_size in [11, 12]:
                    for dtype in ['float16', 'float32']:
                        print("axis: {} | act_type: {} | ctx: {} | channel_size: {} | dtype: {}".format(
                              axis, act_type, ctx, channel_size, dtype))
                        input_shape = [10, 5, 5, 5]
                        input_shape[axis] = channel_size
                        input_shape = tuple(input_shape)
                        bn = mx.sym.BatchNorm(lhs, axis=axis, name=bn_name)
                        add = bn + rhs
                        act = mx.sym.Activation(add, act_type=act_type)
                        os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '0'
                        executor = act.simple_bind(ctx, lhs=input_shape, rhs=input_shape,
                                                   grad_req='null', force_rebind=True,
                                                   type_dict={'lhs': dtype, 'rhs': dtype})
                        if (axis == 3 and act_type == 'relu' and ctx == mx.gpu(0) and
                            channel_size % 4 == 0 and dtype == 'float16'):
                            assert executor.get_optimized_symbol().name == bn_name + "_add_relu"
                            arg_params = {'lhs': mx.random.uniform(shape=input_shape, dtype=dtype),
                                          'rhs': mx.random.uniform(shape=input_shape, dtype=dtype),
                                          'batchnorm_gamma': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_beta': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            aux_params = {'batchnorm_moving_mean': mx.random.uniform(shape=(channel_size,), dtype='float32'),
                                          'batchnorm_moving_var': mx.random.uniform(shape=(channel_size,), dtype='float32')}
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            fused_out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            fused_out_train = executor.outputs[0].asnumpy()
                            os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '1'
                            executor = act.simple_bind(ctx, lhs=input_shape, rhs=input_shape,
                                                       grad_req='null', force_rebind=True,
                                                       type_dict={'lhs': dtype, 'rhs': dtype})
                            print(executor.get_optimized_symbol().get_internals())
                            print(act.get_internals())
                            if ctx == mx.gpu(0):  # Still have pointwise fusion
                                assert (len(executor.get_optimized_symbol().get_internals()) ==
                                        len(act.get_internals()) - 1)
                            else:
                                assert(len(executor.get_optimized_symbol().get_internals()) ==
                                       len(act.get_internals()))
                            executor.copy_params_from(arg_params, aux_params)
                            executor.forward(is_train=False)
                            out_predict = executor.outputs[0].asnumpy()
                            executor.forward(is_train=True)
                            out_train = executor.outputs[0].asnumpy()
                            assert_allclose(out_predict, fused_out_predict, rtol, atol)
                            assert_allclose(out_train, fused_out_train, rtol, atol)
                        elif ctx == mx.gpu(0):  # Still have pointwise fusion
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()) - 1)
                        else:
                            assert (len(executor.get_optimized_symbol().get_internals()) ==
                                    len(act.get_internals()))
    input_shape=(10, 5, 5, 12)
    os.environ['MXNET_USE_FUSION'] = '1'
    os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '0'
    bn = mx.sym.BatchNorm(rhs, axis=3, name=bn_name)
    add = lhs + bn
    act = mx.sym.Activation(add, act_type='relu')

    executor = act.simple_bind(mx.gpu(0), lhs=input_shape, rhs=input_shape,
                               grad_req='null', force_rebind=True,
                               type_dict={'lhs': 'float16', 'rhs': 'float16'})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()) - 1)

    os.environ['MXNET_USE_FUSION'] = '0'
    os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '0'
    bn = mx.sym.BatchNorm(lhs, axis=3, name=bn_name)
    add = bn + rhs
    act = mx.sym.Activation(add, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), lhs=input_shape, rhs=input_shape,
                               grad_req='null', force_rebind=True,
                               type_dict={'lhs': 'float16', 'rhs': 'float16'})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()))

    os.environ['MXNET_USE_FUSION'] = '1'
    os.environ['MXNET_DISABLE_BNADDRELU_FUSION'] = '1'
    bn = mx.sym.BatchNorm(lhs, axis=3, name=bn_name)
    add = bn + rhs
    act = mx.sym.Activation(add, act_type='relu')
    executor = act.simple_bind(mx.gpu(0), lhs=input_shape, rhs=input_shape,
                               grad_req='null', force_rebind=True,
                               type_dict={'lhs': 'float16', 'rhs': 'float16'})
    assert (len(executor.get_optimized_symbol().get_internals()) ==
            len(act.get_internals()) - 1)

    set_back_env_var('MXNET_DISABLE_BNADDRELU_FUSION', old_env_var)

def check_norm_convolution():
    old_env_var = os.environ.get('MXNET_DISABLE_NORMCONV_FUSION', None)
    conv0_name = "conv0"
    conv1_name = "conv1"
    kernel = (3, 3)
    stride = (1, 1)
    bn_name = "batchnorm"
    data = mx.sym.Variable('data')
    rtol = 1e-2
    # We don't really check all the conditions because we trust the Supports function of the Op
    # Which are tested in unit tests of norm_convolution
    for layout, axis in [('NCHW', 1), ('NHWC', 3)]:
        for no_bias in [True, False]:
            for act_type in [None, 'relu']:
                atol = 1e-1 if act_type == 'relu' else 1.
                for ctx in [mx.gpu(0), mx.cpu(0)]:
                    for channel_size in [15, 32]:
                        for dtype in ['float16', 'float32']:
                            print("axis: {} | act_type: {} | ctx: {} | channel_size: {} | dtype: {}".format(
                                  axis, act_type, ctx, channel_size, dtype))
                            input_shape = [10, 5, 5, 5]
                            input_shape[axis] = channel_size
                            input_shape = tuple(input_shape)
                            conv = mx.sym.Convolution(data, kernel=kernel, stride=stride, layout=layout,
                                                      num_filter=channel_size * 2, no_bias=no_bias,
                                                      name=conv0_name)
                            bn = mx.sym.BatchNorm(conv, axis=axis, act_type=act_type, name=bn_name)
                            conv = mx.sym.Convolution(bn, kernel=kernel, stride=stride, layout=layout,
                                                      num_filter=channel_size, no_bias=no_bias, name=conv1_name)
                            os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '0'
                            executor = conv.simple_bind(ctx, data=input_shape,
                                                        grad_req='write', force_rebind=True,
                                                        type_dict={'data': dtype})
                            if (axis == 3 and no_bias and channel_size % 32 == 0 and
                                ctx == mx.gpu(0) and get_sm_arch(ctx.device_id) == 70 and
                                dtype == 'float16'):
                                assert executor.get_optimized_symbol().get_internals()[2].name == conv0_name + "_normalized"
                                assert executor.get_optimized_symbol().name == conv1_name + "_normalized"
                                arg_params = {'data': mx.random.uniform(shape=input_shape, dtype=dtype),
                                              'conv0_weight': mx.random.uniform(shape=(channel_size * 2,) + kernel + (channel_size,), dtype=dtype),
                                              'conv1_weight': mx.random.uniform(shape=(channel_size,) + kernel + (channel_size * 2,), dtype=dtype),
                                              'batchnorm_gamma': mx.random.uniform(shape=(channel_size * 2,), dtype='float32'),
                                              'batchnorm_beta': mx.random.uniform(shape=(channel_size * 2,), dtype='float32')}
                                aux_params = {'batchnorm_moving_mean': mx.random.uniform(shape=(channel_size * 2,), dtype='float32'),
                                              'batchnorm_moving_var': mx.random.uniform(shape=(channel_size * 2,), dtype='float32')}
                                executor.copy_params_from(arg_params, aux_params)
                                executor.forward(is_train=False)
                                fused_out_predict = executor.outputs[0].asnumpy()
                                executor.forward(is_train=True)
                                fused_out_train = executor.outputs[0].asnumpy()
                                os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '1'
                                executor = conv.simple_bind(ctx, data=input_shape,
                                                           grad_req='null', force_rebind=True,
                                                           type_dict={'data': dtype})
                                assert (len(executor.get_optimized_symbol().get_internals()) ==
                                        len(conv.get_internals()))
                                executor.copy_params_from(arg_params, aux_params)
                                executor.forward(is_train=False)
                                out_predict = executor.outputs[0].asnumpy()
                                executor.forward(is_train=True)
                                out_train = executor.outputs[0].asnumpy()
                                assert_allclose(out_predict, fused_out_predict, rtol, atol)
                                assert_allclose(out_train, fused_out_train, rtol, atol)
                            else:
                                assert (len(executor.get_optimized_symbol().get_internals()) ==
                                        len(conv.get_internals()))
        os.environ['MXNET_USE_FUSION'] = '0'
        os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '0'
        input_shape = (10, 5, 5, 32)
        channel_size = 32
        conv = mx.sym.Convolution(data, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size * 2, no_bias=True,
                                  name=conv0_name)
        bn = mx.sym.BatchNorm(conv, axis=axis, act_type=act_type, name=bn_name)
        conv = mx.sym.Convolution(bn, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size, no_bias=True, name=conv1_name)
        executor = conv.simple_bind(ctx, data=input_shape,
                                    grad_req='write', force_rebind=True,
                                    type_dict={'data': 'float16'})
        assert (len(executor.get_optimized_symbol().get_internals()) ==
                len(conv.get_internals()))

        os.environ['MXNET_USE_FUSION'] = '1'
        os.environ['MXNET_DISABLE_NORMCONV_FUSION'] = '1'
        input_shape = (10, 5, 5, 32)
        channel_size = 32
        conv = mx.sym.Convolution(data, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size * 2, no_bias=True,
                                  name=conv0_name)
        bn = mx.sym.BatchNorm(conv, axis=axis, act_type=act_type, name=bn_name)
        conv = mx.sym.Convolution(bn, kernel=kernel, stride=stride, layout=layout,
                                  num_filter=channel_size, no_bias=True, name=conv1_name)
        executor = conv.simple_bind(ctx, data=input_shape,
                                    grad_req='write', force_rebind=True,
                                    type_dict={'data': 'float16'})
        assert (len(executor.get_optimized_symbol().get_internals()) ==
                len(conv.get_internals()))
        set_back_env_var('MXNET_DISABLE_NORMCONV_FUSION', old_env_var)

@with_seed()
def test_fusion():
    old_mxnet_use_fusion = os.environ.get('MXNET_USE_FUSION', None)
    os.environ['MXNET_USE_FUSION'] = '1'

    check_unary_ops()
    check_binary_ops()
    check_other_ops()
    check_batch_norm_activ()
    check_batch_norm_add_relu()
    check_norm_convolution()

    set_back_env_var('MXNET_USE_FUSION', old_mxnet_use_fusion)

if __name__ == '__main__':
    import nose
    nose.runmodule()
