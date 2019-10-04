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

from __future__ import print_function
import sys
import os
import mxnet as mx
import numpy as np
from mxnet.test_utils import assert_allclose

def convert_weight(F, q_weight, k_weight, v_weight, num_heads):
    q_weight = F.reshape(q_weight, shape=(num_heads, -1, 0), reverse=True)
    k_weight = F.reshape(k_weight, shape=(num_heads, -1, 0), reverse=True)
    v_weight = F.reshape(v_weight, shape=(num_heads, -1, 0), reverse=True)
    all_weights = F.concat(q_weight, k_weight, v_weight, dim=-2)
    all_weights = F.reshape(all_weights, shape=(-1, 0), reverse=True)
    return all_weights

def convert_bias(F, q_bias, k_bias, v_bias, num_heads):
    q_bias = F.reshape(q_bias, shape=(num_heads, -1))
    k_bias = F.reshape(k_bias, shape=(num_heads, -1))
    v_bias = F.reshape(v_bias, shape=(num_heads, -1))
    all_bias = F.stack(q_bias, k_bias, v_bias, axis=1)
    all_bias = F.reshape(all_bias, shape=(-1,))
    return all_bias

def test_multihead_attention_selfatt():
    batch_size = 2
    qkv_length = 7  # length of a sequence
    qkv_dim = 9     # dimension of encoding
    num_heads = 3   # number of attention head
    head_dim = 5    # head size
    out_dim = 13 * num_heads
    qkv_units = num_heads * head_dim

    arg_params = {
        'qkv': mx.nd.array(np.random.rand(*(batch_size, qkv_length, qkv_dim)).astype('float16') * 0.1, dtype='float16'),
        'q_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype('float16') * 0.1, dtype='float16'),
        'k_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype('float16') * 0.1, dtype='float16'),
        'v_weight': mx.nd.array(np.random.rand(*(qkv_units, qkv_dim)).astype('float16') * 0.1, dtype='float16'),
        'q_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype('float16') * 0.1, dtype='float16'),
        'k_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype('float16') * 0.1, dtype='float16'),
        'v_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype('float16') * 0.1, dtype='float16'),
        'out_weight': mx.nd.array(np.random.rand(*(out_dim, qkv_units)).astype('float16') * 0.1, dtype='float16'),
        'out_bias': mx.nd.array(np.random.rand(*(out_dim,)).astype('float16') * 0.1, dtype='float16'),
        }

    qkv = mx.sym.Variable('qkv')
    sonde = mx.sym.Variable('sonde')
    q_weight = mx.sym.Variable('q_weight')
    k_weight = mx.sym.Variable('k_weight')
    v_weight = mx.sym.Variable('v_weight')
    q_bias = mx.sym.Variable('q_bias')
    k_bias = mx.sym.Variable('k_bias')
    v_bias = mx.sym.Variable('v_bias')
    out_weight = mx.sym.Variable('out_weight')
    out_bias = mx.sym.Variable('out_bias')
    qkv_weight = convert_weight(mx.sym, q_weight, k_weight, v_weight, num_heads)
    qkv_bias = convert_bias(mx.sym, q_bias, k_bias, v_bias, num_heads)
    qkv = mx.sym.transpose(qkv, axes=(1, 0, 2))
    qkv_proj = mx.sym.FullyConnected(qkv, weight=qkv_weight, bias=qkv_bias, flatten=False,
                                     num_hidden=qkv_units * 3, no_bias=False)
    att_score = mx.sym.interleaved_matmul_selfatt_qk(qkv_proj, heads=num_heads)
    att_score = att_score + sonde
    weighted_value = mx.sym.interleaved_matmul_selfatt_valatt(qkv_proj, att_score, heads=num_heads)
    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
                                   num_hidden=out_dim, no_bias=False)
    output = mx.sym.transpose(output, axes=(1, 0, 2))
    executor = output.simple_bind(ctx=mx.gpu(0),
                                  qkv=(batch_size, qkv_length, qkv_dim),
                                  q_weight=(qkv_units, qkv_dim),
                                  q_bias=(qkv_units,),
                                  k_weight=(qkv_units, qkv_dim),
                                  k_bias=(qkv_units,),
                                  v_weight=(qkv_units, qkv_dim),
                                  v_bias=(qkv_units,),
                                  type_dict={'qkv': 'float16',
                                             'q_weight': 'float16',
                                             'q_bias': 'float16',
                                             'k_weight': 'float16',
                                             'k_bias': 'float16',
                                             'v_weight': 'float16',
                                             'v_bias': 'float16',
                                              },
                                  grad_req='write', force_rebind=True)
    output_shape = executor.outputs[0].shape
    output_grads = np.random.rand(*output_shape).astype('float16') * 0.1
    executor.copy_params_from(arg_params, {})
    executor.arg_dict['sonde'][:] = 0.
    executor.arg_dict['sonde'].wait_to_read()
    executor.forward(is_train=True)
    output_opti = executor.outputs[0].asnumpy()
    executor.backward(mx.nd.array(output_grads, dtype='float16'))
    grads_opti = {k: v.asnumpy() for k, v in executor.grad_dict.items()}

    qkv = mx.sym.Variable('qkv')
    sonde = mx.sym.Variable('sonde')
    q_weight = mx.sym.Variable('q_weight')
    k_weight = mx.sym.Variable('k_weight')
    v_weight = mx.sym.Variable('v_weight')
    q_bias = mx.sym.Variable('q_bias')
    k_bias = mx.sym.Variable('k_bias')
    v_bias = mx.sym.Variable('v_bias')
    out_weight = mx.sym.Variable('out_weight')
    out_bias = mx.sym.Variable('out_bias')

    q = mx.sym.FullyConnected(qkv, weight=q_weight, bias=q_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    k = mx.sym.FullyConnected(qkv, weight=k_weight, bias=k_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    v = mx.sym.FullyConnected(qkv, weight=v_weight, bias=v_bias, flatten=False,
                              num_hidden=qkv_units, no_bias=False)
    q = mx.sym.reshape(q, shape=(0, 0, num_heads, -1))
    q = mx.sym.transpose(q, axes=(0, 2, 1, 3))
    q = mx.sym.reshape(q, shape=(-1, 0, 0), reverse=True)
    k = mx.sym.reshape(k, shape=(0, 0, num_heads, -1))
    k = mx.sym.transpose(k, axes=(0, 2, 1, 3))
    k = mx.sym.reshape(k, shape=(-1, 0, 0), reverse=True)
    q = mx.sym.contrib.div_sqrt_dim(q)
    att_score = mx.sym.batch_dot(q, k, transpose_b=True)
    att_score = att_score + sonde
    v = mx.sym.reshape(v, shape=(0, 0, num_heads, -1))
    v = mx.sym.transpose(v, axes=(0, 2, 1, 3))
    v = mx.sym.reshape(v, shape=(-1, 0, 0), reverse=True)
    weighted_value = mx.sym.batch_dot(att_score, v)
    weighted_value = mx.sym.reshape(weighted_value, shape=(-1, num_heads, 0, 0),
                                    reverse=True)
    weighted_value = mx.sym.transpose(weighted_value, axes=(0, 2, 1, 3))
    weighted_value = mx.sym.reshape(weighted_value, shape=(0, 0, -1))
    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
                                   num_hidden=out_dim, no_bias=False)
    executor = output.simple_bind(ctx=mx.gpu(0),
                                  qkv=(batch_size, qkv_length, qkv_dim),
                                  type_dict={'qkv': 'float16'},
                                  grad_req='write', force_rebind=True)
    executor.copy_params_from(arg_params, {})
    executor.arg_dict['sonde'][:] = 0.
    executor.arg_dict['sonde'].wait_to_read()
    executor.forward(is_train=True)
    output_orig = executor.outputs[0].asnumpy()
    executor.backward(mx.nd.array(output_grads, dtype='float16'))
    grads_orig = {k : v.asnumpy() for k, v in executor.grad_dict.items()}

    assert_allclose(output_orig, output_opti, rtol=1e-2, atol=1e-3)

    for k in grads_opti.keys():
        assert(grads_orig[k].dtype == grads_opti[k].dtype)
        assert(grads_orig[k].shape == grads_opti[k].shape)
        assert_allclose(grads_orig[k], grads_opti[k], rtol=1e-2, atol=1e-3)
        #print("\ngrad_{}{}: {}".format(k, diff.shape, diff.max()))
