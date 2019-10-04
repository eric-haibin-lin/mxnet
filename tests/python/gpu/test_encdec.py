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

#import mxnet as mx
#import numpy as np
#
#batch_size = 10
#q_length = 30
#kv_length = 32  # length of a sequence
##qkv_dim = 16     # dimension of encoding
#q_dim = 16
#kv_dim = 20
#num_heads = 28   # number of attention head
#head_dim = 18    # head size
#out_dim = 22
#qkv_units = num_heads * head_dim
#
#def convert_weight(k_weight, v_weight, num_heads):
#    k_weight = mx.sym.reshape(k_weight, shape=(num_heads, -1, 0), reverse=True)
#    v_weight = mx.sym.reshape(v_weight, shape=(num_heads, -1, 0), reverse=True)
#    all_weights = mx.sym.concat(k_weight, v_weight, dim=-2)
#    all_weight = mx.sym.reshape(all_weights, shape=(-1, 0), reverse=True)
#    return all_weight
#
#def convert_bias(k_bias, v_bias, num_heads):
#    k_bias = mx.sym.reshape(k_bias, shape=(num_heads, -1))
#    v_bias = mx.sym.reshape(v_bias, shape=(num_heads, -1))
#    all_bias = mx.sym.stack(k_bias, v_bias, axis=1)
#    all_bias = mx.sym.reshape(all_bias, shape=(-1))
#    return all_bias
#
#q = mx.sym.Variable('q')
#kv = mx.sym.Variable('kv') # (batch_size, qkv_length, qkv_dim)
#q_weight = mx.sym.Variable('q_weight')
#k_weight = mx.sym.Variable('k_weight')
#v_weight = mx.sym.Variable('v_weight')
#out_weight = mx.sym.Variable('out_weight')
#q_bias = mx.sym.Variable('q_bias')
#k_bias = mx.sym.Variable('k_bias')
#v_bias = mx.sym.Variable('v_bias')
#out_bias = mx.sym.Variable('out_bias')
#arg_params = {
#    'q': mx.nd.array(np.random.rand(*(batch_size, q_length, q_dim)).astype('float16') * 1., dtype='float16'),
#    'kv': mx.nd.array(np.random.rand(*(batch_size, kv_length, kv_dim)).astype('float16') * 1., dtype='float16'),
#    'q_weight': mx.nd.array(np.random.rand(*(qkv_units, q_dim)).astype('float16') * 0.1, dtype='float16'),
#    'k_weight': mx.nd.array(np.random.rand(*(qkv_units, kv_dim)).astype('float16') * 0.1, dtype='float16'),
#    'v_weight': mx.nd.array(np.random.rand(*(qkv_units, kv_dim)).astype('float16') * 0.1, dtype='float16'),
#    'out_weight': mx.nd.array(np.random.rand(*(out_dim, qkv_units)).astype('float16') * 0.1, dtype='float16'),
#    'q_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype('float16') * 0.1, dtype='float16'),
#    'k_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype('float16') * 0.1, dtype='float16'),
#    'v_bias': mx.nd.array(np.random.rand(*(qkv_units,)).astype('float16') * 0.1, dtype='float16'),
#    'out_bias': mx.nd.array(np.random.rand(*(out_dim,)).astype('float16') * 0.1, dtype='float16'),
#    }
#if True:
#    kv_weight = convert_weight(k_weight=k_weight, v_weight=v_weight, num_heads=num_heads)
#    kv_bias = convert_bias(k_bias=k_bias, v_bias=v_bias, num_heads=num_heads)
#    kv_transposed = mx.sym.transpose(kv, axes=(1, 0, 2))
#    kv_proj = mx.sym.FullyConnected(kv_transposed, weight=kv_weight, bias=kv_bias, flatten=False,
#                                     num_hidden=qkv_units * 2, no_bias=False)
#    q_transposed = mx.sym.transpose(q, axes=(1, 0, 2))
#    q_proj = mx.sym.FullyConnected(q_transposed, weight=q_weight, bias=q_bias, flatten=False,
#                                   num_hidden=qkv_units, no_bias=False)
#    att_score = mx.sym.interleaved_matmul_encdec_qk(q_proj, kv_proj, heads=num_heads)
#    weighted_value = mx.sym.interleaved_matmul_encdec_valatt(kv_proj, att_score, heads=num_heads)
#
#    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
#                                   num_hidden=out_dim, no_bias=False)
#    output = mx.sym.transpose(output, axes=(1, 0, 2))
#    executor = output.simple_bind(ctx=mx.gpu(0),
#                                  q=(batch_size, q_length, q_dim),
#                                  kv=(batch_size, kv_length, kv_dim),
#                                  q_weight=(qkv_units, q_dim),
#                                  q_bias=(qkv_units,),
#                                  k_weight=(qkv_units, kv_dim),
#                                  k_bias=(qkv_units,),
#                                  v_weight=(qkv_units, kv_dim),
#                                  v_bias=(qkv_units,),
#                                  out_weight=(out_dim, qkv_units),
#                                  out_bias=(out_dim,),
#                                  type_dict={'q': 'float16',
#                                             'q_weight': 'float16',
#                                             'q_bias': 'float16',
#                                             'k': 'float16',
#                                             'k_weight': 'float16',
#                                             'k_bias': 'float16',
#                                             'v_weight': 'float16',
#                                             'v_bias': 'float16',
#                                             'out_weight' : 'float16',
#                                             'out_bias': 'float16'
#                                            },
#                                  grad_req='write', force_rebind=True)
#    executor.copy_params_from(arg_params, {})
#    executor.forward(is_train=True)
#    output_opti = executor.outputs[0].asnumpy()
#    executor.backward(out_grads=
#    print(output_opti.shape)
#
#if True:
#    q = mx.sym.FullyConnected(q, weight=q_weight, bias=q_bias, flatten=False,
#                              num_hidden=qkv_units, no_bias=False)  # (batch_size, q_length, qkv_units)
#    k = mx.sym.FullyConnected(kv, weight=k_weight, bias=k_bias, flatten=False,
#                              num_hidden=qkv_units, no_bias=False)  # (batch_size, kv_length, qkv_units)
#    v = mx.sym.FullyConnected(kv, weight=v_weight, bias=v_bias, flatten=False,
#                              num_hidden=qkv_units, no_bias=False)  # (batch_size, kv_length, qkv_units)
#    q = mx.sym.reshape(q, shape=(0, 0, num_heads, -1))
#    q = mx.sym.transpose(q, axes=(0, 2, 1, 3))
#    q = mx.sym.reshape(q, shape=(-1, 0, 0), reverse=True)
#    k = mx.sym.reshape(k, shape=(0, 0, num_heads, -1))
#    k = mx.sym.transpose(k, axes=(0, 2, 1, 3))
#    k = mx.sym.reshape(k, shape=(-1, 0, 0), reverse=True)
#    q = mx.sym.contrib.div_sqrt_dim(q)
#    att_score = mx.sym.batch_dot(q, k, transpose_b=True)
#    v = mx.sym.reshape(v, shape=(0, 0, num_heads, -1))
#    v = mx.sym.transpose(v, axes=(0, 2, 1, 3))
#    v = mx.sym.reshape(v, shape=(-1, 0, 0), reverse=True)
#    weighted_value = mx.sym.batch_dot(att_score, v)
#    weighted_value = mx.sym.reshape(weighted_value, shape=(-1, num_heads, 0, 0),
#                                    reverse=True)
#    weighted_value = mx.sym.transpose(weighted_value, axes=(0, 2, 1, 3))
#    weighted_value = mx.sym.reshape(weighted_value, shape=(0, 0, -1))
#    output = mx.sym.FullyConnected(weighted_value, weight=out_weight, bias=out_bias, flatten=False,
#                                   num_hidden=out_dim, no_bias=False)
#    executor = output.simple_bind(ctx=mx.gpu(0),
#                                  q=(batch_size, q_length, q_dim),
#                                  kv=(batch_size, kv_length, kv_dim),
#                                  q_weight=(qkv_units, q_dim),
#                                  q_bias=(qkv_units,),
#                                  k_weight=(qkv_units, kv_dim),
#                                  k_bias=(qkv_units,),
#                                  v_weight=(qkv_units, kv_dim),
#                                  v_bias=(qkv_units,),
#                                  out_weight=(out_dim, qkv_units),
#                                  out_bias=(out_dim,),
#                                  type_dict={'q': 'float16',
#                                             'q_weight': 'float16',
#                                             'q_bias': 'float16',
#                                             'k': 'float16',
#                                             'k_weight': 'float16',
#                                             'k_bias': 'float16',
#                                             'v_weight': 'float16',
#                                             'v_bias': 'float16',
#                                             'out_weight': 'float16',
#                                             'out_bias': 'float16',
#                                            },
#                                  grad_req='write', force_rebind=True)
#    executor.copy_params_from(arg_params, {})
#    executor.forward(is_train=True)
#    output_orig = executor.outputs[0].asnumpy()
#    print(output_orig.shape)
#
#diff = abs(output_orig.flatten() - output_opti.flatten())
#print((diff / (abs(output_orig.flatten()) + abs(output_opti.flatten()))).mean())
