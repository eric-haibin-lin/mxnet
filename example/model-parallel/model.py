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

import mxnet as mx

def gpu_id(i):
    return 'gpu_' + str(i)

def model(num_gpus, num_hidden, num_input):
    act_list = []
    loss_list = []
    for gpu in range(num_gpus):
        with mx.AttrScope(ctx_group=gpu_id(gpu)):
            data = mx.symbol.Variable('data_' + str(gpu))
            fc1  = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden/num_gpus)
            act1 = mx.symbol.Activation(data=fc1, act_type="relu")
            act_list.append(act1)

    for gpu in range(num_gpus):
        with mx.AttrScope(ctx_group=gpu_id(gpu)):
            label = mx.symbol.var('label_' + str(gpu))
            concat = mx.symbol.Concat(*act_list)
            fc2 = mx.symbol.FullyConnected(concat, num_hidden=num_input/num_gpus)
            loss = mx.symbol.LinearRegressionOutput(fc2, label)
            loss_list.append(loss)
    return mx.sym.Group(loss_list)
