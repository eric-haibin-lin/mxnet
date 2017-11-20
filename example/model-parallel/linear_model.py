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
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)

def linear_model(num_gpus, num_hidden, num_input):
    assert(num_gpus == 2)
    act_list = []
    loss_list = []
    for gpu in range(num_gpus):
        with mx.AttrScope(ctx_group='gpu_' + str(gpu)):
            data = mx.symbol.Variable('data_' + str(gpu))
            fc1  = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden/num_gpus)
            act1 = mx.symbol.Activation(data=fc1, act_type="relu")
            act_list.append(act1)

    for gpu in range(num_gpus):
        with mx.AttrScope(ctx_group='gpu_' + str(gpu)):
            label = mx.symbol.var('label_' + str(gpu))
            concat = mx.symbol.Concat(*act_list)
            fc2 = mx.symbol.FullyConnected(concat, num_hidden=num_input/num_gpus)
            loss = mx.symbol.LinearRegressionOutput(fc2, label)
            loss_list.append(loss)
    return mx.sym.Group(loss_list)

group2ctx = {
    'gpu_0' : mx.cpu(0),
    'gpu_1' : mx.cpu(1)
}

num_epoch = 5
num_batches = 10
num_input = 10
num_hidden = 2
batch_size = 8
num_gpus = 2
sym = linear_model(num_gpus, num_hidden, num_input)

rand_data = mx.nd.ones((batch_size * num_batches, num_input))
rand_labels = rand_data.split(axis=1, num_outputs=num_gpus)

iters = []
data_names = ['data_' + str(i) for i in range(num_gpus)]
label_names = ['label_' + str(i) for i in range(num_gpus)]
for i in range(num_gpus):
    data_iter_i = mx.io.NDArrayIter({data_names[i]:rand_data}, {label_names[i]:rand_labels[i]},
                                    batch_size=batch_size)
    iters.append(data_iter_i)
train_data = mx.io.PrefetchingIter(iters)

mod = mx.mod.Module(symbol=sym, data_names=data_names, label_names=label_names)
mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
mod.init_params()
optim = mx.optimizer.create('sgd', learning_rate=0.01)
mod.init_optimizer(optimizer=optim)
metric = mx.metric.create(['MSE'])
speedometer = mx.callback.Speedometer(batch_size, 1)

logging.info('Training started ...')
data_iter = iter(train_data)
for epoch in range(num_epoch):
    nbatch = 0
    metric.reset()
    for batch in data_iter:
        nbatch += 1
        mod.forward_backward(batch)
        # update all parameters (including the weight parameter)
        mod.update()
        # update training metric
        mod.update_metric(metric, batch.label)
        speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                   eval_metric=metric, locals=locals())
        speedometer(speedometer_param)
    data_iter.reset()
    score = mod.score(train_data, ['MSE'])
    logging.info('epoch %d, eval metric = %s ' % (epoch, score[0][1]))
    data_iter.reset()

mx.nd.waitall()
