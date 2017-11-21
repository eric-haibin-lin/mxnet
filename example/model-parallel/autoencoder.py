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
import argparse
import logging
from model import *
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.INFO, format=head)

parser = argparse.ArgumentParser(description="Run autoencoder with model parallelism on multi-GPU",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-gpus', type=int, default=2,
                    help='number of gpus to use')
parser.add_argument('--num-epochs', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='number of examples per data batch')
parser.add_argument('--num-hidden', type=int, default=512,
                    help='number of hidden units')

args = parser.parse_args()
logging.info(args)

num_epoch = args.num_epochs
num_hidden = args.num_hidden
batch_size = args.batch_size
num_gpus = args.num_gpus
num_batches = 10
num_input = 240000

# model
net = model(num_gpus, num_hidden, num_input)
# parallel gpu groups
group2ctx = {gpu_id(i) : mx.gpu(i) for i in range(num_gpus)}

# dataset
iters = []
rand_data = mx.nd.ones((batch_size * num_batches, num_input))
# split labels vertically, each GPU gets a subset of inputs
rand_labels = rand_data.split(axis=1, num_outputs=num_gpus)
rand_labels = rand_labels if isinstance(rand_labels, list) else [rand_labels]
data_names = ['data_' + str(i) for i in range(num_gpus)]
label_names = ['label_' + str(i) for i in range(num_gpus)]

for i in range(num_gpus):
    # each GPU gets a complete batch of data
    iter_i = mx.io.NDArrayIter({data_names[i]:rand_data},
                               {label_names[i]:rand_labels[i]},
                               batch_size=batch_size)
    iters.append(iter_i)
train_data = mx.io.PrefetchingIter(iters)

# module
mod = mx.mod.Module(symbol=net, data_names=data_names,
                    label_names=label_names, group2ctxs=[group2ctx])
mod.bind(data_shapes=train_data.provide_data,
         label_shapes=train_data.provide_label)
mod.init_params()
mod.init_optimizer(optimizer='adam')
metric = mx.metric.create(['MSE'])
speedometer = mx.callback.Speedometer(batch_size, 10)

logging.info('Training started ...')
data_iter = iter(train_data)
for epoch in range(num_epoch):
    nbatch = 0
    metric.reset()
    for batch in data_iter:
        nbatch += 1
        mod.forward_backward(batch)
        mod.update()
        mod.update_metric(metric, batch.label)
        speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                   eval_metric=None, locals=locals())
        speedometer(speedometer_param)
    data_iter.reset()
