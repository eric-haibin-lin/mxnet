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

from __future__ import division

import argparse, time, os
import logging
os.environ['MXNET_UPDATE_ON_KVSTORE'] = '0'

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import models as override_models
from mxnet.gluon.model_zoo import vision as models
from mxnet import autograd as ag
from mxnet.test_utils import get_mnist_iterator
from mxnet.metric import Accuracy, TopKAccuracy, CompositeEvalMetric
import numpy as np
import mxnet.contrib.amp as amp
import horovod.mxnet as hvd

from common.dali import get_rec_iter, add_dali_args
import common.helper as helper

# logging
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('image-classification.log')
logger = logging.getLogger()
logger.addHandler(fh)
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
fh.setLevel(logging.DEBUG)
logging.debug('\n%s', '-' * 100)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
fh.setFormatter(formatter)

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='/data/imagenet/train-480-val-256-recordio/',
                  help='training directory of imagenet images, contains train/val subdirs.')
parser.add_argument('-b', '--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument("-n", "--n-GPUs", type=int, default=8, help="number of GPUs to use; " +\
                    "default = 8")
parser.add_argument('-e', '--epochs', type=int, default=90,
                    help='number of training epochs.')
parser.add_argument('--accuracy-threshold', type=float, default=1.0,
                    help='stop training after top1 reaches this value; default = 1.0')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123.')
parser.add_argument('--mode', type=str, default='hybrid',
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--network', type=str, default='resnet50_v1',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--prefix', default='', type=str,
                    help='path to checkpoint prefix, default is current working dir')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='starting epoch, 0 for fresh training, > 0 to resume')
parser.add_argument('--resume', type=str, default='',
                    help='path to saved weight where you want resume')
parser.add_argument('--lr-factor', default=0.1, type=float,
                    help='learning rate decay ratio')
parser.add_argument('--lr-steps', default='30,60,80,90', type=str,
                    help='list of learning rate decay epochs as in str')
parser.add_argument('--dtype', default='float16', type=str,
                    help='data type, float32 or float16 if applicable')
parser.add_argument('--save-frequency', default=10, type=int,
                    help='epoch frequence to save model, best model will always be saved')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--resize', type=int, default=256,
                    help='Shorter size to which resize validation images (keeping aspect ratio).')
parser.add_argument('--image-shape', type=str, default='3,224,224',
                    help="Shape of the input to the network.")
parser.add_argument('--input-layout', type=str, default="NCHW",
                    help="Layout of the input to the network.")
parser.add_argument('--conv-layout', type=str, default="NCHW",
                    help="Layout of the convolution layers.")
parser.add_argument('--bn-layout', type=str, default="NCHW",
                    help="Layout of the BatchNorm layers.")
parser.add_argument('--pooling-layout', type=str, default="NCHW",
                    help="Layout of the Pooling layers.")
parser.add_argument('--no-val', action='store_true',
                    help='If enabled, does not perform validation.')
parser.add_argument('-s', '--num-examples', type=int, default=1281167,
                    help='Override number of training examples in an epoch.')
parser.add_argument('--amp', action='store_true',
                    help='If enabled, turn on AMP (Automatic Mixed Precision.)')

add_dali_args(parser)
opt = parser.parse_args()

# Horovod: initialize Horovod
if 'horovod' in opt.kvstore:
    hvd.init()

# global variables
logger.info('Starting new image-classification task:, %s',opt)
mx.random.seed(opt.seed)
if opt.amp:
    if opt.dtype == 'float16':
        logging.warning('Automatic Mixed Precision expects float32 dtype and float16 provided. ' +
                        'Overriding to float32. Run with --dtype float32 to suppress this warning.')
        opt.dtype = 'float32'
    amp.init()
model_name = opt.network
batch_size, classes = opt.batch_size, 1000

if 'horovod' in opt.kvstore:
    context = [mx.gpu(hvd.local_rank())]
    total_batch_size = batch_size * hvd.size()
else:
    context = [mx.gpu(int(i)) for i in range(opt.n_GPUs)] if opt.n_GPUs > 0 else [mx.cpu()]
    num_gpus = len(context)
    total_batch_size = batch_size * max(1, num_gpus)

modified_lr = opt.lr * total_batch_size / 256
lr_steps = [int(x) for x in opt.lr_steps.split(',') if x.strip()]
metric = CompositeEvalMetric([Accuracy(), TopKAccuracy(5)])
kv = None if 'horovod' in opt.kvstore else mx.kv.create(opt.kvstore)

class fp16_model(gluon.block.HybridBlock):
    def __init__(self, net, **kwargs):
        super(fp16_model, self).__init__(**kwargs)
        with self.name_scope():
            self._net = net

    def hybrid_forward(self, F, x):
        y = self._net(x)
        y = F.cast(y, dtype='float32')
        return y

def get_model(model, ctx, opt):
    """Model initialization."""
    kwargs = {'classes': classes}
    if model.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm

    helper._CONFIG_DTYPE = opt.dtype
    # first try to find the model in overriden models
    try:
        net = override_models.get_model(model,
                                        input_layout=opt.input_layout,
                                        conv_layout=opt.conv_layout,
                                        bn_layout=opt.bn_layout,
                                        pooling_layout=opt.pooling_layout,
                                        **kwargs)
    except:
        if (opt.input_layout   == 'NHWC' or
            opt.conv_layout    == 'NHWC' or
            opt.bn_layout      == 'NHWC' or
            opt.pooling_layout == 'NHWC'):
            raise RuntimeError("Official models do not have NHWC layout support")
        net = models.get_model(model, **kwargs)
    if opt.resume:
        net.load_parameters(opt.resume)
        net.cast(opt.dtype)
    else:
        net.cast(opt.dtype)
        if model in ['alexnet']:
            net.initialize(mx.init.Normal())
        else:
            net.initialize(mx.init.Xavier(magnitude=2))
    if opt.dtype == 'float16':
        net = fp16_model(net)
    return net

net = get_model(model_name, context, opt)

def test(ctx, val_data):
    metric.reset()
    val_data.reset()
    for batches in val_data:
        data = [b.data[0][0:-b.pad if b.pad != 0 else None] for b in batches]
        label = [b.label[0][0:-b.pad if b.pad != 0 else None] for b in batches]
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def test_hvd(ctx, val_data):
    metric.reset()
    val_data.reset()
    for batches in val_data:
        data = [b.data[0][0:-b.pad if b.pad != 0 else None] for b in batches]
        label = [b.label[0][0:-b.pad if b.pad != 0 else None] for b in batches]
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    # Aggregate metric stats across ranks
    name = []
    val_acc = []
    for i in range(len(metric.metrics)):
        acc = metric.get_metric(i)
        vals = mx.nd.array([acc.sum_metric, acc.num_inst], dtype='int32')
        reduced_vals = hvd.allreduce(vals, average=False)
        reduced_vals = reduced_vals.asnumpy()
        name.append(acc.name)
        val_acc.append(float(reduced_vals[0]) / reduced_vals[1])
    return name, val_acc

def update_learning_rate(lr, trainer, step, num_examples, ratio, steps):
    """Set the learning rate to the initial value decayed by ratio every N epochs."""
    lr = helper.LRSchedule(lr, step, num_examples, total_batch_size, 5, steps, ratio)
    trainer.set_learning_rate(lr)
    return lr, trainer

def save_checkpoint(epoch, top1, best_acc):
    if opt.save_frequency and (epoch + 1) % opt.save_frequency == 0:
        fname = os.path.join(opt.prefix, '%s_%d_acc_%.4f.params' % (model_name, epoch, top1))
        net.save_parameters(fname)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)
    if top1 > best_acc[0]:
        best_acc[0] = top1
        fname = os.path.join(opt.prefix, '%s_best.params' % (model_name))
        net.save_parameters(fname)
        logger.info('[Epoch %d] Saving checkpoint to %s with Accuracy: %.4f', epoch, fname, top1)

def train(opt, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    train_data, val_data = get_rec_iter(opt, kv)
    net.collect_params().reset_ctx(ctx)

    if 'horovod' in opt.kvstore:
        # Fetch and broadcast parameters
        params = net.collect_params()
        if params is not None:
            hvd.broadcast_parameters(params, root_rank=0)

        # Using Horovod DistributedTrainer
        trainer = hvd.DistributedTrainer(net.collect_params(), 'sgd',
                                optimizer_params={'learning_rate': opt.lr,
                                                  'wd': opt.wd,
                                                  'momentum': opt.momentum,
                                                  'multi_precision': True})
    else:
        trainer = gluon.Trainer(net.collect_params(), 'sgd',
                                optimizer_params={'learning_rate': opt.lr,
                                                  'wd': opt.wd,
                                                  'momentum': opt.momentum,
                                                  'multi_precision': True},
                                kvstore=kv,
                                update_on_kvstore=False)
    if opt.amp:
        amp.init_trainer(trainer)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    if opt.mode == "hybrid":
        loss.hybridize(static_shape=True, static_alloc=True)

    total_time = 0
    num_epochs = 0
    best_acc = [0]
    step = 0
    for epoch in range(opt.start_epoch, opt.epochs):
        tic = time.time()
        train_data.reset()
        metric.reset()
        btic = time.time()
        for i, batches in enumerate(train_data):
            lr, trainer = update_learning_rate(modified_lr, trainer, step, opt.num_examples, opt.lr_factor, lr_steps)
            data = [b.data[0] for b in batches]
            label = [b.label[0].as_in_context(b.data[0].context) for b in batches]
            outputs = []
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    z = net(x)
                    L = loss(z, y)
                    # store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    outputs.append(z)
                if opt.amp:
                    with amp.scale_loss(Ls, trainer) as scaled_loss:
                        ag.backward(scaled_loss)
                else:
                    ag.backward(Ls)
            metric.update(label, outputs)
            if opt.log_interval and not (i+1)%opt.log_interval:
                if ('horovod' in opt.kvstore and hvd.rank() == 0) or ('horovod' not in opt.kvstore):
                    name, acc = metric.get()
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\tLR=%f\t%s=%f, %s=%f'%(
                                   epoch, i, opt.log_interval * total_batch_size/(time.time()-btic),
                                   lr, name[0], acc[0], name[1], acc[1]))
                metric.reset()
                btic = time.time()
            step += 1
            if 'horovod' in opt.kvstore:
                trainer.step(batch_size)
            else:
                trainer.step(total_batch_size)

        # Sync params across workers (to ensure common BN statistics)
        if 'horovod' in opt.kvstore:
            params = net.collect_params()
            tensors = [p.data() for _,p in sorted(params.items())]
            for i, tensor in enumerate(tensors):
                hvd.allreduce_(tensor, average=True, name="param_{}".format(i))
        else:
            params = net.collect_params()
            for _,p in sorted(params.items()):
               p.set_data(p._reduce())

        epoch_time = time.time()-tic

        # First epoch will usually be much slower than the subsequent epics,
        # so don't factor into the average
        if num_epochs > 0:
          total_time = total_time + epoch_time
        num_epochs = num_epochs + 1

        if ('horovod' in opt.kvstore and hvd.rank() == 0) or ('horovod' not in opt.kvstore):
            name, acc = metric.get_global()
            logger.info('[Epoch %d] training: %s=%f, %s=%f'%(epoch, name[0], acc[0], name[1], acc[1]))
            logger.info('[Epoch %d] time cost: %f'%(epoch, epoch_time))

        if not opt.no_val:
            if 'horovod' in opt.kvstore:
                name, val_acc = test_hvd(ctx, val_data)
            else:
                name, val_acc = test(ctx, val_data)

            if ('horovod' in opt.kvstore and hvd.rank() == 0) or ('horovod' not in opt.kvstore):
                logger.info('[Epoch %d] validation: %s=%f, %s=%f'%(epoch, name[0], val_acc[0], name[1], val_acc[1]))
                # save model if meet requirements
                save_checkpoint(epoch, val_acc[0], best_acc)

            # Stop training if validation top1 meets accuracy threshold
            if val_acc[0] >= opt.accuracy_threshold:
                break

    if num_epochs > 1:
        print('Average epoch time: {}'.format(float(total_time)/(num_epochs - 1)))

def main():
    if opt.mode == 'symbolic':
        data = mx.sym.var('data')
        out = net(data)
        softmax = mx.sym.SoftmaxOutput(out, name='softmax')
        mod = mx.mod.Module(softmax, context=context)
        train_data, val_data = get_rec_iter(opt, kv)
        mod.fit(train_data,
                eval_data=val_data,
                num_epoch=opt.epochs,
                kvstore=kv,
                batch_end_callback = mx.callback.Speedometer(total_batch_size, max(1, opt.log_interval)),
                epoch_end_callback = mx.callback.do_checkpoint('image-classifier-%s'% model_name),
                optimizer = 'sgd',
                optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'multi_precision': True},
                initializer = mx.init.Xavier(magnitude=2))
        mod.save_params('image-classifier-%s-%d-final.params'%(model_name, opt.epochs))
    else:
        if opt.mode == 'hybrid':
            net.hybridize(static_shape=True, static_alloc=True)
        train(opt, context)

if __name__ == '__main__':
    main()
