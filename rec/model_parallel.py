import time
import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
import argparse

parser = argparse.ArgumentParser(description="Run auto-encoder with model parallel",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-epoch', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=128,
                    help='number of examples per batch')
parser.add_argument('--num-gpus', type=int, default=2,
                    help='number of gpus')

if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)

    # arg parser
    args = parser.parse_args()
    logging.info(args)
    num_gpus = args.num_gpus
    epochs = args.num_epoch
    batch_size = args.batch_size

    ctx_list = [mx.gpu(i) for i in range(num_gpus)]

    num_inputs = 24000
    num_hidden = 512
    num_outputs = 24000
    num_examples = 10 * 128

    def real_fn(X):
        return X

    X = nd.random.normal(shape=(num_examples, num_inputs))
    y = X

    train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X, y),
                                          batch_size=batch_size, shuffle=True)

    w1_cpu = nd.random.normal(shape=(num_inputs, num_hidden))
    b1_cpu = nd.random.normal(shape=num_hidden)
    w2_cpu = nd.random.normal(shape=(num_hidden, num_outputs))
    b2_cpu = nd.random.normal(shape=num_outputs)

    # shard w1 and w2
    w1 = gluon.utils.split_and_load(w1_cpu, ctx_list, batch_axis=1)
    w2 = gluon.utils.split_and_load(w2_cpu, ctx_list, batch_axis=1)
    b1 = gluon.utils.split_and_load(b1_cpu, ctx_list)
    b2 = gluon.utils.split_and_load(b2_cpu, ctx_list)

    params = w1 + w2 + b1 + b2
    for param in params:
        param.attach_grad()

    def net(Xs):
        hiddens = [mx.nd.dot(Xs[i], w1[i]) + b1[i] for i in range(num_gpus)]
        acts = [mx.nd.relu(hiddens[i]) for i in range(num_gpus)]
        broadcasts = []
        for i in range(num_gpus):
            broadcast = [act.copyto(mx.gpu(i)) if act.context.device_id != i else act for act in acts]
            broadcasts.append(broadcast)
        concats = [mx.nd.concat(*broadcast) for broadcast in broadcasts]
        outputs = [mx.nd.dot(concats[i], w2[i]) + b2[i] for i in range(num_gpus)]
        return outputs

    def square_loss(yhat, y):
        return nd.mean((yhat - y) ** 2)

    def SGD(params, lr):
        for param in params:
            mx.nd.sgd_update(weight=param, grad=param.grad, lr=lr, out=param)

    learning_rate = .001
    smoothing_constant = .01
    moving_loss = 0
    niter = 0
    losses = []

    mx.nd.waitall()
    start = time.time()
    datas = None
    labels = None
    profiler_name = 'mp_' + str(num_gpus) + '_profile_output.json'
    mx.profiler.profiler_set_config(mode='all', filename=profiler_name)
    mx.profiler.profiler_set_state('run')
    for e in range(epochs):
        nbatch = 0
        data_iter = iter(train_data)
        next_batch = next(data_iter)
        datas = [next_batch[0].copyto(ctx) for ctx in ctx_list]
        labels = gluon.utils.split_and_load(next_batch[1], ctx_list,
                                            batch_axis=1, even_split=False)
        end_of_batch = False
        while not end_of_batch:
            nbatch += 1
            # forward
            with autograd.record():
                outputs = net(datas)
                losses = [square_loss(output, label) for output, label in zip(outputs, labels)]
                total_loss = sum([l.copyto(mx.gpu(0)) for l in losses]) / num_gpus
            # calculate gradients
            total_loss.backward()
            # perform update
            SGD(params, learning_rate)
            try:
                next_batch = next(data_iter)
                datas = [next_batch[0].copyto(ctx) for ctx in ctx_list]
                labels = gluon.utils.split_and_load(next_batch[1], ctx_list,
                                                    batch_axis=1, even_split=False)
            except StopIteration:
                end_of_batch = True
            niter += 1
            curr_loss = total_loss.asscalar()
            if nbatch % 10 == 0:
                print("Epoch %s, batch %s. loss of current batch: %s" % (e, nbatch, curr_loss))

    mx.nd.waitall()
    mx.profiler.profiler_set_state('stop')
    end = time.time()
    print(num_gpus, end - start)
