import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon

mx.random.seed(0)
np.random.seed(0)
num_gpus = 2
ctx_list = [mx.gpu(i) for i in range(num_gpus)]

num_inputs = 240
num_hidden = 512
num_outputs = 240
num_examples = 320
batch_size = 32

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
        broadcast = []
        for act in acts:
            broadcast.append(act.copyto(mx.gpu(i)) if act.context.device_id != i else act)
        broadcasts.append(broadcast)
    concats = [mx.nd.concat(*broadcast) for broadcast in broadcasts]
    outputs = [mx.nd.dot(concats[i], w2[i]) + b2[i] for i in range(num_gpus)]
    return outputs

def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

epochs = 5
learning_rate = .001
smoothing_constant = .01
moving_loss = 0
niter = 0
losses = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        datas = [data.copyto(ctx) for ctx in ctx_list]
        labels = gluon.utils.split_and_load(label, ctx_list, batch_axis=1)
        with autograd.record():
            outputs = net(datas)
            losses = [square_loss(output, label) for output, label in zip(outputs, labels)]
        for l in losses:
            l.backward()

        #print(nd.sum(b1[0].grad))
        #print(nd.sum(b1[1].grad))
        # perform update
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter += 1
        losses_cpu = [loss.copyto(mx.cpu()) for loss in losses]
        curr_loss = mx.nd.mean(*losses_cpu).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correct the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)

        if (i + 1) % 10 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, est_loss))
