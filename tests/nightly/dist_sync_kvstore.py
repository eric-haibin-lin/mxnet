#!/usr/bin/env python
# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), A.asnumpy()

# setup
keys = [3, 5, 7]
rate = 2
shape = (2, 2)
big_shape = (1200, 1200)        # bigger than BIGARRAY_BOUND

def init_kv():
    kv = mx.kv.create('dist_sync')
    # init kv
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init(99, mx.nd.ones(big_shape))
    my_rank = kv.rank
    nworker = kv.num_workers
    return kv, my_rank, nworker

def init_optimizer(kv):
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
    init_optimizer(kv)
    nrepeat = 3
    for i in range(nrepeat):
        kv.push(3, mx.nd.ones(shape)*(my_rank+1))
        kv.push(99, mx.nd.ones(big_shape)*(my_rank+1))

    num = (nworker + 1 ) * nworker * rate / 2 * nrepeat + 1
    val = mx.nd.zeros(shape)
    kv.pull(3, out = val)
    check_diff_to_scalar(val, num)

    val2 = mx.nd.zeros(big_shape)
    kv.pull(99, out = val2)
    check_diff_to_scalar(val2, num)

if __name__ == "__main__":
    test_sync_push_pull()
