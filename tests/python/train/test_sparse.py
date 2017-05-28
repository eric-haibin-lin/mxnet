# pylint: skip-file
import mxnet as mx
from mxnet.test_utils import *
import numpy as np
import scipy.sparse as sp
import os, gzip
import pickle as pickle
import time
import sys
from common import get_data

def test_dot_real(dataset):
    def get_iter(path, data_shape, batch_size):
        data_train = mx.io.LibSVMIter(data_libsvm=path,
                                      data_shape=data_shape,
                                      batch_size=batch_size)
        data_iter = iter(data_train)
        return data_iter

    batch_size = 512
    if dataset == 'train':
        size = 2858375979 >> 20
        path = "/home/ubuntu/svm/avazu-app"
    elif dataset == 'test':
        size = 336490781 >> 20
        path = "/home/ubuntu/svm/avazu-app.t"
    else:
        raise Exception('unknown dataset')

    # model
    feature_dim = 1000000
    data_shape = (feature_dim, )
    train_iter = get_iter(path, data_shape, batch_size)

    k = 500
    weight = mx.nd.random_uniform(low=0, high=1, shape=(feature_dim, k)) 
    weight.wait_to_read()

    # start workload
    start = time.time()
    results = []
    num_batch = 0
    for batch in train_iter:
        data = train_iter.getdata()
        results.append(mx.nd.dot(data, weight))
        num_batch += 1
    for result in results:
        result.wait_to_read()

    end = time.time()
    duration = end - start
    print(size / duration, duration, num_batch, num_batch / duration)

def test_dot_synthetic():
    def bench_dot(m, k, n, density):
        weight = mx.nd.random_uniform(low=0, high=1, shape=(k, n))
        data_shape = (m, k)

        csr_data = rand_ndarray(data_shape, 'csr', density)
        dns_data = csr_data.to_dense()
        num_iter = 50

        data = [dns_data, csr_data]
        durations = []
        for d in data:
            weight.wait_to_read()
            d.wait_to_read()
            # start bench
            start = time.time()
            results = []
            for i in range(num_iter):
                results.append(mx.nd.dot(d, weight))
            for result in results:
                result.wait_to_read()
            end = time.time()
            duration = end - start
            durations.append(duration)
        ratio = durations[1] / durations[0]
        print("%0.2f\t%d\t%d\t%d\t%0.3f" % (density, n, m, k, ratio))

    print('density\tn\tm\tk\tt_sparse/t_dense')
    m = 512
    k = 50000
    n = 50
    density = [0.8, 0.5, 0.2, 0.1, 0.01, 0.001]
    for den in density:
        bench_dot(m, k, n, den)

if __name__ == "__main__":
    #test_dot_real('test')
    test_dot_synthetic()
