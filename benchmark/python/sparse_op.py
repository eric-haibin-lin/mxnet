# pylint: skip-file
import mxnet as mx
from mxnet.test_utils import *
import numpy as np
import scipy.sparse as sp
import os, gzip
import pickle as pickle
import time
import sys

def get_avazu(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('avazu-app.t')):
        import urllib, zipfile
        zippath = os.path.join(data_dir, "avazu-app.t.bz2")
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2"
        urllib.urlretrieve(url, zippath)
        # decompress
        os.system("bzip2 -d avazu-app.t.bz2")
    os.chdir("..")

def test_dot_real():
    def get_iter(path, data_shape, batch_size):
        data_train = mx.io.LibSVMIter(data_libsvm=path,
                                      data_shape=data_shape,
                                      batch_size=batch_size)
        data_iter = iter(data_train)
        return data_iter
    data_dir = os.path.join(os.getcwd(), 'data')
    get_avazu(data_dir)
    path = os.path.join(data_dir, 'avazu-app.t')
    size = 336490781 >> 20

    # model
    batch_size = 512
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
    cost = end - start
    print(size / cost, cost, num_batch, num_batch / cost)

def test_dot_synthetic():
    """benchmark mx.nd.dot(sparse_ndarray, dense_ndarray) with given density.
    `t_sparse` is the time cost of dot(csr, dns), while `t_dense` is the time cost
    of dot(dns, dns), with the same matrix except that it is in default storage type.
    """
    def measure_cost(repeat, f, *args, **kwargs):
       # start bench
       start = time.time()
       results = []
       for i in range(repeat):
           results.append(f(*args, **kwargs))
       for result in results:
           result.wait_to_read()
       end = time.time()
       diff = end - start
       return diff / repeat

    def bench_dot_forward(m, k, n, density, ctx, repeat):
        set_default_context(ctx)
        dns = mx.nd.random_uniform(shape=(k, n)).copyto(ctx)
        data_shape = (m, k)
        csr_data = rand_ndarray(data_shape, 'csr', density)
        dns_data = csr_data.to_dense()

        data = [dns_data, csr_data]
        costs = []
        for d in data:
            dns.wait_to_read()
            d.wait_to_read()
            cost = measure_cost(repeat, mx.nd.dot, d, dns)
            costs.append(cost / repeat)
        ratio = costs[1] / costs[0]
        fmt = "%0.1f\t\t%s\t%d\t%d\t%d\t%0.6f\t%0.5f\t%0.2f"
        print(fmt % (density * 100, str(ctx), n, m, k, costs[1], costs[0], ratio))

    def bench_dot_backward(m, k, n, density, ctx, repeat):
        set_default_context(ctx)
        dns = mx.nd.random_uniform(shape=(m, n)).copyto(ctx)
        data_shape = (m, k)
        csr_data = rand_ndarray(data_shape, 'csr', density)
        dns_data = csr_data.to_dense()

        data = [dns_data, csr_data]
        costs = []
        for d in data:
            dns.wait_to_read()
            d.wait_to_read()
            cost = measure_cost(repeat, mx.nd.dot, d, dns, transpose_a=True)
            costs.append(cost)
        ratio = costs[1] / costs[0]
        fmt = "%0.1f\t\t%s\t%d\t%d\t%d\t%0.6f\t%0.5f\t%0.2f"
        print(fmt % (density * 100, str(ctx), n, m, k, costs[1], costs[0], ratio))


    print("A = sparse NDArray of shape(m, k)")
    print("B = dense NDArray of shape(k, n)")
    print("dot_forward\tdot(csr, dns)")
    print('density(%)\tcontext\tn\tm\tk\tt_sparse\tt_dense\tt_sparse/t_dense')
    # TODO(haibin) make these runtime options
    m = 512
    k = [50000, 100000]
    n = [50, 100]
    density = [0.05, 0.02, 0.01, 0.005, 0.001]
    num_repeat = 1
    # contexts = [mx.cpu(), mx.gpu(0)]
    contexts = [mx.cpu()]
    for i in range(2):
        for ctx in contexts:
            for den in density:
                bench_dot_forward(m, k[i], n[i], den, ctx, num_repeat)

    print("dot_backward\tdot(csr.T, dns)")
    print('density(%)\tcontext\tn\tm\tk\tt_sparse\tt_dense\tt_sparse/t_dense')
    for i in range(2):
        for ctx in contexts:
            for den in density:
                bench_dot_backward(m, k[i], n[i], den, ctx, num_repeat)

if __name__ == "__main__":
    test_dot_real()
    test_dot_synthetic()
