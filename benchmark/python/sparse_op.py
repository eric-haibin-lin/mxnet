import ctypes

from mxnet.test_utils import *
import scipy.sparse as sp
import os
import time
import argparse

from mxnet.base import check_call, _LIB

parser = argparse.ArgumentParser(description="Benchmark sparse operators",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-omp-threads', type=int, default=1, help='number of omp threads to set in MXNet')
args = parser.parse_args()


def test_dot_real(batch_size):
    def get_iter(path, data_shape, batch_size):
        data_train = mx.io.LibSVMIter(data_libsvm=path,
                                      data_shape=data_shape,
                                      batch_size=batch_size)
        data_iter = iter(data_train)
        return data_iter

    # dataset info
    data_dir = os.path.join(os.getcwd(), 'data')
    data_name = 'kdda.t.mini'
    feature_dim = 20216830
    density = 0.000186630098465
    path = os.path.join(data_dir, data_name)

    if not os.path.exists(os.path.join(path)):
        raise Exception("Data is not prepared!")

    size = float(os.path.getsize(path)) / (2**20)

    # model
    data_shape = (feature_dim, )
    train_iter = get_iter(path, data_shape, batch_size)

    k = 200 
    weight = mx.nd.random_uniform(low=0, high=1, shape=(feature_dim, k))
    weight.wait_to_read()

    csr_data = []
    dns_data = []
    num_batch = 0
    for batch in train_iter:
        data = train_iter.getdata()
        csr_data.append(data)
        dns_data.append(data.to_dense())
        num_batch += 1

    results = []
    start = time.time()
    for d in csr_data:
        results.append(mx.nd.dot(d, weight))
    for res in results:
        res.wait_to_read()
    end = time.time()
    t_sparse = (end - start) / batch_size

    results = []
    start = time.time()
    for d in dns_data:
        results.append(mx.nd.dot(d, weight))
    for res in results:
        res.wait_to_read()
    end = time.time()
    t_dense = (end - start) / batch_size

    ratio = t_dense / t_sparse
    print('density(%)\tn\tm\tk\tt_dense/t_sparse\tt_dense\tt_sparse')
    fmt = "%0.6f\t%d\t%d\t%d\t\t%0.2f\t\t%0.4f\t%0.6f"
    print(
	fmt % (density * 100, batch_size, feature_dim, k, ratio, t_dense, t_sparse)
    )


def test_dot_synthetic():
    """benchmark mx.nd.dot(sparse_ndarray, dense_ndarray) with given density.
    `t_sparse` is the time cost of dot(csr, dns), while `t_dense` is the time cost
    of dot(dns, dns), with the same matrix except that it is in default storage type.
    """
    def measure_cost_forward_baseline(repeat, dot, lhs, rhs):
        start = time.time()
        for i in range(repeat):
            dot(lhs, rhs)
        end = time.time()
        diff = end - start
        return diff / repeat

    def measure_cost_backward_baseline(repeat, dot, transpose, lhs, rhs):
        start = time.time()
        for i in range(repeat):
            dot(transpose(lhs), rhs)
        end = time.time()
        diff = end -start
        return diff / repeat

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
        rhs_dns_np = dns.asnumpy()
        lhs_csr_sp = sp.csr_matrix(dns_data.asnumpy())  # csr in scipy
        lhs_dns_np = lhs_csr_sp.todense()

        data = [dns_data, csr_data]
        costs = []
        for d in data:
            dns.wait_to_read()
            d.wait_to_read()
            cost = measure_cost(repeat, mx.nd.dot, d, dns)
            costs.append(cost / repeat)
        ratio = costs[0] / costs[1]

        costs_baseline = []
        cost = measure_cost_forward_baseline(repeat, np.dot, lhs_dns_np, rhs_dns_np)
        costs_baseline.append(cost)
        cost = measure_cost_forward_baseline(repeat, sp.spmatrix.dot, lhs_csr_sp, rhs_dns_np)
        costs_baseline.append(cost)
        ratio_baseline = costs_baseline[0] / costs_baseline[1]
        fmt = "%0.1f\t\t%s\t%d\t%d\t%d\t%0.2f\t\t\t%0.2f\t%0.5f\t%0.2f\t\t\t%0.6f\t\t%0.5f"
        print(fmt % (density * 100, str(ctx), n, m, k, ratio, costs[0], costs[1],
                     ratio_baseline, costs_baseline[0], costs_baseline[1]))

    def bench_dot_backward(m, k, n, density, ctx, repeat):
        set_default_context(ctx)
        dns = mx.nd.random_uniform(shape=(m, n)).copyto(ctx)
        data_shape = (m, k)
        csr_data = rand_ndarray(data_shape, 'csr', density)
        dns_data = csr_data.to_dense()
        rhs_dns_np = dns.asnumpy()
        lhs_csr_sp = sp.csr_matrix(dns_data.asnumpy())
        lhs_dns_np = lhs_csr_sp.todense()

        data = [dns_data, csr_data]
        costs = []
        for d in data:
            dns.wait_to_read()
            d.wait_to_read()
            cost = measure_cost(repeat, mx.nd.dot, d, dns, transpose_a=True)
            costs.append(cost)
        ratio = costs[0] / costs[1]

        costs_baseline = []
        cost = measure_cost_backward_baseline(repeat, np.dot, np.transpose, lhs_dns_np, rhs_dns_np)
        costs_baseline.append(cost)
        cost = measure_cost_backward_baseline(repeat, sp.spmatrix.dot, sp.spmatrix.transpose, lhs_csr_sp, rhs_dns_np)
        costs_baseline.append(cost)
        ratio_baseline = costs_baseline[0] / costs_baseline[1]
        fmt = "%0.1f\t\t%s\t%d\t%d\t%d\t%0.2f\t\t\t%0.2f\t%0.5f\t%0.2f\t\t\t%0.6f\t\t%0.5f"
        print(fmt % (density * 100, str(ctx), n, m, k, ratio, costs[0], costs[1],
                     ratio_baseline, costs_baseline[0], costs_baseline[1]))

    print("A = sparse NDArray of shape(m, k)")
    print("B = dense NDArray of shape(k, n)")
    print("dot_forward\tdot(csr, dns)")
    print('density(%)\t\tcontext\tn\tm\tk\tt_dense/t_sparse\t\t\tt_dense\tt_sparse'
          '\tt_scipy_dense/t_scipy_sparse\t\t\tt_scipy_dense\t\tt_scipy_sparse')

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    # TODO(haibin) make these runtime options
    m = 512
    k = [50000, 100000]
    n = [64, 128]
    density = [1.00, 0.90, 0.70, 0.50, 0.30, 0.20, 0.10, 0.07, 0.05, 0.02, 0.01, 0.005, 0.001]
    num_repeat = 10
    # contexts = [mx.cpu(), mx.gpu(0)]
    contexts = [mx.cpu()]
    for i in range(2):
        for ctx in contexts:
            for den in density:
                bench_dot_forward(m, k[i], n[i], den, ctx, num_repeat)

    print("dot_backward\tdot(csr.T, dns)")
    print('density(%)\tcontext\tn\tm\tk\tt_sparse/t_dense\tt_dense\tt_sparse'
          '\tt_scipy_sparse/t_scipy_dense\tt_scipy_dense\tt_scipy_sparse')
    for i in range(2):
        for ctx in contexts:
            for den in density:
                bench_dot_backward(m, k[i], n[i], den, ctx, num_repeat)

if __name__ == "__main__":
    import pdb
    test_dot_real(batch_size=64)
    test_dot_real(batch_size=128)
#    print "====================="
#    test_dot_synthetic()
