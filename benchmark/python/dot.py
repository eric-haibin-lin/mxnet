import ctypes

from mxnet.test_utils import *
import scipy.sparse as sp
import os
import time
import argparse
import subprocess

from mxnet.base import check_call, _LIB
from util import get_data, estimate_density

parser = argparse.ArgumentParser(description="Benchmark sparse operators",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num-omp-threads', type=int, default=1, help='number of omp threads to set in MXNet')
args = parser.parse_args()

# some data information
kdda = {
    'data_mini': 'kdda.t.mini',
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216830,
    'm': [1, 8, 32],
    'batch_size': [64, 128],
    'default_index': {'batch_size': 1,
                       'output_dim': 2},
    'num_batches': 10
}

avazu = {
    'data_mini': 'avazu-app.t.mini',
    'data_name': 'avazu-app.t',
    'data_origin_name': 'avazu-app.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2",
    'feature_dim': 1000000,
    'm': [1, 1000, 2000],
    'batch_size': [128, 256],
    'default_index': {'batch_size': 0,
                      'output_dim': 1},
    'num_batches': 10
}

criteo = {
    'data_mini': 'criteo.t.mini',
    'data_name': 'criteo.t',
    'data_origin_name': 'criteo.t.bz2',
    'url' : "https://s3-us-west-2.amazonaws.com/sparse-dataset/criteo.t.bz2",
    'feature_dim': 16000000,
    'm': [1, 8, 16, 32, 64],
    'batch_size': [64, 128],
    'default_index': {'batch_size': 1,
                      'output_dim': 3},
    'num_batches': 10
}


def measure_cost(wait, repeat, f, *args, **kwargs):
    start = time.time()
    if wait:
        for i in range(repeat):
            (f(*args, **kwargs)).wait_to_read()
    else:
        for i in range(repeat):
            f(*args, **kwargs)
    end = time.time()
    diff = end - start
    return diff / repeat

def _get_iter(path, data_shape, batch_size):
    data_train = mx.io.LibSVMIter(data_libsvm=path,
                                  data_shape=data_shape,
                                  batch_size=batch_size)
    data_iter = iter(data_train)
    return data_iter

def _line_count(path):
    return int(subprocess.check_output('wc -l {}'.format(path), shell=True).split()[0])


def _compare_sparse_dense(data_dir, file_name, mini_file_name, feature_dim,  output_dim, density, batch_size, num_batches=3, num_repeat=5):

    def create_mini_path(mini_path, path, num_batches):
        if not os.path.exists(mini_path):
            last = _line_count(path) - num_batches * batch_size
            last = last if last >= 1 else 1
            start = int(rnd.uniform(1, last))
            os.system("sed -n '%d,%dp' %r > %r" %(start,start + num_batches * batch_size, path, mini_path))
            assert os.path.exists(mini_path)


    def run_benchmark(mini_path):
        #print("Running Benchmarking on %r data") % mini_file_name
        #print("batch_size is %d") % batch_size
        data_shape = (feature_dim, )
        train_iter = _get_iter(mini_path, data_shape, batch_size)
        weight = mx.nd.random_uniform(low=0, high=1, shape=(feature_dim, output_dim))
        total_cost = {}
        average_cost = {}
        count = 0
        total_cost["sparse"] = 0.
        total_cost["dense"] = 0.
        weight.wait_to_read()
        for batch in train_iter:
            csr_data = train_iter.getdata()
            dns_data = csr_data.todense()
            csr_data.wait_to_read()
            dns_data.wait_to_read()
            cost_sparse = measure_cost(True, num_repeat, mx.nd.dot, csr_data, weight)
            cost_dense = measure_cost(True, num_repeat, mx.nd.dot, dns_data, weight)
            total_cost["sparse"] +=  cost_sparse
            total_cost["dense"]  += cost_dense
            count = count + 1
        average_cost["sparse"] = total_cost["sparse"] / count
        average_cost["dense"] = total_cost["dense"] / count
        return (average_cost["sparse"], average_cost["dense"])


    def print_result(average_cost_sparse, average_cost_dense):
        ratio = average_cost_dense / average_cost_sparse
        print('density(%)\tn\tm\tk\tt_dense/t_sparse\tt_dense\tt_sparse')
        fmt = "%0.4f\t\t%d\t%d\t%d\t%0.2f\t\t\t%0.4f\t%0.6f"
        print(fmt % (density * 100, batch_size, output_dim, feature_dim,
              ratio, average_cost_dense, average_cost_sparse))

    mini_path = os.path.join(data_dir, mini_file_name)
    path = os.path.join(data_dir, file_name)
    create_mini_path(mini_path, path, num_batches)
    average_cost_sparse, average_cost_dense = run_benchmark(mini_path)
    print_result(average_cost_sparse, average_cost_dense)

def test_dot_real(data_dict):
    data_dir = os.path.join(os.getcwd(), 'data')

    path = os.path.join(data_dir, data_dict['data_name'])
    if not os.path.exists(path):
        get_data(
            data_dir,
            data_dict['data_name'],
            data_dict['url'],
            data_dict['data_origin_name']
        )
        assert os.path.exists(path)

    k = data_dict['feature_dim']
    m = data_dict['m']
    batch_size_list = data_dict['batch_size']

    default_output_index = data_dict['default_index']['output_dim']
    default_batch_size_index = data_dict['default_index']['batch_size']
    density = estimate_density(path, data_dict['feature_dim'])
    num_batches = data_dict['num_batches']

    assert default_batch_size_index < len(batch_size_list)
    assert default_output_index < len(m)

    for output_dim in m:
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, output_dim, density,
                              batch_size_list[default_batch_size_index], num_batches)

    for batch_size in batch_size_list:
        _compare_sparse_dense(data_dir, data_dict['data_name'], data_dict['data_mini'],
                              k, m[default_output_index], density, batch_size, num_batches)


def test_dot_synthetic():
    """benchmark sparse mxnet dot and scipy dot operator with matrices of given density.
    `t_sparse` is the runtime of the invoked sparse dot operator in ms, while `t_dense` is the 
    runtime of dot(dns, dns), with the same matrices except that they are in default storage type.
    """
    # Benchmark MXNet's sparse dot operator
    def bench_mx_dot(lhs_shape, rhs_shape, lhs_stype, rhs_stype, lhs_den, rhs_den, trans_lhs, ctx, repeat):
        set_default_context(ctx)
        # Create matrix instances
        lhs_nd = rand_ndarray(lhs_shape, lhs_stype, density=lhs_den)
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_den)
        lhs_dns = lhs_nd if lhs_stype == 'default' else lhs_nd.todense()
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.todense()
        # One warm up run, verify correctness
        out = mx.nd.dot(lhs_nd, rhs_dns, trans_lhs)
        out_expected = mx.nd.dot(lhs_dns, rhs_dns, trans_lhs)
        assert_almost_equal(out.asnumpy(), out_expected.asnumpy(), rtol=1e-2, atol=1e-3)
        # Start benchmarking
        lhs_nd.wait_to_read()
        rhs_nd.wait_to_read()
        sparse_cost = measure_cost(True, repeat, mx.nd.dot, lhs_nd, rhs_nd, trans_lhs)
        dense_cost = measure_cost(True, repeat, mx.nd.dot, lhs_dns, rhs_dns, trans_lhs)
        speedup = dense_cost / sparse_cost
        # Print results
        m = lhs_shape[0]
        k = lhs_shape[1]
        n = rhs_shape[1]
        results = '{:15.1f} {:15.1f} {:>10} {:8d} {:8d} {:8d} {:13.2f} {:13.2f} {:8.2f}'.format(lhs_den*100, rhs_den*100, str(ctx), m, k, n, sparse_cost*1000, dense_cost*1000, speedup)
        print(results)

    # Benchmark Scipy's sparse dot operator
    def bench_sp_dot(lhs_shape, rhs_shape, lhs_stype, rhs_stype, lhs_den, rhs_den, trans_lhs, ctx, repeat):
        set_default_context(ctx)
        assert default_context().device_type is 'cpu'
        assert lhs_stype is 'csr'
        assert rhs_stype is 'default'
        # Create matrix instances
        lhs_nd = rand_ndarray(lhs_shape, lhs_stype, density=lhs_den)
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=rhs_den)
        lhs_nd.wait_to_read()
        rhs_nd.wait_to_read()
        lhs_dns_np = np.transpose(lhs_nd.asnumpy()) if trans_lhs else lhs_nd.asnumpy()
        rhs_dns_np = rhs_nd.asnumpy()
        lhs_csr_sp = sp.spmatrix.transpose(sp.csr_matrix(lhs_nd.asnumpy())) if trans_lhs else sp.csr_matrix(lhs_nd.asnumpy())
        # One warm up run
        out = sp.spmatrix.dot(lhs_csr_sp, rhs_dns_np)
        # Start benchmarking
        sparse_cost = measure_cost(False, repeat, sp.spmatrix.dot, lhs_csr_sp, rhs_dns_np)
        dense_cost = measure_cost(False, repeat, np.dot, lhs_dns_np, rhs_dns_np)
        speedup = dense_cost / sparse_cost
        # Print results
        m = lhs_shape[0]
        k = lhs_shape[1]
        n = rhs_shape[1]
        results = '{:15.1f} {:15.1f} {:>10} {:8d} {:8d} {:8d} {:13.2f} {:13.2f} {:8.2f}'.format(lhs_den*100, rhs_den*100, str(ctx), m, k, n, sparse_cost*1000, dense_cost*1000, speedup)
        print(results)

    check_call(_LIB.MXSetNumOMPThreads(ctypes.c_int(args.num_omp_threads)))
    # TODO(haibin): make these runtime options
    # params
    # m, n, k        rows and columns of lhs and rhs matrix
    #                forward  pass:  m x k    * k x n = m x n
    #                backward pass: (m x k)^T * m x n = k x n
    # density_lhs    density of the left-hand side matrix
    # density_rhs    density of the right-hand side matrix, if applicable
    # num_repeat     number of benchmark runs to average over
    # context        mx.cpu(), mx.gpu()
    #                note: benchmark different contexts separately; to benchmark cpu, compile without CUDA
    # mx_benchmarks  csr_dns, csr.T_dns, csr_rsp
    # sp_benchmarks  csr_dns, csr.T_dns
    #                note: scipy benchmarks are only conducted if context is mx.cpu()
    m = 512
    k = [50000, 100000]
    n = [64, 128]
    density_lhs = [0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
    density_rhs = [0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
    num_repeat = 10
    context = mx.cpu()
    mx_benchmarks = ["csr_dns", "csr.T_dns", "csr_rsp"]
    sp_benchmarks = ["csr_dns", "csr.T_dns"]

    headline = '{:>15} {:>15} {:>10} {:>8} {:>8} {:>8} {:>13} {:>13} {:>8}'.format('lhs_density(%)', 'rhs_density(%)', 'context', 'm', 'k', 'n', 't_sparse(ms)', 't_dense(ms)', 'speedup')
    if "csr_dns" in mx_benchmarks:
        print("==================================================")
        print("  mxnet sparse dot benchmark: dot(csr, dns) = dns ")
        print("  (matrix multiplication: m x k * k x n = m x n)  ")
        print("==================================================")
        print(headline)
        transpose_lhs = False
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_mx_dot((m, k[i]), (k[i], n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print("")

    if "csr_dns" in sp_benchmarks and mx.cpu() == context:
        print("==================================================")
        print("  scipy sparse dot benchmark: dot(csr, dns) = dns ")
        print("  (matrix multiplication: m x k * k x n = m x n)  ")
        print("==================================================")
        print(headline)
        transpose_lhs = False
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_sp_dot((m, k[i]), (k[i], n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print("")

    if "csr.T_dns" in mx_benchmarks:
        print("==================================================")
        print(" mxnet sparse dot benchmark: dot(csr.T, dns) = rsp")
        print("(matrix multiplication: (m x k)^T * m x n = k x n)")
        print("==================================================")
        print(headline)
        transpose_lhs = True
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_mx_dot((m, k[i]), (m, n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print("")

    if "csr.T_dns" in sp_benchmarks and mx.cpu() == context:
        print("==================================================")
        print(" scipy sparse dot benchmark: dot(csr.T, dns) = dns")
        print("(matrix multiplication: (m x k)^T * m x n = k x n)")
        print("==================================================")
        print(headline)
        transpose_lhs = True
        for i in range(len(n)):
            for d_lhs in density_lhs:
                bench_sp_dot((m, k[i]), (m, n[i]), 'csr', 'default', d_lhs, 1, transpose_lhs, context, num_repeat)
            print("")

    if "csr_rsp" in mx_benchmarks:
        print("==================================================")
        print("  mxnet sparse dot benchmark: dot(csr, rsp) = dns ")
        print("  (matrix multiplication: m x k * k x n = m x n)  ")
        print("==================================================")
        print(headline)
        transpose_lhs = False
        for i in range(len(n)):
            for d_lhs in density_lhs:
              for d_rhs in density_rhs:
                bench_mx_dot((m, k[i]), (k[i], n[i]), 'csr', 'row_sparse', d_lhs, d_rhs, transpose_lhs, context, num_repeat)
              print("")
            print("")


if __name__ == "__main__":
    #test_dot_synthetic()
    start_time = time.time()
    #test_dot_real(kdda)
    #test_dot_real(avazu)
    test_dot_real(criteo)
    end_time = time.time() - start_time
    print("total time is %f") % end_time
    #test_dot_real(kdda)
