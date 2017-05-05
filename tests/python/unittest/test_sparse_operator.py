# pylint: skip-file
import numpy as np
import mxnet as mx
from numpy.testing import assert_allclose
from mxnet.test_utils import *


def test_elemwise_add_dense():
    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2')
    shape = (1, 1)
    data1_tmp = np.ones(shape)
    data2_tmp = np.zeros(shape) + 2
    test = mx.symbol.elemwise_add(data1, data2)
    # check_numeric_gradient(test, [data_tmp])
    check_symbolic_forward(test, {'data1':data1_tmp,
                                  'data2':data2_tmp}, [data1_tmp + data2_tmp])
    #check_symbolic_backward(test, [data_tmp], [np.ones(shape)], [2 * data_tmp])
    arr_grad1 = mx.nd.empty(shape)
    arr_grad2 = mx.nd.empty(shape)
    # init grad arrays before bind
    exec_test = test.bind(default_context(), args={'data1':mx.nd.array(data1_tmp), 'data2':mx.nd.array(data2_tmp)},
                          args_grad=[arr_grad1, arr_grad2])
    exec_test.forward(is_train=True)
    assert_almost_equal(exec_test.outputs[0].asnumpy(), data1_tmp + data2_tmp)
    exec_test.backward(out_grads = exec_test.outputs)
    assert_almost_equal(arr_grad1.asnumpy(), arr_grad2.asnumpy())


def test_elemwise_add_dense_sparse():
    # prep data
    dense_np = np.array([[1,2],[3,4],[5,6]])
    sparse_np1 = np.array([[5,10],[0,0],[0,0]])
    dense_nd = mx.nd.array(dense_np)

    val = mx.nd.array([5, 10]);
    idx = mx.nd.array([0], dtype=np.int32);
    sparse_nd1 = mx.sparse_nd.row_sparse(val, idx, (3,2))

    data1 = mx.symbol.Variable('data1')
    data2 = mx.symbol.Variable('data2', storage_type='row_sparse')
    test = mx.symbol.elemwise_add(data1, data2, name='plus')
    check_symbolic_forward(test, {'data1':dense_nd,
                                  'data2':sparse_nd1}, [dense_np + sparse_np1])


def test_elemwise_add_sparse_sparse():
    # prep data
    shape = (4, 2)
    sparse_np1 = np.array([[5,10],[0,0],[0,0],[0,0]])
    sparse_np2 = np.array([[0,0],[5,10],[0,0],[0,0]])

    val1 = mx.nd.array([5, 10])
    val2 = mx.nd.array([5, 10])
    idx1 = mx.nd.array([0], dtype=np.int32);
    idx2 = mx.nd.array([1], dtype=np.int32);
    sparse_nd1 = mx.sparse_nd.row_sparse(val1, idx1, shape)
    sparse_nd2 = mx.sparse_nd.row_sparse(val2, idx2, shape)

    data1 = mx.symbol.Variable('data1', storage_type='row_sparse')
    data2 = mx.symbol.Variable('data2', storage_type='row_sparse')
    test = mx.symbol.elemwise_add(data1, data2, name='plus')
    check_symbolic_forward(test, {'data1':sparse_nd1,
                                  'data2':sparse_nd2}, [sparse_np1 + sparse_np2])
    arr_grad1 = mx.sparse_nd.zeros(shape, 'row_sparse')
    arr_grad2 = mx.sparse_nd.zeros(shape, 'row_sparse')
    exec_test = test.bind(default_context(), args={'data1':sparse_nd1, 'data2':sparse_nd2},
                          args_grad=[arr_grad1, arr_grad2])
    exec_test.forward(is_train=True)
    assert_almost_equal(exec_test.outputs[0].asnumpy(), sparse_np1 + sparse_np2)
    exec_test.backward(out_grads = exec_test.outputs)
    assert_almost_equal(arr_grad1.asnumpy(), arr_grad2.asnumpy())


def test_elemwise_add_multiple_stages():
    # prep data
    shape = (4, 2)
    ds_np = np.array([[1,2],[3,4],[5,6],[7,8]])
    sp_np1 = np.array([[5,10],[0,0],[0,0],[0,0]])
    sp_np2 = np.array([[0,0],[5,10],[0,0],[0,0]])

    val1 = mx.nd.array([5, 10]);
    val2 = mx.nd.array([5, 10]);
    idx1 = mx.nd.array([0], dtype=np.int32);
    idx2 = mx.nd.array([1], dtype=np.int32);
    sp_nd1 = mx.sparse_nd.row_sparse(val1, idx1, shape)
    sp_nd2 = mx.sparse_nd.row_sparse(val2, idx2, shape)
    ds_nd = mx.nd.array(ds_np)

    # sparse + sparse = sparse
    sp_data1 = mx.symbol.Variable('sp_data1', storage_type='row_sparse')
    sp_data2 = mx.symbol.Variable('sp_data2', storage_type='row_sparse')
    ds_data = mx.symbol.Variable('ds_data')
    plus  = mx.symbol.elemwise_add(sp_data1, sp_data2, name='plus')
    # sparse + dense = dense
    test  = mx.symbol.elemwise_add(plus, ds_data)
    check_symbolic_forward(test, {'sp_data1':sp_nd1, 'sp_data2':sp_nd2,
                          'ds_data':ds_nd}, [sp_np1 + sp_np2 + ds_np])

    arr_grads = [mx.nd.zeros(shape) for i in xrange(3)]
    exec_test = test.bind(default_context(), args={'sp_data1':sp_nd1, 'sp_data2':sp_nd2,
                          'ds_data':ds_nd}, args_grad=arr_grads)
    exec_test.forward(is_train=True)
    assert_almost_equal(exec_test.outputs[0].asnumpy(), sp_np1 + sp_np2 + ds_np)
    exec_test.backward(out_grads = exec_test.outputs)
    assert_almost_equal(arr_grads[0].asnumpy(), arr_grads[1].asnumpy())


def test_cast_storage():
    def test_rsp_to_dns(data, row_idx, shape):
        rsp = mx.sparse_nd.array(values=data, index_list=[row_idx], storage_type='row_sparse', shape=shape, aux_types = [np.int32])
        dns_out = mx.nd.cast_storage(rsp, storage_type='default')
        dns_expected = np.zeros(shape, dtype=default_dtype())
        for k, v in enumerate(row_idx):
            dns_expected[v, :] = data[k]
        assert same(dns_out.asnumpy(), dns_expected)

    def test_dns_to_rsp(dns_in):
        dns_in = np.array(dns_in)
        rsp_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), storage_type='row_sparse')
        ret = mx.nd.cast_storage(rsp_out, storage_type='default')
        assert same(ret.asnumpy(), dns_in)

    def test_csr_to_dns(data, indptr, col_idx, shape):
        indptr = np.array(indptr, dtype=np.int32)
        col_idx = np.array(col_idx, dtype=np.int32)
        csr = mx.sparse_nd.array(values=data, index_list=[col_idx, indptr], storage_type='csr', shape=shape,
                                 aux_types=[np.int32, np.int32])
        dns_out = mx.nd.cast_storage(csr, storage_type='default')
        dns_expected = np.zeros(shape, dtype=default_dtype())
        i = 0
        while i < len(indptr) - 1:
            j = indptr[i]
            while j < indptr[i+1]:
                dns_expected[i, col_idx[j]] = data[j]
                j = j + 1
            i = i + 1
        assert same(dns_out.asnumpy(), dns_expected)

    def test_dns_to_csr(dns_in):
        dns_in= np.array(dns_in)
        csr_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), storage_type='csr')
        ret = mx.nd.cast_storage(csr_out, storage_type='default')
        assert same(ret.asnumpy(), dns_in)

    test_rsp_to_dns([], [], (10, 3))
    test_rsp_to_dns(random_arrays((4,2)), [2, 4, 5, 7], (10, 2))
    test_dns_to_rsp([[0, 1, 0], [0, 2, 0], [3, 0, 0], [0, 0, 4], [5, 6, 0], [0, 0, 7]])
    test_csr_to_dns([], [0, 0, 0, 0, 0], [], (4, 4))
    test_csr_to_dns([5, 8, 3, 6], [0, 0, 2, 3, 4], [0, 1, 2, 1], (4, 4))
    test_dns_to_csr([[0, 1, 0], [0, 2, 0], [3, 0, 0], [0, 0, 4], [5, 6, 0], [0, 0, 7]])

# TODO(junwu): The backward of the operator dot cannot be tested for now
# since the backend function CopyFromTo does not support taking two arguments
# of the different storage types. Will add backward test after removing this
# restriction on CopyFromTo(@haibin). Nevertheless, both backward and forward use
# the same impl function of dot(csr, dns) = rsp and it has been tested
# in the forward test cases as the following.
def test_sparse_dot():
    def test_dot_csr_dns_rsp(dns1, dns2, trans_csr):
        dns1 = mx.nd.array(dns1)
        dns2 = mx.nd.array(dns2)
        csr = mx.nd.cast_storage(dns1, storage_type='csr')
        rsp_out = mx.nd.dot(csr, dns2, transpose_a=trans_csr)
        rsp_expected = mx.nd.dot(csr.to_dense(), dns2, transpose_a=trans_csr)
        # TODO(junwu): may need to compare rsp_out and rsp_expected in rsp format
        # instead of converting them to the dense format
        assert same(rsp_out.asnumpy(), rsp_expected.asnumpy())

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', storage_type='csr')
        rhs = mx.symbol.Variable('rhs', storage_type='default')
        sym_dot = mx.symbol.dot(lhs, rhs, transpose_a=trans_csr)
        dns2_grad = mx.sparse_nd.zeros(dns2.shape, 'row_sparse')
        exec_dot = sym_dot.bind(default_context(), args={'lhs': csr, 'rhs': dns2}, args_grad={'rhs': dns2_grad},
                                grad_req={'lhs': 'null', 'rhs': 'write'})
        exec_dot.forward(is_train=True)
        assert same(exec_dot.outputs[0].asnumpy(), rsp_expected.asnumpy())

    test_dot_csr_dns_rsp(dns1=[[0, 0, 1, 4], [2, 0, 0, 0], [0, 0, 0, 0], [2, 9, 0, 5], [0, 0, 0, 1]],
                         dns2=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                         trans_csr=False)
    test_dot_csr_dns_rsp(dns1=[[0, 0, 1, 4], [2, 0, 0, 0], [0, 0, 0, 0], [2, 9, 0, 5], [0, 0, 0, 1]],
                         dns2=[[1, 2, 3, 4, 5], [5, 6, 7, 8, 6], [9, 10, 11, 12, 6], [13, 14, 15, 16, 7],
                               [1, 1, 1, 1, 2]], trans_csr=True)


if __name__ == '__main__':
    test_elemwise_add_dense()
    test_elemwise_add_dense_sparse()
    test_elemwise_add_sparse_sparse()
    test_elemwise_add_multiple_stages()
    test_cast_storage()
    test_sparse_dot()
