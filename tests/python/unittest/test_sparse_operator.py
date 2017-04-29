# pylint: skip-file
import numpy as np
import mxnet as mx
import random
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
    test  = mx.symbol.elemwise_add(data1, data2, name='plus')
    check_symbolic_forward(test, {'data1':dense_nd,
                                  'data2':sparse_nd1}, [dense_np + sparse_np1])

def test_elemwise_add_sparse_sparse():
    # prep data
    shape = (4, 2)
    sparse_np1 = np.array([[5,10],[0,0],[0,0],[0,0]])
    sparse_np2 = np.array([[0,0],[5,10],[0,0],[0,0]])

    val1 = mx.nd.array([5, 10]);
    val2 = mx.nd.array([5, 10]);
    idx1 = mx.nd.array([0], dtype=np.int32);
    idx2 = mx.nd.array([1], dtype=np.int32);
    sparse_nd1 = mx.sparse_nd.row_sparse(val1, idx1, shape)
    sparse_nd2 = mx.sparse_nd.row_sparse(val2, idx2, shape)

    data1 = mx.symbol.Variable('data1', storage_type='row_sparse')
    data2 = mx.symbol.Variable('data2', storage_type='row_sparse')
    test  = mx.symbol.elemwise_add(data1, data2, name='plus')
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
'''
def test_cast_storage():
    dns_np = np.array([[0, 0], [5, 10], [0, 0], [0, 0], [0, 0]])

    val = np.array([5, 10])
    idx = np.array([1])
    b = mx.nd.array(idx, dtype=np.int32)
    sp_nd = mx.sparse_nd.array(val, [b], 'row_sparse', (5,2))
    var = mx.symbol.Variable('sp_data', storage_type='row_sparse')
    # 1 for row_storage type
    test = mx.symbol.cast_storage(var, storage_type=1)
    check_symbolic_forward(test, {'sp_data':sp_nd}, [dns_np])
'''
if __name__ == '__main__':
    test_elemwise_add_dense()
    test_elemwise_add_dense_sparse()
    test_elemwise_add_sparse_sparse()
    test_elemwise_add_multiple_stages()
