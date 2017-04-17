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
    data2_tmp = np.ones(shape)
    test = mx.symbol.elemwise_add(data1, data2)
    # check_numeric_gradient(test, [data_tmp])
    # TODO implement symbolic execution
    check_symbolic_forward(test, {'data1':data1_tmp,
                                  'data2':data2_tmp}, [data1_tmp + data2_tmp])
    #check_symbolic_backward(test, [data_tmp], [np.ones(shape)], [2 * data_tmp])

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
    #dense_np = np.array([[1,2],[3,4],[5,6]])
    sparse_np1 = np.array([[5,10],[0,0],[0,0]])
    sparse_np2 = np.array([[5,10],[0,0],[0,0]])
    #dense_nd = mx.nd.array(dense_np)

    val1 = mx.nd.array([5, 10]);
    val2 = mx.nd.array([5, 10]);
    idx1 = mx.nd.array([0], dtype=np.int32);
    idx2 = mx.nd.array([0], dtype=np.int32);
    sparse_nd1 = mx.sparse_nd.row_sparse(val1, idx1, (3,2))
    sparse_nd2 = mx.sparse_nd.row_sparse(val2, idx2, (3,2))

    data1 = mx.symbol.Variable('data1', storage_type='row_sparse')
    data2 = mx.symbol.Variable('data2', storage_type='row_sparse')
    test  = mx.symbol.elemwise_add(data1, data2, name='plus')
    check_symbolic_forward(test, {'data1':sparse_nd1,
                                  'data2':sparse_nd2}, [sparse_np1 + sparse_np2])

if __name__ == '__main__':
    test_elemwise_add_dense()
    test_elemwise_add_dense_sparse()
    test_elemwise_add_sparse_sparse()
    print("done")
