import os
import mxnet as mx
import numpy as np
import pickle as pkl
from mxnet.test_utils import *
from numpy.testing import assert_allclose

def check_with_uniform(uf, arg_shapes, dim=None, npuf=None, rmin=-10, type_list=[np.float32]):
    """check function consistency with uniform random numbers"""
    if isinstance(arg_shapes, int):
        assert dim
        shape = tuple(np.random.randint(1, int(1000**(1.0/dim)), size=dim))
        arg_shapes = [shape] * arg_shapes
    for dtype in type_list:
        ndarray_arg = []
        numpy_arg = []
        for s in arg_shapes:
            npy = np.random.uniform(rmin, 10, s).astype(dtype)
            narr = mx.nd.array(npy, dtype=dtype)
            ndarray_arg.append(narr)
            numpy_arg.append(npy)
        out1 = uf(*ndarray_arg)
        if npuf is None:
            out2 = uf(*numpy_arg).astype(dtype)
        else:
            out2 = npuf(*numpy_arg).astype(dtype)

        assert out1.shape == out2.shape
        if isinstance(out1, mx.nd.NDArray):
            out1 = out1.asnumpy()
        if dtype == np.float16:
            assert_almost_equal(out1, out2, rtol=2e-3)
        else:
            assert_almost_equal(out1, out2)

def test_ndarray_elementwise_add():
    dense_np = np.array([[1,2],[3,4],[5,6]])
    sparse_np1 = np.array([[5,10],[0,0],[0,0]])
    dense_nd = mx.nd.array(dense_np)

    val = mx.nd.array([5, 10]); 
    idx = mx.nd.array([0], dtype=np.int32); 
    sparse_nd1 = mx.sparse_nd.row_sparse(val, idx, (3,2)) 
    sparse_nd2 = mx.sparse_nd.row_sparse(val, idx, (3,2)) 
    #TODO register under mx.sparse_nd namespace
    # dense - dense addition
    dense_plus_dense = mx.nd.elemwise_add(dense_nd, dense_nd);
    assert_almost_equal(dense_plus_dense.asnumpy(), dense_np + dense_np)
    # dense - sparse addition
    dense_plus_sparse = mx.nd.elemwise_add(dense_nd, sparse_nd1)
    assert_almost_equal(dense_plus_sparse.asnumpy(), dense_np + sparse_np1)
    # sparse - sparse addition
    sparse_plus_sparse = mx.nd.elemwise_add(sparse_nd1, sparse_nd2)
    assert_almost_equal(sparse_plus_sparse.asnumpy(), sparse_np1 + sparse_np1)

def test_ndarray_elementwise_fallback():
    dense_np = np.array([[1,2],[3,4],[5,6]])
    sparse_np1 = np.array([[5,10],[0,0],[0,0]])
    dense_nd = mx.nd.array(dense_np)

    val = mx.nd.array([5, 10]);
    idx = mx.nd.array([0], dtype=np.int32);
    sparse_nd1 = mx.sparse_nd.row_sparse(val, idx, (3,2))
    # dense - dense addition
    dense_plus_dense = mx.nd.add_n(dense_nd, dense_nd);
    assert_almost_equal(dense_plus_dense.asnumpy(), dense_np + dense_np)
    
    # dense - sparse addition
    dense_plus_sparse = mx.nd.add_n(dense_nd, sparse_nd1)
    assert_almost_equal(dense_plus_sparse.asnumpy(), dense_np + sparse_np1)

    # sparse - sparse addition
    sparse_plus_sparse = mx.nd.add_n(sparse_nd1, sparse_nd1)
    assert_almost_equal(sparse_plus_sparse.asnumpy(), sparse_np1 + sparse_np1)

def check_conversion_row_sparse():
    val = np.array([5, 10])
    idx = np.array([1])
    sparse_val = np.array([[0, 0], [5, 10], [0, 0], [0, 0], [0, 0]])
    a = mx.nd.array(val)
    b = mx.nd.array(idx, dtype=np.int32)
    d = mx.sparse_nd.array(a, [b], 'row_sparse', (5,2))
    f = mx.sparse_nd.to_dense(d)
    assert_almost_equal(f.asnumpy(), sparse_val)

def check_conversion_csr():
    val = mx.nd.array([1, 2, 3, 4, 5, 6])
    indices = mx.nd.array([0, 2, 2, 0, 1, 2], dtype=np.int32)
    indptr = mx.nd.array([0, 2, 3, 6], dtype=np.int32)
    shape = (3, 3)
    #sparse_val = np.array([[0, 0], [5, 10], [0, 0], [0, 0], [0, 0]])
    d = mx.sparse_nd.csr(val, indices, indptr, (5,2))
    #f = mx.sparse_nd.to_dense(d)
    #assert_almost_equal(f.asnumpy(), sparse_val)

def test_ndarray_conversion():
    check_conversion_row_sparse()
    #TODO check_conversion_csr()

def test_ndarray_zeros():
    zero = mx.nd.zeros((2,2))
    sparse_zero = mx.sparse_nd.zeros((2,2), 'row_sparse')
    assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

def test_ndarray_copyto():
    zero = mx.nd.zeros((2,2))
    e = mx.nd.ones((2,2))
    zero.copyto(e)

if __name__ == '__main__':
    test_ndarray_elementwise_add()
    test_ndarray_conversion()
    test_ndarray_zeros()
    test_ndarray_copyto()
    test_ndarray_elementwise_fallback()
