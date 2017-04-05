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

def test_ndarray_elementwise():
    x = mx.nd.array([1, 2, 3])
    y = mx.nd.COOPlusScalar(x, 1)

    a = mx.nd.array([100]); 
    b = mx.nd.array([0]); 
    c = mx.sparse_nd.array(a, b, 'row_sparse', shape=(1,1)); 
    d = mx.sparse_nd.array(a, b, 'row_sparse', shape=(10,1))

    res = mx.nd.COOPlusScalar(a, 1); 
    res_sparse = mx.nd.COOPlusScalar(d, 1)
    print(y.asnumpy())

if __name__ == '__main__':
    test_ndarray_elementwise() #TODO remove other functions
