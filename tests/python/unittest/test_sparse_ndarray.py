import os
import random
import mxnet as mx
import numpy as np
import pickle as pkl
from mxnet.test_utils import *
from numpy.testing import assert_allclose

def check_sparse_nd_elemwise_binary(shapes, storage_types, f, g):
    # generate inputs
    nds = []
    for i, storage_type in enumerate(storage_types):
        if storage_type == 'row_sparse':
            nd = random_sparse_ndarray(shapes[i], storage_type, allow_zeros = True)
        elif storage_type == 'default':
            nd = mx.nd.array(random_arrays(shapes[i]), dtype = np.float32)
        else:
            assert(False)
        nds.append(nd)
    # check result
    test = f(nds[0], nds[1])
    assert_almost_equal(test.asnumpy(), g(nds[0].asnumpy(), nds[1].asnumpy()))

def test_sparse_nd_elemwise_add():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.elemwise_add
    for i in xrange(num_repeats):
        shape = [(random.randint(1, 10),random.randint(1, 10))] * 2
        check_sparse_nd_elemwise_binary(shape, ['default'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

# Test a operator which doesn't implement FComputeEx
def test_sparse_nd_elementwise_fallback():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.add_n
    for i in xrange(num_repeats):
        shape = [(random.randint(1, 10),random.randint(1, 10))] * 2
        check_sparse_nd_elemwise_binary(shape, ['default'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

def check_conversion_row_sparse():
    val = np.array([5, 10])
    idx = np.array([1])
    sparse_val = np.array([[0, 0], [5, 10], [0, 0], [0, 0], [0, 0]])
    a = mx.nd.array(val)
    b = mx.nd.array(idx, dtype=np.int32)
    d = mx.sparse_nd.array(a, [b], 'row_sparse', (5,2))
    f = mx.sparse_nd.to_dense(d)
    assert_almost_equal(f.asnumpy(), sparse_val)

def test_sparse_nd_conversion():
    check_conversion_row_sparse()

def test_sparse_nd_zeros():
    zero = mx.nd.zeros((2,2))
    sparse_zero = mx.sparse_nd.zeros((2,2), 'row_sparse')
    assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

def check_sparse_nd_copy(from_stype, to_stype):
    shape = (random.randint(1, 10), random.randint(1, 10))
    from_nd = random_ndarray(shape, from_stype)
    # copy to ctx
    to_ctx = from_nd.copyto(default_context())
    # copy to stype
    to_stype = random_ndarray(shape, to_stype)
    to_stype = from_nd.copyto(to_stype)
    assert np.sum(np.abs(from_nd.asnumpy() != to_ctx.asnumpy())) == 0.0
    assert np.sum(np.abs(from_nd.asnumpy() != to_stype.asnumpy())) == 0.0

def test_sparse_nd_copy():
    check_sparse_nd_copy('row_sparse', 'row_sparse')
    check_sparse_nd_copy('row_sparse', 'default')
    check_sparse_nd_copy('default', 'row_sparse')

def test_sparse_nd_property():
    storage_type = 'row_sparse'
    a = random_sparse_ndarray((10, 10), storage_type, allow_zeros = True)
    assert(a.num_aux == 1)
    assert(a.aux_type(0) == np.int32)
    assert(a.storage_type == 'row_sparse')

if __name__ == '__main__':
    test_sparse_nd_conversion()
    test_sparse_nd_zeros()
    test_sparse_nd_elementwise_fallback()
    test_sparse_nd_copy()
    test_sparse_nd_elemwise_add()
    test_sparse_nd_property()
