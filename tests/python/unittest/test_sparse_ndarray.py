import os
import mxnet as mx
import numpy as np
import pickle as pkl
from mxnet.test_utils import *
from numpy.testing import assert_allclose
import numpy.random as rnd

def check_sparse_nd_elemwise_binary(shapes, storage_types, f, g):
    # generate inputs
    nds = []
    for i, storage_type in enumerate(storage_types):
        if storage_type == 'row_sparse':
            nd, _, _ = rand_sparse_ndarray(shapes[i], storage_type, allow_zeros = True)
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
        shape = [(rnd.randint(1, 10),rnd.randint(1, 10))] * 2
        check_sparse_nd_elemwise_binary(shape, ['default'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

# Test a operator which doesn't implement FComputeEx
def test_sparse_nd_elementwise_fallback():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.add_n
    for i in xrange(num_repeats):
        shape = [(rnd.randint(1, 10), rnd.randint(1, 10))] * 2
        check_sparse_nd_elemwise_binary(shape, ['default'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

def test_sparse_nd_zeros():
    shape = (rnd.randint(1, 10), rnd.randint(1, 10))
    zero = mx.nd.zeros(shape)
    sparse_zero = mx.sparse_nd.zeros(shape, 'row_sparse')
    assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

def check_sparse_nd_copy(from_stype, to_stype):
    shape = (rnd.randint(1, 10), rnd.randint(1, 10))
    from_nd = rand_ndarray(shape, from_stype)
    # copy to ctx
    to_ctx = from_nd.copyto(default_context())
    # copy to stype
    to_stype = rand_ndarray(shape, to_stype)
    to_stype = from_nd.copyto(to_stype)
    assert np.sum(np.abs(from_nd.asnumpy() != to_ctx.asnumpy())) == 0.0
    assert np.sum(np.abs(from_nd.asnumpy() != to_stype.asnumpy())) == 0.0

def test_sparse_nd_copy():
    check_sparse_nd_copy('row_sparse', 'row_sparse')
    check_sparse_nd_copy('row_sparse', 'default')
    check_sparse_nd_copy('default', 'row_sparse')

def check_sparse_nd_prop_rsp():
    storage_type = 'row_sparse'
    shape = (rnd.randint(1, 2), rnd.randint(1, 2))
    nd, v, idx = rand_sparse_ndarray(shape, storage_type, allow_zeros = True)
    assert(nd.num_aux == 1)
    assert(nd.aux_type(0) == np.int32)
    assert(nd.storage_type == 'row_sparse')
    #if v is not None:
    if v is None:
      v = mx.nd.array([]).asnumpy()
    if idx is None:
      idx = mx.nd.array([]).asnumpy()
    assert_almost_equal(nd._data().asnumpy(), v)
    # TODO(haibin) test data
    #assert_almost_equal(nd._aux_data(0).asnumpy(), idx)

def test_sparse_nd_property():
    check_sparse_nd_prop_rsp()

def test_sparse_nd_setitem():
    shape = (3, 4)
    # ndarray assignment
    x = mx.sparse_nd.zeros(shape, storage_type='row_sparse')
    x[:] = mx.nd.ones(shape)
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

    # numpy assignment
    x = mx.sparse_nd.zeros(shape, storage_type='row_sparse')
    x[:] = np.ones(shape)
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

if __name__ == '__main__':
    test_sparse_nd_zeros()
    test_sparse_nd_elementwise_fallback()
    test_sparse_nd_copy()
    test_sparse_nd_elemwise_add()
    test_sparse_nd_setitem()
    test_sparse_nd_property()
