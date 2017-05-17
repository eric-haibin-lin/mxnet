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
            nd, _ = rand_sparse_ndarray(shapes[i], storage_type)
        elif storage_type == 'default_storage':
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
        check_sparse_nd_elemwise_binary(shape, ['default_storage'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default_storage', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

# Test a operator which doesn't implement FComputeEx
def test_sparse_nd_elementwise_fallback():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.add_n
    for i in xrange(num_repeats):
        shape = [(rnd.randint(1, 10), rnd.randint(1, 10))] * 2
        check_sparse_nd_elemwise_binary(shape, ['default_storage'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default_storage', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(shape, stype):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.sparse_nd.zeros('row_sparse', shape)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = (rnd.randint(1, 10), rnd.randint(1, 10))
    check_sparse_nd_zeros(shape, 'row_sparse')
    check_sparse_nd_zeros(shape, 'csr')


def test_sparse_nd_copy():
    def check_sparse_nd_copy(from_stype, to_stype):
        shape = (rnd.randint(1, 10), rnd.randint(1, 10))
        from_nd = rand_ndarray(shape, from_stype)
        # copy to ctx
        to_ctx = from_nd.copyto(default_context())
        # copy to stype
        to_nd = rand_ndarray(shape, to_stype)
        to_nd = from_nd.copyto(to_nd)
        assert np.sum(np.abs(from_nd.asnumpy() != to_ctx.asnumpy())) == 0.0
        assert np.sum(np.abs(from_nd.asnumpy() != to_nd.asnumpy())) == 0.0

    check_sparse_nd_copy('row_sparse', 'row_sparse')
    check_sparse_nd_copy('row_sparse', 'default_storage')
    check_sparse_nd_copy('default_storage', 'row_sparse')
    check_sparse_nd_copy('default_storage', 'csr')

def check_sparse_nd_prop_rsp():
    storage_type = 'row_sparse'
    shape = (rnd.randint(1, 2), rnd.randint(1, 2))
    nd, (v, idx) = rand_sparse_ndarray(shape, storage_type)
    assert(nd._num_aux == 1)
    assert(nd.indices.dtype == np.int32)
    assert(nd.storage_type == 'row_sparse')
    assert_almost_equal(nd._data().asnumpy(), v)
    assert_almost_equal(nd._aux_data(0).asnumpy(), idx)

def test_sparse_nd_basic():
    def check_rsp_creation(values, indices, shape):
        rsp = mx.sparse_nd.row_sparse(values, indices, shape)
        dns = mx.nd.zeros(shape)
        dns[1] = mx.nd.array(values[0])
        dns[3] = mx.nd.array(values[1])
        assert_almost_equal(rsp.asnumpy(), dns.asnumpy())
        indices = mx.nd.array(indices).asnumpy()
        assert_almost_equal(rsp.indices.asnumpy(), indices)

    def check_csr_creation(shape):
        csr, (indptr, indices, values) = rand_sparse_ndarray(shape, 'csr')
        assert_almost_equal(csr.indptr.asnumpy(), indptr)
        assert_almost_equal(csr.indices.asnumpy(), indices)
        assert_almost_equal(csr.values.asnumpy(), values)

    shape = (4,2)
    values = np.random.rand(2,2)
    indices = np.array([1,3])
    check_rsp_creation(values, indices, shape)

    values = mx.nd.array(np.random.rand(2,2))
    indices = mx.nd.array([1,3], dtype='int32')
    check_rsp_creation(values, indices, shape)

    values = [[0.1, 0.2], [0.3, 0.4]]
    indices = [1,3]
    check_rsp_creation(values, indices, shape)

    check_csr_creation(shape)
    check_sparse_nd_prop_rsp()


def test_sparse_nd_setitem():
    shape = (3, 4)
    # ndarray assignment
    x = mx.sparse_nd.zeros('row_sparse', shape)
    x[:] = mx.nd.ones(shape)
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

    # numpy assignment
    x = mx.sparse_nd.zeros('row_sparse', shape)
    x[:] = np.ones(shape)
    x_np = np.ones(shape, dtype=x.dtype)
    assert same(x.asnumpy(), x_np)

def test_sparse_nd_slice():
    def check_sparse_nd_csr_slice(shape):
        storage_type = 'csr'
        A, _ = rand_sparse_ndarray(shape, storage_type)
        A2 = A.asnumpy()
        start = rnd.randint(0, shape[0] - 1)
        end = rnd.randint(start + 1, shape[0])
        assert same(A[start:end].asnumpy(), A2[start:end])

    shape = (rnd.randint(2, 10), rnd.randint(1, 10))
    check_sparse_nd_csr_slice(shape)

if __name__ == '__main__':
    test_sparse_nd_zeros()
    test_sparse_nd_elementwise_fallback()
    test_sparse_nd_copy()
    test_sparse_nd_elemwise_add()
    test_sparse_nd_setitem()
    test_sparse_nd_basic()
    test_sparse_nd_slice()
