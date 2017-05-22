import os
import mxnet as mx
import numpy as np
import pickle as pkl
from mxnet.test_utils import *
from numpy.testing import assert_allclose
import numpy.random as rnd

def assert_fcompex(f, *args, **kwargs):
    prev_val = mx.test_utils.set_env_var("MXNET_EXEC_STORAGE_FALLBACK", "0", "1")
    f(*args, **kwargs)
    mx.test_utils.set_env_var("MXNET_EXEC_STORAGE_FALLBACK", prev_val)

def rand_shape_2d():
    return (rnd.randint(1, 10), rnd.randint(1, 10))

def sparse_nd_zeros(shape, stype):
    return mx.nd.cast_storage(mx.nd.zeros(shape), storage_type=stype)

def sparse_nd_ones(shape, stype):
    return mx.nd.cast_storage(mx.nd.ones(shape), storage_type=stype)

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
    for i in range(num_repeats):
        shape = [rand_shape_2d()] * 2
        assert_fcompex(check_sparse_nd_elemwise_binary,
                       shape, ['default_storage'] * 2, op, g)
        assert_fcompex(check_sparse_nd_elemwise_binary,
                       shape, ['default_storage', 'row_sparse'], op, g)
        assert_fcompex(check_sparse_nd_elemwise_binary,
                       shape, ['row_sparse', 'row_sparse'], op, g)

# Test a operator which doesn't implement FComputeEx
def test_sparse_nd_elementwise_fallback():
    num_repeats = 10
    g = lambda x,y: x + y
    op = mx.nd.add_n
    for i in range(num_repeats):
        shape = [rand_shape_2d()] * 2
        check_sparse_nd_elemwise_binary(shape, ['default_storage'] * 2, op, g)
        check_sparse_nd_elemwise_binary(shape, ['default_storage', 'row_sparse'], op, g)
        check_sparse_nd_elemwise_binary(shape, ['row_sparse', 'row_sparse'], op, g)

def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(shape, stype):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.sparse_nd.zeros('row_sparse', shape)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros(shape, 'row_sparse')
    check_sparse_nd_zeros(shape, 'csr')


def test_sparse_nd_copy():
    def check_sparse_nd_copy(from_stype, to_stype):
        shape = rand_shape_2d()
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
    shape = rand_shape_2d()
    nd, (v, idx) = rand_sparse_ndarray(shape, storage_type)
    assert(nd._num_aux == 1)
    assert(nd.indices.dtype == np.int32)
    assert(nd.storage_type == 'row_sparse')
    assert_almost_equal(nd.indices.asnumpy(), idx)

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
    def check_sparse_nd_setitem(storage_type, shape, dst):
        x = mx.sparse_nd.zeros(storage_type, shape)
        x[:] = dst
        dst_nd = mx.nd.array(dst) if isinstance(dst, (np.ndarray, np.generic)) else dst
        assert same(x.asnumpy(), dst_nd.asnumpy())

    shape = rand_shape_2d()
    for stype in ['row_sparse', 'csr']:
        # ndarray assignment
        check_sparse_nd_setitem(stype, shape, rand_ndarray(shape, 'default_storage'))
        check_sparse_nd_setitem(stype, shape, rand_ndarray(shape, stype))
        # numpy assignment
        check_sparse_nd_setitem(stype, shape, np.ones(shape))

def test_sparse_nd_slice():
    def check_sparse_nd_csr_slice(shape):
        storage_type = 'csr'
        A, _ = rand_sparse_ndarray(shape, storage_type)
        A2 = A.asnumpy()
        start = rnd.randint(0, shape[0] - 1)
        end = rnd.randint(start + 1, shape[0])
        values = A[start:end].values
        indptr = A[start:end].indptr
        assert same(A[start:end].asnumpy(), A2[start:end])

    shape = (rnd.randint(2, 10), rnd.randint(1, 10))
    check_sparse_nd_csr_slice(shape)

def test_sparse_nd_equal():
    stype = 'csr'
    shape = rand_shape_2d()
    x = sparse_nd_zeros(shape, stype)
    y = sparse_nd_ones(shape, stype)
    z = x == y
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = 0 == x
    assert (z.asnumpy() == np.ones(shape)).all()

def test_sparse_nd_not_equal():
    stype = 'csr'
    shape = rand_shape_2d()
    x = sparse_nd_zeros(shape, stype)
    y = sparse_nd_ones(shape, stype)
    z = x != y
    assert (z.asnumpy() == np.ones(shape)).all()
    z = 0 != x
    assert (z.asnumpy() == np.zeros(shape)).all()

def test_sparse_nd_greater():
    stype = 'csr'
    shape = rand_shape_2d()
    x = sparse_nd_zeros(shape, stype)
    y = sparse_nd_ones(shape, stype)
    z = x > y
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = y > 0
    assert (z.asnumpy() == np.ones(shape)).all()
    z = 0 > y
    assert (z.asnumpy() == np.zeros(shape)).all()

def test_sparse_nd_greater_equal():
    stype = 'csr'
    shape = rand_shape_2d()
    x = sparse_nd_zeros(shape, stype)
    y = sparse_nd_ones(shape, stype)
    z = x >= y
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = y >= 0
    assert (z.asnumpy() == np.ones(shape)).all()
    z = 0 >= y
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = y >= 1
    assert (z.asnumpy() == np.ones(shape)).all()

def test_sparse_nd_lesser():
    stype = 'csr'
    shape = rand_shape_2d()
    x = sparse_nd_zeros(shape, stype)
    y = sparse_nd_ones(shape, stype)
    z = y < x
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = 0 < y
    assert (z.asnumpy() == np.ones(shape)).all()
    z = y < 0
    assert (z.asnumpy() == np.zeros(shape)).all()

def test_sparse_nd_lesser_equal():
    stype = 'csr'
    shape = rand_shape_2d()
    x = sparse_nd_zeros(shape, stype)
    y = sparse_nd_ones(shape, stype)
    z = y <= x
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = 0 <= y
    assert (z.asnumpy() == np.ones(shape)).all()
    z = y <= 0
    assert (z.asnumpy() == np.zeros(shape)).all()
    z = 1 <= y
    assert (z.asnumpy() == np.ones(shape)).all()


if __name__ == '__main__':
    test_sparse_nd_zeros()
    test_sparse_nd_elemwise_add()
    test_sparse_nd_elementwise_fallback()
    test_sparse_nd_copy()
    test_sparse_nd_setitem()
    test_sparse_nd_basic()
    test_sparse_nd_slice()
