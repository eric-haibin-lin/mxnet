# pylint: skip-file
import numpy as np
import mxnet as mx
from numpy.testing import assert_allclose
from mxnet.test_utils import *

def check_elemwise_add_ex(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
    lhs = mx.symbol.Variable('lhs', storage_type = lhs_stype)
    rhs = mx.symbol.Variable('rhs', storage_type = rhs_stype)
    if lhs_grad_stype is not None:
        lhs._set_attr(grad_stype_hint=str(lhs_grad_stype))
    if rhs_grad_stype is not None:
        rhs._set_attr(grad_stype_hint=str(rhs_grad_stype))

    lhs_nd = rand_ndarray(shape, lhs_stype, sparsity=0)
    rhs_nd = rand_ndarray(shape, rhs_stype, sparsity=0)
    lhs_np = lhs_nd.asnumpy()
    rhs_np = rhs_nd.asnumpy()

    out_np = lhs_np + rhs_np
    test = mx.symbol.elemwise_add(lhs, rhs)
    location = {'lhs':lhs_nd, 'rhs':rhs_nd}
    check_symbolic_forward(test, location, [out_np])
    check_numeric_gradient(test, location)
    check_symbolic_backward(test, location, [out_np], [out_np, out_np])

def test_elemwise_add_ex():
    shape = (rnd.randint(1, 10),rnd.randint(1, 10))
    check_elemwise_add_ex('default', 'default', shape)
    check_elemwise_add_ex('default', 'row_sparse', shape)
    check_elemwise_add_ex('row_sparse', 'default', shape)
    check_elemwise_add_ex('row_sparse', 'row_sparse', shape,
                       lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')

# TODO(haibin) randomize this test
def test_elemwise_add_ex_multiple_stages():
    # prep data
    shape = (4, 2)
    ds_np = np.array([[1,2],[3,4],[5,6],[7,8]])
    sp_np1 = np.array([[5,10],[0,0],[0,0],[0,0]])
    sp_np2 = np.array([[0,0],[5,10],[0,0],[0,0]])

    val1 = mx.nd.array([[5, 10]]);
    val2 = mx.nd.array([[5, 10]]);
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

# TODO(haibin) also add test for backward pass
def test_cast_storage_ex():
    def test_rsp_to_dns(shape):
        rsp, data, row_idx = rand_sparse_ndarray(shape, 'row_sparse', allow_zeros = True)
        dns_out = mx.nd.cast_storage(rsp, storage_type='default')
        dns_expected = np.zeros(shape, dtype=default_dtype())
        if row_idx is not None:
            for k, v in enumerate(row_idx):
                dns_expected[v, :] = data[k]
        assert same(dns_out.asnumpy(), dns_expected)

    def test_dns_to_rsp(shape):
        dns_in = rand_ndarray(shape, 'default')
        rsp_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), storage_type='row_sparse')
        ret = mx.nd.cast_storage(rsp_out, storage_type='default')
        assert same(ret.asnumpy(), dns_in.asnumpy())

    def test_csr_to_dns(data, indptr, col_idx, shape):
        indptr = np.array(indptr, dtype=np.int32)
        col_idx = np.array(col_idx, dtype=np.int32)
        csr = mx.sparse_nd.csr(data, indptr, col_idx, shape=shape)
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

    shape = (rnd.randint(1, 10),rnd.randint(1, 10))
    test_rsp_to_dns(shape)
    test_dns_to_rsp(shape)
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
    def test_dot_csr_dns_rsp(csr_shape, dns_shape, trans_csr):
        dns1 = rand_ndarray(csr_shape, 'default')
        dns2 = rand_ndarray(dns_shape, 'default')
        csr = mx.nd.cast_storage(dns1, storage_type='csr')
        rsp_out = mx.nd.dot(csr, dns2, transpose_a=trans_csr)
        rsp_expected = mx.nd.dot(dns1, dns2, transpose_a=trans_csr)
        out_np = rsp_expected.asnumpy()
        backward_trans = not trans_csr
        rhs_backward_grad = mx.nd.dot(dns1, rsp_expected, transpose_a=backward_trans).asnumpy()
        # TODO(junwu): may need to compare rsp_out and rsp_expected in rsp format
        # instead of converting them to the dense format
        assert same(rsp_out.asnumpy(), out_np)

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', storage_type='csr')
        rhs = mx.symbol.Variable('rhs', storage_type='default')
        rhs._set_attr(grad_stype_hint=str('row_sparse'))
        # TODO(haibin) since backward op is not fully implemented, here we add a dense zero ndarray
        # so that the output gradient is dense.
        zeros = mx.symbol.Variable('zero', storage_type='default')

        sym_dot = mx.symbol.dot(lhs, rhs, transpose_a=trans_csr)
        test = mx.symbol.elemwise_add(sym_dot, zeros)
        location = {'lhs':csr, 'rhs':dns2, 'zero':mx.nd.zeros(rsp_expected.shape)}
        expected = {'rhs':rhs_backward_grad, 'zero':out_np}
        # dot(lhs, rhs) + zeros
        check_symbolic_forward(test, location, [rsp_expected.asnumpy()])
        check_symbolic_backward(test, location, [out_np], expected,
                                grad_req={'lhs': 'null', 'rhs': 'write', 'zero' : 'write'})

    lhs_shape = (rnd.randint(1, 10),rnd.randint(1, 10))
    test_dot_csr_dns_rsp(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), trans_csr=False)
    test_dot_csr_dns_rsp(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), trans_csr=True)

def test_sparse_embedding():
    in_dim = 10
    out_dim = 4
    batch = 24

    data = mx.sym.Variable("data", dtype=np.int32)
    embed = mx.sym.SparseEmbedding(data=data, input_dim=in_dim, output_dim=out_dim, name="embed")
    # TODO(haibin) test again when simple_bind cpp api is ready
    exe_test = embed.simple_bind(default_context(), grad_req={'data': 'null', 'embed_weight': 'write'},
                                 data=(batch,))
    arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
    grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
    np_data = np.random.randint(low=0, high=in_dim, size=batch)
    np_weight = np.random.uniform(-0.01, 0.01, arg_map["embed_weight"].shape)
    np_onehot = np.zeros((batch, in_dim))
    np_onehot[np.arange(batch), np_data] = 1.0
    # forward
    arg_map["data"][:] = np_data
    arg_map["embed_weight"][:] = np_weight
    exe_test.forward(is_train=True)
    assert_almost_equal(exe_test.outputs[0].asnumpy(), np.dot(np_onehot, np_weight))
    # backward
    np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
    grad = mx.nd.zeros(np_grad.shape)
    grad[:] = np_grad
    exe_test.backward([grad])
    assert_almost_equal(grad_map["embed_weight"].asnumpy(), np.dot(np_onehot.T, np_grad))

if __name__ == '__main__':
    test_elemwise_add_ex()
    test_elemwise_add_ex_multiple_stages()
    test_cast_storage_ex()
    test_sparse_dot()
    test_sparse_embedding()
