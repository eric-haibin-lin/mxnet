from mxnet.test_utils import *

def check_binary_ops():
    def test_binary_op(name, lhs_stype, rhs_stype, shape,
                       forward_mxnet_call, forward_numpy_call, backward_numpy_call,
                       lhs_grad_stype, rhs_grad_stype):
        lhs = mx.symbol.Variable('lhs', storage_type=lhs_stype)
        rhs = mx.symbol.Variable('rhs', storage_type=rhs_stype)
        if lhs_grad_stype is not None:
            lhs._set_attr(grad_stype_hint=str(lhs_grad_stype))
        if rhs_grad_stype is not None:
            rhs._set_attr(grad_stype_hint=str(rhs_grad_stype))

        lhs_nd = rand_ndarray(shape, lhs_stype)
        #lhs_nd[0][0] = 2
        rhs_nd = rand_ndarray(shape, rhs_stype)
        #rhs_nd[0][0] = 3
        lhs_np = lhs_nd.asnumpy()
        rhs_np = rhs_nd.asnumpy()

        # print("lhs input:")
        # print(lhs_np)
        # print("rhs input:")
        # print(rhs_np)

        out_np = forward_numpy_call(lhs_np, rhs_np)
        test = forward_mxnet_call(lhs, rhs)
        out_grad = np.ones(shape)
        ingrad_lhs_np, ingrad_rhs_np = backward_numpy_call(out_grad, lhs_np, rhs_np)

        # print(ingrad_lhs_np)
        # print(ingrad_rhs_np)

        location = {'lhs': lhs_nd, 'rhs': rhs_nd}
        check_symbolic_forward(test, location, [out_np])
        check_numeric_gradient(test, location)
        check_symbolic_backward(test, location, [out_grad], [ingrad_lhs_np, ingrad_rhs_np])

    def test_binary_ops(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
        test_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                          lambda l, r: mx.sym.elemwise_add(l, r),
                          lambda l, r: l + r,
                          lambda outg, l, r: (outg, outg),
                          lhs_grad_stype, rhs_grad_stype)
        # test_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
        #                lambda l, r: mx.sym.elemwise_sub(l, r),
        #                lambda l, r: l - r,
        #                lambda outg, l, r: (outg, -outg),
        #                lhs_grad_stype, rhs_grad_stype)
        # test_binary_op("elemwise_mul", lhs_stype, rhs_stype, shape,
        #                lambda l, r: mx.sym.elemwise_mul(l, r),
        #                lambda l, r: l * r,
        #                lambda outg, l, r: (r, l),
        #                lhs_grad_stype, rhs_grad_stype)
        # test_binary_op("elemwise_div", lhs_stype, rhs_stype, shape,
        #                lambda l, r: mx.sym.elemwise_div(l, r),
        #                lambda l, r: l / r,
        #                lambda outg, l, r: (1/r, -l/(r*r)),
        #                lhs_grad_stype, rhs_grad_stype)

    #shape = (1, 1)
    shape = rand_shape_2d()
    test_binary_ops('default', 'default', shape)
    test_binary_ops('default', 'row_sparse', shape)
    test_binary_ops('row_sparse', 'default', shape)
    #test_binary_ops('row_sparse', 'row_sparse', shape, lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')


def check_elemwise_add_ex(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
    lhs = mx.symbol.Variable('lhs', storage_type=lhs_stype)
    rhs = mx.symbol.Variable('rhs', storage_type=rhs_stype)
    if lhs_grad_stype is not None:
        lhs._set_attr(grad_stype_hint=str(lhs_grad_stype))
    if rhs_grad_stype is not None:
        rhs._set_attr(grad_stype_hint=str(rhs_grad_stype))

    lhs_nd = rand_ndarray(shape, lhs_stype)
    rhs_nd = rand_ndarray(shape, rhs_stype)
    lhs_np = lhs_nd.asnumpy()
    rhs_np = rhs_nd.asnumpy()

    out_np = lhs_np + rhs_np
    test = mx.symbol.elemwise_add(lhs, rhs)
    location = {'lhs': lhs_nd, 'rhs': rhs_nd}
    check_symbolic_forward(test, location, [out_np])
    check_numeric_gradient(test, location)
    check_symbolic_backward(test, location, [out_np], [out_np, out_np])


def test_elemwise_add_ex():
    shape = rand_shape_2d()
    check_elemwise_add_ex('default', 'default', shape)
    check_elemwise_add_ex('default', 'row_sparse', shape)
    check_elemwise_add_ex('row_sparse', 'default', shape)
    check_elemwise_add_ex('row_sparse', 'row_sparse', shape,
                          lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse')


# TODO(haibin) randomize this test
def test_elemwise_add_ex_multiple_stages():
    # prep data
    shape = (4, 2)
    ds_np = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    sp_np1 = np.array([[5, 10], [0, 0], [0, 0], [0, 0]])
    sp_np2 = np.array([[0, 0], [5, 10], [0, 0], [0, 0]])

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
    plus = mx.symbol.elemwise_add(sp_data1, sp_data2, name='plus')
    # sparse + dense = dense
    test = mx.symbol.elemwise_add(plus, ds_data)
    check_symbolic_forward(test, {'sp_data1': sp_nd1, 'sp_data2': sp_nd2,
                                  'ds_data': ds_nd}, [sp_np1 + sp_np2 + ds_np])

    arr_grads = [mx.nd.zeros(shape) for i in range(3)]
    exec_test = test.bind(default_context(), args={'sp_data1': sp_nd1, 'sp_data2': sp_nd2,
                                                   'ds_data': ds_nd}, args_grad=arr_grads)
    exec_test.forward(is_train=True)
    assert_almost_equal(exec_test.outputs[0].asnumpy(), sp_np1 + sp_np2 + ds_np)
    exec_test.backward(out_grads=exec_test.outputs)
    assert_almost_equal(arr_grads[0].asnumpy(), arr_grads[1].asnumpy())

# TODO(haibin) also add test for backward pass.
def test_cast_storage_ex():
    def test_rsp_to_dns(shape):
        rsp, (data, row_idx) = rand_sparse_ndarray(shape, 'row_sparse')
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

    def test_csr_to_dns(shape):
        csr, (indptr, indices, values) = rand_sparse_ndarray(shape, 'csr')
        mx_dns = csr.to_dense()
        np_dns = sp.csr_matrix((values, indices, indptr), shape).todense()
        assert_almost_equal(mx_dns.asnumpy(), np_dns)

    def test_dns_to_csr(dns_in):
        dns_in = np.array(dns_in)
        csr_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), storage_type='csr')
        ret = mx.nd.cast_storage(csr_out, storage_type='default')
        assert same(ret.asnumpy(), dns_in)

    shape = rand_shape_2d()
    test_rsp_to_dns(shape)
    test_dns_to_rsp(shape)
    test_csr_to_dns((4, 4))
    test_dns_to_csr([[0, 1, 0], [0, 2, 0], [3, 0, 0], [0, 0, 4], [5, 6, 0], [0, 0, 7]])

def test_sparse_dot():
    def test_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs):
        lhs_dns = rand_ndarray(lhs_shape, 'default')
        lhs_nd = mx.nd.cast_storage(lhs_dns, storage_type='csr')
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=1)
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.to_dense()
        out = mx.nd.dot(lhs_nd, rhs_dns, transpose_a=trans_lhs)
        assert out.storage_type == 'default'
        out_expected = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=trans_lhs)
        out_np = out_expected.asnumpy()
        backward_trans = not trans_lhs
        rhs_backward_grad = mx.nd.dot(lhs_dns, out_expected, transpose_a=backward_trans).asnumpy()
        assert_almost_equal(out.asnumpy(), out_np, rtol=1e-4, atol=1e-5)

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', storage_type='csr')
        rhs = mx.symbol.Variable('rhs', storage_type=rhs_stype)
        test = mx.symbol.dot(lhs, rhs, transpose_a=trans_lhs)
        location = {'lhs': lhs_nd, 'rhs': rhs_nd}
        expected = {'rhs': rhs_backward_grad}
        check_symbolic_forward(test, location, [out_np], rtol=1e-3, atol=1e-4)
        # test symbolic backward
        check_symbolic_backward(test, location, [out_np], expected,
                                grad_req={'lhs': 'null', 'rhs': 'write'},
                                rtol=1e-3, atol=1e-4)

    lhs_shape = rand_shape_2d()
    test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'default', False)
    test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'default', True)
    test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False)
    test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True)

def test_sparse_mathematical_core():
    def mathematical_core(name, stype,
                          forward_mxnet_call, forward_numpy_call, backward_numpy_call,
                          grad_stype=None, data_init=5., grad_init=2.):
        data = mx.symbol.Variable('data', storage_type=stype)
        if grad_stype is not None:
            data._set_attr(grad_stype_hint=str(grad_stype))
        shape = (3, 4)
        data_tmp = np.ones(shape)
        data_tmp[:] = data_init
        arr_data = mx.nd.array(data_tmp)

        if stype != 'default':
            arr_data = mx.nd.cast_storage(arr_data, storage_type=stype)

        arr_grad = mx.nd.empty(shape)
        arr_grad[:] = 3
        if grad_stype is not None and grad_stype != 'default':
            arr_grad = mx.nd.cast_storage(arr_grad, storage_type=grad_stype)

        test = forward_mxnet_call(data)
        exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
        exe_test.forward(is_train=True)
        out = exe_test.outputs[0].asnumpy()
        npout = forward_numpy_call(data_tmp)
        assert_almost_equal(out, npout)

        out_grad = mx.nd.empty(shape)
        out_grad[:] = grad_init
        npout_grad = out_grad.asnumpy()
        temp = backward_numpy_call(data_tmp)
        npout_grad = npout_grad * temp

        if grad_stype is not None and grad_stype != 'default':
            out_grad = mx.nd.cast_storage(out_grad, storage_type=grad_stype)

        exe_test.backward(out_grad)
        arr_grad = arr_grad.asnumpy()

        # print(name)
        # print(arr_grad)
        # print(npout_grad)

        assert_almost_equal(arr_grad, npout_grad)

    def check_mathematical_core(stype, grad_stype=None):
        # sqrt
        mathematical_core("sqrt", stype,
                          lambda x: mx.sym.sqrt(x),
                          lambda x: np.sqrt(x),
                          lambda x: 1.0/(2.0 * np.sqrt(x)),
                          grad_stype)
        # rsqrt
        mathematical_core("rsqrt", stype,
                          lambda x: mx.sym.rsqrt(x),
                          lambda x: 1 / np.sqrt(x),
                          lambda x: -(1.0 / (2.0 * x * np.sqrt(x))),
                          grad_stype)

    check_mathematical_core('default')
    #check_mathematical_core('csr', 'csr')
    check_mathematical_core('row_sparse')
    check_mathematical_core('row_sparse', 'default')
    check_mathematical_core('row_sparse', 'row_sparse')

def test_sparse_embedding():
    in_dim = 10
    out_dim = 4
    batch = 24

    data = mx.sym.Variable("data", storage_type='csr')
    embed = mx.sym.SparseEmbedding(data=data, input_dim=in_dim, output_dim=out_dim, name="embed")
    exe_test = embed.simple_bind(default_context(), grad_req={'data': 'null', 'embed_weight': 'write'},
                                 data=(batch, in_dim))

    arg_map = dict(zip(embed.list_arguments(), exe_test.arg_arrays))
    grad_map = dict(zip(embed.list_arguments(), exe_test.grad_arrays))
    np_data = np.random.randint(low=0, high=in_dim, size=batch)
    np_weight = np.random.uniform(-0.01, 0.01, arg_map["embed_weight"].shape)
    np_onehot = np.zeros((batch, in_dim))
    np_onehot[np.arange(batch), np_data] = 1.0
    nd_onehot = mx.nd.array(np_onehot).to_csr()
    # forward
    arg_map["data"][:] = nd_onehot
    arg_map["embed_weight"][:] = np_weight
    exe_test.forward(is_train=True)
    assert_almost_equal(exe_test.outputs[0].asnumpy(), np.dot(np_onehot, np_weight))
    # backward
    np_grad = np.random.uniform(-1, 1, exe_test.outputs[0].shape)
    grad = mx.nd.zeros(np_grad.shape)
    grad[:] = np_grad
    exe_test.backward([grad])
    assert_almost_equal(grad_map["embed_weight"].asnumpy(), np.dot(np_onehot.T, np_grad), atol=1e-5)


def test_sparse_slice():
    def check_csr_slice(shape, slice_input):
        storage_type = 'csr'
        A, _ = rand_sparse_ndarray(shape, storage_type)
        B = A._slice(1, shape[0] - 1) if slice_input else A
        np = B.asnumpy()
        begin = rnd.randint(0, B.shape[0] - 1)
        end = rnd.randint(begin + 1, B.shape[0])
        nd_slice = mx.nd.crop(B, begin=begin, end=end)
        assert same(nd_slice.asnumpy(), np[begin:end]), (nd_slice.asnumpy(), np[begin:end])

    shape = (rnd.randint(7, 15), rnd.randint(1, 10))
    check_csr_slice(shape, True)
    check_csr_slice(shape, False)


def test_sparse_retain():
    for _ in range(10):
        shape = rand_shape_2d()
        num_rows = shape[0]
        rsp, _ = rand_sparse_ndarray(shape=shape, storage_type='row_sparse', density=0.5)
        length = np.random.randint(1, num_rows + 1)
        idx = random_sample(list(range(0, num_rows)), length)
        idx.sort()
        dns = rsp.asnumpy()
        tensor_retained_expected = np.zeros(shape)
        for i in idx:
            tensor_retained_expected[i][:] = dns[i]
        indices = mx.nd.array(idx)
        rsp_retained = mx.nd.sparse_retain(rsp, indices=indices)
        assert same(tensor_retained_expected, rsp_retained.asnumpy())

        # check numeric gradient
        data = mx.symbol.Variable('data')
        idx = mx.symbol.Variable('indices')
        sym = mx.sym.sparse_retain(data=data, indices=idx)
        check_numeric_gradient(sym, [rsp, indices], grad_nodes=['data'], grad_stype_dict={'data': 'row_sparse'})

if __name__ == '__main__':
    #import nose
    #nose.runmodule()
    check_binary_ops()
    #test_sparse_mathematical_core()
