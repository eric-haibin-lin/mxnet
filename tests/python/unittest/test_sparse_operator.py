from mxnet.test_utils import *
import random

def get_result_type(call, dflt_stype):
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        result = do_normalize(call(zero))
        if not almost_equal(result, zero, equal_nan=True):
            expected_result_type = 'default'
        else:
            if dflt_stype is not None:
                expected_result_type = dflt_stype;
            else:
                expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type


def get_result_type_2(call, dflt_stype):
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        need_default = False
        for outer in [zero, np.ones(zero.shape)]:
            for inner in [zero, np.ones(zero.shape)]:
                result = do_normalize(call(outer, inner))
                if not almost_equal(result, zero, equal_nan=True):
                    need_default = True
                    break
            if need_default is True:
                break

        if not need_default and dflt_stype is not None:
            expected_result_type = dflt_stype
        else:
            expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type

def get_result_type_3(call, dflt_stype):
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        need_default = False
        for moon in [zero]:
            for outer in [zero]:
                for inner in [zero]:
                    res_1, res_2 = call(moon, outer, inner)
                    result = do_normalize(res_1)
                    if not almost_equal(result, zero, equal_nan=True):
                        need_default = True
                        break
                    result = do_normalize(res_2)
                    if not almost_equal(result, zero, equal_nan=True):
                        need_default = True
                        break
                if need_default is True:
                    break
            if need_default is True:
                break

        if not need_default and dflt_stype is not None:
            expected_result_type = dflt_stype
        else:
            expected_result_type = 'default'
    else:
        expected_result_type = 'default'

    return expected_result_type

def get_fw_bw_result_types(forward_numpy_call,  fwd_res_dflt,
                           backward_numpy_call, bwd_res_dflt):

    return (get_result_type(forward_numpy_call,  fwd_res_dflt),
            get_result_type(backward_numpy_call, bwd_res_dflt))


def get_fw_bw_result_types_2(forward_numpy_call,  fwd_res_dflt,
                             backward_numpy_call, bwd_res_dflt):
    return (get_result_type(forward_numpy_call,  fwd_res_dflt),
            get_result_type_2(backward_numpy_call, bwd_res_dflt))


def gen_rsp_random_indices(shape, density=.5, force_indices=None):
    assert density >= 0 and density <= 1
    indices = set(force_indices) if force_indices is not None else set()
    if not np.isclose(density, .0, rtol=1.e-3, atol=1.e-3, equal_nan=True) and len(shape) > 0:
        row_count = shape[0]
        for i in range(row_count):
            r = random.uniform(0, 1)
            if r <= density:
                indices.add(i)
    return list(indices)


def rand_bool():
    return True if random.uniform(0, 1) <= 0.5 else False


def rand_choice(a, b):
    return a if random.uniform(0, 1) <= 0.5 else b


def check_elemwise_binary_ops():
    def test_elemwise_binary_op(name, lhs_stype, rhs_stype, shape,
                                forward_mxnet_call, forward_numpy_call, backward_numpy_call,
                                lhs_grad_stype, rhs_grad_stype,
                                expected_result_storage_type=None,
                                modifier_func=None,
                                density=.5, force_overlap=False,
                                ograd_density=0.0,
                                skip_gradient_check=False,
                                verbose=False):
        if verbose is True:
            print("test_elemwise_binary_op:", name)

        if lhs_grad_stype is None:
            lhs_grad_stype = lhs_stype
        if rhs_grad_stype is None:
            rhs_grad_stype = rhs_stype

        lhs_stype = get_result_type_3(backward_numpy_call, lhs_grad_stype)
        rhs_stype = get_result_type_3(backward_numpy_call, rhs_grad_stype)

        # Output type should be same as lvalue type, unless otherwise specified
        if expected_result_storage_type is None:
            if lhs_stype == 'default' or rhs_stype == 'default':
                expected_result_storage_type = 'default'
            else:
                expected_result_storage_type = lhs_stype

        lhs = mx.symbol.Variable('lhs', storage_type=lhs_stype)
        rhs = mx.symbol.Variable('rhs', storage_type=rhs_stype)
        if lhs_grad_stype != 'default':
            lhs._set_attr(input_grad_stype_hint=lhs_grad_stype)
        if rhs_grad_stype != 'default':
            rhs._set_attr(input_grad_stype_hint=rhs_grad_stype)

        if lhs_stype == 'default':
            lhs_nd = rand_ndarray(shape, 'default')
            lhs_nd = mx.nd.array(assign_each(lhs_nd.asnumpy(), modifier_func))
        else:
            lhs_nd = create_sparse_array_zd(
                shape, lhs_stype,
                modifier_func=modifier_func,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=density,
                    force_indices=[(shape[0]/2)] if force_overlap is True else None
                ))

        if rhs_stype == 'default':
            rhs_nd = rand_ndarray(shape, 'default')
            rhs_nd = mx.nd.array(assign_each(rhs_nd.asnumpy(), modifier_func))
        else:
            rhs_nd = create_sparse_array_zd(
                shape, rhs_stype,
                modifier_func=modifier_func,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=density,
                    force_indices=[(shape[0]/2)] if force_overlap is True else None
                    ))

        lhs_np = lhs_nd.asnumpy()
        rhs_np = rhs_nd.asnumpy()

        if verbose is True:
            print("lhs input:")
            print(lhs_np)
            print("rhs input:")
            print(rhs_np)

        out_np = forward_numpy_call(lhs_np, rhs_np)

        if verbose is True:
            print("out_np", out_np)

        test = forward_mxnet_call(lhs, rhs)

        location = {'lhs': lhs_nd, 'rhs': rhs_nd}

        outputs = check_symbolic_forward(test, location, [out_np], equal_nan=True)
        assert len(outputs) == 1
        assert outputs[0].storage_type == expected_result_storage_type

        if verbose is True:
            print ("lhs_nd: ", lhs_nd.storage_type)
            print ("rhs_nd: ", rhs_nd.storage_type)
            print ("forward output: ", outputs[0].storage_type)

        if outputs[0].storage_type != 'default':
            out_grad = create_sparse_array_zd(
                shape, outputs[0].storage_type,
                data_init=1,
                modifier_func=lambda x: 2,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=ograd_density,
                    force_indices=[(shape[0]/2)] if force_overlap is True else None
                ))
        else:
            out_grad = mx.nd.array(np.ones(shape))

        out_grad_np = out_grad.asnumpy()

        if verbose is True:
            print("out_grad_np", out_grad_np)

        ingrad_lhs_np, ingrad_rhs_np = backward_numpy_call(out_grad_np, lhs_np, rhs_np)

        if verbose is True:
            print("out_grad", out_grad.asnumpy())
            print("ingrad_lhs_np", ingrad_lhs_np)
            print("ingrad_rhs_np", ingrad_rhs_np)

        igrads_result = check_symbolic_backward(test, location, [out_grad], [ingrad_lhs_np, ingrad_rhs_np],
                                                equal_nan=True)

        if verbose is True:
            print("ingrad_lhs", igrads_result['lhs'].asnumpy())
            print("ingrad_rhs", igrads_result['rhs'].asnumpy())

        assert len(igrads_result) == 2

        if lhs_grad_stype is not None:
            assert igrads_result['lhs'].storage_type == lhs_grad_stype
        if rhs_grad_stype is not None:
            assert igrads_result['rhs'].storage_type == rhs_grad_stype

        if skip_gradient_check is not True:
            check_numeric_gradient(test, location)

    def check_all(l, r, check_function):
        assert l.shape == r.shape

        it_l = np.nditer(l, flags=['f_index'])
        it_r = np.nditer(r, flags=['f_index'])

        output = np.zeros(l.shape)
        it_out = np.nditer(output, flags=['f_index'], op_flags=['writeonly'])

        while not it_l.finished:
            val_l = it_l[0]
            val_r = it_r[0]
            if check_function(val_l, val_r):
                it_out[0] = 1
            it_l.iternext()
            it_r.iternext()
            it_out.iternext()

        return output

    def gt(l, r):
        return check_all(l, r, lambda a, b: a > b)

    def ge(l, r):
        return check_all(l, r, lambda a, b: a >= b)

    def lt(l, r):
        return check_all(l, r, lambda a, b: a < b)

    def le(l, r):
        return check_all(l, r, lambda a, b: a <= b)

    def least_sparse(lstype, rstype):
        if lstype == 'default' or rstype == 'default':
            return 'default'
        else:
            return lstype

    def test_elemwise_binary_ops(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None,
                                 density=.5, force_overlap=False, ograd_density=0.0):
        test_elemwise_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_add(l, r),
                                lambda l, r: l + r,
                                lambda outg, l, r: (outg, outg),
                                lhs_grad_stype, rhs_grad_stype,
                                ograd_density=ograd_density,
                                force_overlap=force_overlap, density=density)

        test_elemwise_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_sub(l, r),
                                lambda l, r: l - r,
                                lambda outg, l, r: (outg, -outg),
                                lhs_grad_stype, rhs_grad_stype,
                                ograd_density=ograd_density,
                                force_overlap=force_overlap, density=density)


        test_elemwise_binary_op("elemwise_mul", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_mul(l, r),
                                lambda l, r: l * r,
                                lambda outg, l, r: (outg * r, outg * l),
                                least_sparse(lhs_stype, rhs_stype),
                                least_sparse(lhs_stype, rhs_stype),
                                ograd_density=ograd_density,
                                force_overlap=force_overlap, density=density)

        test_elemwise_binary_op("elemwise_div", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_div(l, r),
                                lambda l, r: l / r,
                                lambda outg, l, r: (outg * (1/r), outg * (-l/(r*r))),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_overlap=force_overlap, density=density,
                                ograd_density=ograd_density)

        test_elemwise_binary_op("maximum", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.maximum(l, r),
                                lambda l, r: np.maximum(l, r),
                                lambda outg, l, r: (outg * ge(l, r), outg * lt(l, r)),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_overlap=False, density=density,
                                skip_gradient_check=True,
                                ograd_density=ograd_density,
                                verbose=False)

        test_elemwise_binary_op("minimum", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.minimum(l, r),
                                lambda l, r: np.minimum(l, r),
                                lambda outg, l, r: (outg * le(l, r), outg * gt(l, r)),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_overlap=force_overlap, density=density,
                                ograd_density=ograd_density,
                                skip_gradient_check=True)

        test_elemwise_binary_op("hypot", lhs_stype, rhs_stype, shape,
                                            lambda l, r: mx.sym.hypot(l, r),
                                            lambda l, r: np.hypot(l, r),
                                            lambda outg, l, r: (
                                                outg * assign_each2(l, r, lambda a, b: a/np.sqrt(a * a + b * b)),
                                                outg * assign_each2(l, r, lambda a, b: b/np.sqrt(a * a + b * b))
                                            ),
                                lhs_grad_stype, rhs_grad_stype,
                                force_overlap=force_overlap, density=density,
                                #modifier_func=lambda a: 0,
                                ograd_density=ograd_density,
                                skip_gradient_check=True,
                                verbose=False)

        test_elemwise_binary_op("power", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.pow(l, r),
                                lambda l, r: np.power(l, r),
                                lambda outg, l, r: (
                                    outg * assign_each2(l, r, lambda a, b: np.power(a, b - 1) * b),
                                    outg * assign_each2(l, r, lambda a, b: np.power(a, b) * np.log(a))
                                ),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                #modifier_func=lambda a: 0,
                                force_overlap=force_overlap, density=density,
                                ograd_density=ograd_density,
                                skip_gradient_check=True,
                                verbose=False)

    # Run basic tests
    for ii in range(2):
        for density in [0.0, random.uniform(0, 1), 1.0]:
            for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
                for force_overlap in [False, True]:
                    shape = rand_shape_2d()
                    #shape = (1,1)
                    test_elemwise_binary_ops('default', 'default', shape, density=density,
                                             force_overlap=force_overlap, ograd_density=ograd_density)
                    test_elemwise_binary_ops('default', 'row_sparse', shape, density=density,
                                             force_overlap=force_overlap, ograd_density=ograd_density)
                    test_elemwise_binary_ops('row_sparse', 'default', shape, density=density,
                                             force_overlap=force_overlap, ograd_density=ograd_density)
                    test_elemwise_binary_ops('row_sparse', 'row_sparse', shape,
                                             lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse',
                                             density=density, force_overlap=force_overlap,
                                             ograd_density=ograd_density)


# TODO(haibin) randomize this test
def check_elemwise_add_ex_multiple_stages():
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
def check_cast_storage_ex():
    def check_rsp_to_dns(shape):
        rsp, (data, row_idx) = rand_sparse_ndarray(shape, 'row_sparse')
        dns_out = mx.nd.cast_storage(rsp, storage_type='default')
        dns_expected = np.zeros(shape, dtype=default_dtype())
        if row_idx is not None:
            for k, v in enumerate(row_idx):
                dns_expected[v, :] = data[k]
        assert same(dns_out.asnumpy(), dns_expected)

    def check_dns_to_rsp(shape):
        dns_in = rand_ndarray(shape, 'default')
        rsp_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), storage_type='row_sparse')
        ret = mx.nd.cast_storage(rsp_out, storage_type='default')
        assert same(ret.asnumpy(), dns_in.asnumpy())

    def check_csr_to_dns(shape):
        csr, (indptr, indices, values) = rand_sparse_ndarray(shape, 'csr')
        mx_dns = csr.to_dense()
        np_dns = sp.csr_matrix((values, indices, indptr), shape).todense()
        assert_almost_equal(mx_dns.asnumpy(), np_dns)

    def check_dns_to_csr(dns_in):
        dns_in = np.array(dns_in)
        csr_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), storage_type='csr')
        ret = mx.nd.cast_storage(csr_out, storage_type='default')
        assert same(ret.asnumpy(), dns_in)

    shape = rand_shape_2d()
    check_rsp_to_dns(shape)
    check_dns_to_rsp(shape)
    check_csr_to_dns((4, 4))
    check_dns_to_csr([[0, 1, 0], [0, 2, 0], [3, 0, 0], [0, 0, 4], [5, 6, 0], [0, 0, 7]])


def check_sparse_dot():
    def check_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs):
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
    check_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'default', False)
    check_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'default', True)
    check_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False)
    check_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True)


def as_dense(arr):
    if arr.storage_type != 'default':
        return mx.nd.cast_storage(arr, storage_type='default')
    else:
        return arr;

# Make sure that 0's look like 0's when we do a comparison
def do_normalize(l):
    it_l = np.nditer(l, flags=['f_index'])

    output = np.zeros(l.shape)
    it_out = np.nditer(output, flags=['f_index'], op_flags=['writeonly'])

    while not it_l.finished:
        val_l = it_l[0]
        if np.isclose(val_l, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
            val_l = 0
        it_out[0] = val_l
        it_l.iternext()
        it_out.iternext()

    return output

def check_sparse_mathematical_core():
    # Unary operator check
    def mathematical_core(name, stype,
                          forward_mxnet_call, forward_numpy_call, backward_numpy_call=None,
                          data_init=9., grad_init=2., output_grad_stype=None, input_grad_stype=None,
                          force_overlap=False, density=.5, ograd_density=.5):
        #print("TESTING: " + name)
        data = mx.symbol.Variable('data', storage_type=stype)

        if input_grad_stype is None:
            input_grad_stype = stype;

        expected_result_type, expected_grad_result_type = \
            get_fw_bw_result_types(forward_numpy_call, stype,
                                   backward_numpy_call, output_grad_stype)

        if input_grad_stype != 'default':
            data._set_attr(input_grad_stype_hint=expected_grad_result_type)

        # if expected_result_type != stype and expected_result_type == 'default':
        #     print(name + " >>> Dense forward result: " + expected_result_type)
        #
        # if backward_numpy_call is not None and expected_grad_result_type == 'default' and input_grad_stype != 'default':
        #     print(name + " <<< Dense backward result: " + expected_grad_result_type)

        shape = rand_shape_2d()
        #shape = (9, 1)
        #print("Shape: ", shape, "density: ", density, "force_overlap", force_overlap)

        if stype == 'default':
            data_tmp = np.ones(shape)
            data_tmp[:] = data_init
            arr_data = mx.nd.array(data_tmp)
        else:
            arr_data = create_sparse_array_zd(
                shape, stype,
                data_init=data_init,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=density,
                    #density=0.0,
                    force_indices=[(shape[0]/2)] if force_overlap is True else None
                    #force_indices=(1, 4, 5, 6, 8)
                )
            )
            data_tmp = arr_data.asnumpy()

        #print("input", data_tmp)

        if backward_numpy_call is None:
            arr_grad = None
        elif expected_grad_result_type == 'default':
            arr_grad = mx.nd.ones(shape)
        else:
            arr_grad = create_sparse_array_zd(
                shape, expected_grad_result_type, data_init=1,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=density,
                    #density=0.0,
                    force_indices=[(shape[0]/2)] if force_overlap is True else None
                    #force_indices=[(1, 2)]
                )
            )

        test = forward_mxnet_call(data)

        if arr_grad is not None:
            exe_test = test.bind(default_context(), args=[arr_data], args_grad=[arr_grad])
        else:
            exe_test = test.bind(default_context(), args=[arr_data])
        exe_test.forward(is_train=True)
        assert exe_test.outputs[0].storage_type == expected_result_type
        out = exe_test.outputs[0].asnumpy()
        npout = forward_numpy_call(data_tmp)

        #print("out", out)
        #print("npout", npout)

        assert_almost_equal(out, npout, equal_nan=True)

        if backward_numpy_call is not None:
            if input_grad_stype == 'default' or input_grad_stype is None:
                out_grad = mx.nd.empty(shape)
                out_grad[:] = grad_init
            else:
                out_grad = create_sparse_array_zd(
                    shape, input_grad_stype, data_init=grad_init,
                    rsp_indices=gen_rsp_random_indices(
                        shape,
                        density=ograd_density,
                        force_indices=[(shape[0]/2)] if force_overlap is True else None))

            npout_grad = out_grad.asnumpy()

            #print(npout_grad)

            temp = backward_numpy_call(data_tmp)
            #temp = backward_numpy_call(npout_grad)
            input_grad = npout_grad * temp

            #print(arr_grad.asnumpy())
            exe_test.backward(out_grad)
            #print(arr_grad.asnumpy())

            assert arr_grad.storage_type == expected_grad_result_type

            arr_grad = arr_grad.asnumpy()

            #print(name)
            #print("arr_grad", arr_grad)
            #print("npout_grad", npout_grad)
            #print("input_grad", input_grad)

            assert_almost_equal(arr_grad, input_grad, equal_nan=True)

            #print("Done")

    def util_sign(a):
        if np.isclose(a, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
            return 0
        elif np.isclose(a, 0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
            return 0
        elif a < 0.0:
            return -1
        else:  # a > 0.0:
            return 1

    # Check many basic unary operators
    def test_mathematical_core(stype, output_grad_stype=None, force_overlap=False,
                               density=.5, ograd_density=.5):
        # sqrt
        mathematical_core("sqrt", stype,
                          lambda x: mx.sym.sqrt(x),
                          lambda x: np.sqrt(x),
                          lambda x: 1.0/(2.0 * np.sqrt(x)),
                          output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # rsqrt
        mathematical_core("rsqrt", stype,
                          lambda x: mx.sym.rsqrt(x),
                          lambda x: 1 / np.sqrt(x),
                          lambda x: -(1.0 / (2.0 * x * np.sqrt(x))),
                          output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # square
        mathematical_core("square", stype,
                          lambda x: mx.sym.square(x),
                          lambda x: np.square(x),
                          lambda x: 2 * x,
                          output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                          density=density, ograd_density=ograd_density)

        # tan
        mathematical_core("tan", stype, lambda x: mx.sym.tan(x), lambda x: np.tan(x), lambda x: np.tan(x) ** 2 + 1,
                          output_grad_stype=output_grad_stype, density=density, ograd_density=ograd_density)

        # abs
        mathematical_core("abs", stype,
                          lambda x: mx.sym.abs(x),
                          lambda x: np.abs(x),
                          lambda x: assign_each(x, function=util_sign),
                          output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # negative
        mathematical_core("negative", stype, lambda x: mx.sym.negative(x), lambda x: np.negative(x), force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # floor
        mathematical_core("floor", stype, lambda x: mx.sym.floor(x), lambda x: np.floor(x), force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # ceil
        mathematical_core("ceil", stype, lambda x: mx.sym.ceil(x), lambda x: np.ceil(x), force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # sign
        mathematical_core("sign", stype, lambda x: mx.sym.sign(x), lambda x: np.sign(x), lambda x: np.zeros(x.shape), output_grad_stype=output_grad_stype,
          force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # cos
        mathematical_core("cos", stype, lambda x: mx.sym.cos(x), lambda x: np.cos(x), lambda x: -np.sin(x), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # sin
        mathematical_core("sin", stype, lambda x: mx.sym.sin(x), lambda x: np.sin(x), lambda x: np.cos(x), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # arcsin
        mathematical_core("arcsin", stype,
                          lambda x: mx.sym.arcsin(x),
                          lambda x: np.arcsin(x),
                          lambda x: 1. / (1. - x ** 2) ** (1. / 2.),
                          0.5, 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # arccos
        mathematical_core("arccos", stype, lambda x: mx.sym.arccos(x), lambda x: np.arccos(x),
                          lambda x: -1. / (1. - x ** 2.) ** (1. / 2.), 0.5, 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # arctan
        mathematical_core("arctan", stype, lambda x: mx.sym.arctan(x), lambda x: np.arctan(x),
                          lambda x: 1. / (x ** 2. + 1.), 0.5, 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # degrees
        mathematical_core("degrees", stype,
                          lambda x: mx.sym.degrees(x),
                          lambda x: np.degrees(x),
                          lambda x: assign_each(x, lambda a: 180./np.pi),
                          0.5, 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # radians
        mathematical_core("radians", stype,
                          lambda x: mx.sym.radians(x),
                          lambda x: np.radians(x),
                          lambda x: assign_each(x, lambda a: np.pi / 180.),
                          0.6, 1, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # sinh
        mathematical_core("sinh", stype, lambda x: mx.sym.sinh(x), lambda x: np.sinh(x), lambda x: np.cosh(x), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # cosh
        mathematical_core("cosh", stype, lambda x: mx.sym.cosh(x), lambda x: np.cosh(x), lambda x: np.sinh(x), 5, 5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # tanh
        mathematical_core("tanh", stype,
                          lambda x: mx.sym.tanh(x),
                          lambda x: np.tanh(x),
                          lambda x: 1. - np.tanh(x) ** 2,
                          0.5, 1, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # arcsinh
        mathematical_core("arcsinh", stype, lambda x: mx.sym.arcsinh(x), lambda x: np.arcsinh(x),
                          lambda x: 1./(x**2 + 1.)**(1./2.), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # arccosh
        mathematical_core("arccosh", stype, lambda x: mx.sym.arccosh(x), lambda x: np.arccosh(x),
                          lambda x: 1./(x**2 - 1.)**(1./2.), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # arctanh
        mathematical_core("arctanh", stype, lambda x: mx.sym.arctanh(x), lambda x: np.arctanh(x),
                          lambda x: -1./(x**2 - 1.), 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # log1p
        mathematical_core("log1p", stype, lambda x: mx.sym.log1p(x), lambda x: np.log1p(x),
                          lambda x: 1. / (1.0 + x), 0.5, 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)
        # expm1
        mathematical_core("expm1", stype, lambda x: mx.sym.expm1(x), lambda x: np.expm1(x),
                          lambda x: np.exp(x), 0.5, 0.5, output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # log10
        mathematical_core("log10", stype, lambda x: mx.sym.log10(x), lambda x: np.log10(x),
                          lambda x: (1 / x), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # log2
        mathematical_core("log2", stype, lambda x: mx.sym.log2(x), lambda x: np.log2(x),
                          lambda x: (1 / x), output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # rint
        mathematical_core("rint", stype, lambda x: mx.sym.rint(x), lambda x: np.rint(x), force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        # fix
        mathematical_core("fix", stype, lambda x: mx.sym.fix(x), lambda x: np.fix(x), force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        try:
            from scipy import special as scipy_special
            import_succeeded = True
            # gamma
            mathematical_core("gamma", stype,
                              lambda x: mx.sym.gamma(x),
                              lambda x: scipy_special.gamma(x),
                              lambda x: scipy_special.gamma(x) * scipy_special.psi(x),
                              output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)
            # gammaln
            mathematical_core("gammaln", stype,
                              lambda x: mx.sym.gammaln(x),
                              lambda x: scipy_special.gammaln(x),
                              lambda x: scipy_special.psi(x),
                              output_grad_stype=output_grad_stype, force_overlap=force_overlap, density=density, ograd_density=ograd_density)

        except:
            if import_succeeded == False:
                print("Could not import scipy. Skipping unit tests for special functions")
            else:
                raise

    for i in range(1):
        print("pass", i)
        for density in [0.0, random.uniform(0, 1), 1.0]:
            for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
                for force_overlap in [False, True]:
                    test_mathematical_core('default', force_overlap=force_overlap,
                                          density=density, ograd_density=ograd_density)
                    test_mathematical_core('row_sparse', force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density)
                    test_mathematical_core('row_sparse', output_grad_stype='default',
                                           force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density)
                    test_mathematical_core('row_sparse', output_grad_stype='row_sparse',
                                           force_overlap=force_overlap,
                                           density=density, ograd_density=ograd_density)
    print("Done")

def check_sparse_embedding():
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


def check_sparse_slice():
    def test_csr_slice(shape, slice_input):
        storage_type = 'csr'
        A, _ = rand_sparse_ndarray(shape, storage_type)
        B = A._slice(1, shape[0] - 1) if slice_input else A
        np = B.asnumpy()
        begin = rnd.randint(0, B.shape[0] - 1)
        end = rnd.randint(begin + 1, B.shape[0])
        nd_slice = mx.nd.crop(B, begin=begin, end=end)
        assert same(nd_slice.asnumpy(), np[begin:end]), (nd_slice.asnumpy(), np[begin:end])

    shape = (rnd.randint(7, 15), rnd.randint(1, 10))
    test_csr_slice(shape, True)
    test_csr_slice(shape, False)


def check_sparse_retain():
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


def do_cast(arr, stype):
    if arr.storage_type != stype:
        return mx.nd.cast_storage(arr, storage_type=stype)
    return arr

def test_type(arr, stype):
    if stype is not None:
        assert arr.storage_type == stype
    else:
        assert arr.storage_type == 'default'

# TODO: Requires add_n for backward pass
# def test_sparse_maximum_minimum():
#     def check_sparse_maximum_minimum(stype, grad_stype, expected_result_stype=None):
#         data1 = mx.symbol.Variable('data')
#         data2 = mx.symbol.Variable('data')
#         shape = (3, 4)
#
#         if stype is None or stype == 'default':
#             data_tmp1 = np.random.rand(3,4)
#             data_tmp2 = np.random.rand(3,4)
#             arr_data1 = mx.nd.array(data_tmp1)
#             arr_data2 = mx.nd.array(data_tmp2)
#             data_tmp1 = arr_data1.asnumpy()
#             data_tmp2 = arr_data2.asnumpy()
#             arr_grad1 = mx.nd.empty(shape)
#             arr_grad2 = mx.nd.empty(shape)
#         else:
#             arr_data1 = create_sparse_array(shape, stype, rsp_indices=(1, 3))
#             arr_data2 = create_sparse_array(shape, stype, rsp_indices=(2, 1))
#             data_tmp1 = arr_data1.asnumpy()
#             data_tmp2 = arr_data2.asnumpy()
#             arr_grad1 = create_sparse_array(shape, grad_stype, rsp_indices=(1, 3))
#             arr_grad2 = create_sparse_array(shape, grad_stype, rsp_indices=(2, 1))
#
#         test = mx.sym.maximum(data1,data2) + mx.sym.minimum(data1,data2);
#         exe_test = test.bind(default_context(), args=[arr_data1,arr_data2], args_grad=[arr_grad1,arr_grad2])
#         print("BEGIN FORWARD")
#         exe_test.forward(is_train=True)
#         print("END FORWARD")
#
#         output = exe_test.outputs[0]
#         test_type(output, expected_result_stype)
#         print("BEGIN CAST")
#         out = output.asnumpy()
#         print("END CAST")
#         npout = np.maximum(data_tmp1, data_tmp2) \
#                 + np.minimum(data_tmp1, data_tmp2)
#         assert_almost_equal(out, npout)
#
#         out_grad = mx.nd.empty(shape)
#         out_grad[:] = 2
#         print(out_grad.asnumpy())
#         exe_test.backward(out_grad)
#         np_arr_grad1 = arr_grad1.asnumpy()
#         np_arr_grad2 = arr_grad2.asnumpy()
#
#         npout_grad = np.ones(shape)
#         npout_grad[:] = 2
#         mask1 = (data_tmp1 > data_tmp2).astype('float')
#         mask2 = (data_tmp1 < data_tmp2).astype('float')
#         npout_grad1 = npout_grad * mask1 + npout_grad * mask2
#         npout_grad2 = (npout_grad - npout_grad * mask1) + (npout_grad - npout_grad * mask2)
#
#         assert_almost_equal(np_arr_grad1, npout_grad1)
#         assert_almost_equal(np_arr_grad2, npout_grad2)
#
#     #check_sparse_maximum_minimum('default', 'default', 'default')
#     check_sparse_maximum_minimum('row_sparse', 'row_sparse', 'row_sparse')
#

def test_sparse_unary_with_numerics():
    def check_sparse_simple(name, stype, mxnet_func, forward_numpy_call,
                            backward_numpy_call, output_grad_stype=None):
        if output_grad_stype is None:
            output_grad_stype = stype

        expected_result_type, expected_grad_result_type = \
            get_fw_bw_result_types_2(forward_numpy_call, stype, backward_numpy_call, output_grad_stype)

        shape = (3, 4)
        data = mx.symbol.Variable("data")

        if output_grad_stype != 'default':
            data._set_attr(input_grad_stype_hint=expected_grad_result_type)

        y = mxnet_func(data)
        if stype == 'default':
            xa = np.random.uniform(low=-1.0, high=1.0, size=shape)
            xa_np = xa
        else:
            xa = create_sparse_array(shape, stype, data_init=None, rsp_indices=[1],
                                     modifier_func=lambda a: a - 0.5)
            xa_np = xa.asnumpy()

        if output_grad_stype != 'default':
            out_grad = create_sparse_array(shape, stype, data_init=None, rsp_indices=[1, 2],
                                           modifier_func=lambda a: a - 0.5)
            out_grad_np = out_grad.asnumpy()
        else:
            out_grad_np = np.ones(xa.shape)
            out_grad = mx.nd.array(out_grad_np)

        output_np = forward_numpy_call(xa_np)
        input_grad_np = backward_numpy_call(output_np, out_grad_np)

        outputs = check_symbolic_forward(y, [xa], [output_np])
        output = outputs[0]

        assert output.storage_type == expected_result_type

        input_grad_dict = check_symbolic_backward(y, location=[xa], out_grads=[out_grad], expected=[input_grad_np])
        inp_grad = input_grad_dict["data"]

        assert inp_grad.storage_type == expected_grad_result_type

    def check_sparse_function(name, mxnet_func, forward_numpy_call, backward_numpy_call):
        check_sparse_simple(name, 'default', mxnet_func, forward_numpy_call, backward_numpy_call)
        for output_grad_stype in [None, "row_sparse", "default"]:
            check_sparse_simple(name, 'row_sparse', mxnet_func, forward_numpy_call, backward_numpy_call,
                              output_grad_stype=output_grad_stype)

    check_sparse_function('relu',
                          lambda x: mx.sym.relu(x),
                          lambda x: np.maximum(x, 0.0),
                          lambda input, outg: outg * assign_each(input, lambda x: x > 0.0)
                          )

    check_sparse_function('sigmoid',
                          lambda x: mx.sym.sigmoid(x),
                          lambda x: np.divide(1.0, (1.0 + np.exp(-x))),
                          lambda output, outg: outg * assign_each(output, lambda x: x * (1.0 - x))
                        )

if __name__ == '__main__':
    #import nose
    #nose.runmodule()
    #test_sparse_unary_with_numerics()
    check_elemwise_binary_ops()
    #check_sparse_mathematical_core()
