from mxnet.test_utils import *
import sys
import random


def is_scalar(var):
    return False if hasattr(var, "__len__") else True


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


def get_result_type_with_scalar(call, dflt_stype):
    if call is not None and dflt_stype != 'default':
        zero = np.zeros(([1]))
        result = call(zero, 5)

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

def get_fw_bw_result_types_with_scalar(forward_numpy_call,  fwd_res_dflt,
                                       backward_numpy_call, bwd_res_dflt):
    return (get_result_type_with_scalar(forward_numpy_call,  fwd_res_dflt),
            get_result_type_with_scalar(backward_numpy_call, bwd_res_dflt))

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


def all_zero(var):
    return 0

def test_elemwise_binary_ops():
    def test_elemwise_binary_op(name, lhs_stype, rhs_stype, shape,
                                forward_mxnet_call, forward_numpy_call, backward_numpy_call,
                                lhs_grad_stype,
                                rhs_grad_stype,
                                expected_result_storage_type=None,
                                modifier_func=None,
                                lhs_density=.5,
                                rhs_density=.5,
                                force_lr_overlap=False,
                                force_grad_overlap=False,
                                ograd_density=0.0,
                                skip_gradient_check=False,
                                verbose=False):
        if verbose is True:
          print("testing:", name)

        if lhs_grad_stype is None:
            lhs_grad_stype = lhs_stype
        if rhs_grad_stype is None:
            rhs_grad_stype = rhs_stype

        lhs_grad_stype = get_result_type_3(backward_numpy_call, lhs_grad_stype)
        rhs_grad_stype = get_result_type_3(backward_numpy_call, rhs_grad_stype)

        # Output type should be same as lvalue type, unless otherwise specified
        if expected_result_storage_type is None:
            if lhs_stype == 'default' or rhs_stype == 'default':
                expected_result_storage_type = 'default'
            else:
                expected_result_storage_type = lhs_stype

        lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
        rhs = mx.symbol.Variable('rhs', stype=rhs_stype)

        grad_stypes = dict()
        grad_stypes['lhs'] = lhs_grad_stype
        grad_stypes['rhs'] = rhs_grad_stype

        if lhs_stype == 'default':
            lhs_nd = rand_ndarray(shape, 'default')
            if abs(lhs_density) < 1e-4:
                func = all_zero
            else:
                func = modifier_func
            lhs_nd = mx.nd.array(assign_each(lhs_nd.asnumpy(), func))
        else:
            lhs_nd = create_sparse_array_zd(
                shape, lhs_stype, density=lhs_density,
                modifier_func=modifier_func,
                data_init=.3,  # REMOVE ME
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=lhs_density,
                    force_indices=[(shape[0]/2)] if force_lr_overlap is True else None
                ))

        if rhs_stype == 'default':
            rhs_nd = rand_ndarray(shape, 'default')
            if abs(rhs_density) < 1e-4:
                func = all_zero
            else:
                func = modifier_func
            rhs_nd = mx.nd.array(assign_each(rhs_nd.asnumpy(), func))
        else:
            rhs_nd = create_sparse_array_zd(
                shape, rhs_stype, density=rhs_density,
                modifier_func=modifier_func,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=rhs_density,
                    force_indices=[(shape[0]/2)] if force_lr_overlap is True else None
                    ))

        lhs_np = lhs_nd.asnumpy()
        rhs_np = rhs_nd.asnumpy()

        if verbose is True:
            print("lhs input: {}".format(lhs_np))
            print("rhs input: {}".format(rhs_np))

        out_np = forward_numpy_call(lhs_np, rhs_np)

        if verbose is True:
            print("out_np: {}".format(out_np))

        test = forward_mxnet_call(lhs, rhs)

        location = {'lhs': lhs_nd, 'rhs': rhs_nd}

        outputs = check_symbolic_forward(test, location, [out_np], equal_nan=True)
        assert len(outputs) == 1
        assert outputs[0].stype == expected_result_storage_type

        if verbose is True:
            print ("mx forward output: ", outputs[0].asnumpy())
            print ("lhs_nd: ", lhs_nd.stype)
            print ("rhs_nd: ", rhs_nd.stype)
            print ("forward output: ", outputs[0].stype)

        if outputs[0].stype != 'default':
            out_grad = create_sparse_array_zd(
                shape, outputs[0].stype, density=ograd_density,
                data_init=1,
                modifier_func=lambda x: 2,
                rsp_indices=gen_rsp_random_indices(
                    shape,
                    density=ograd_density,
                    force_indices=[(shape[0]/2)] if force_grad_overlap is True else None
                ))
        else:
            if abs(ograd_density) < 1e-4:
                out_grad = mx.nd.array(np.zeros(shape))
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
                                                grad_stypes=grad_stypes,
                                                equal_nan=True)

        if verbose is True:
            print("ingrad_lhs", igrads_result['lhs'].asnumpy())
            print("ingrad_rhs", igrads_result['rhs'].asnumpy())

        assert len(igrads_result) == 2

        if lhs_grad_stype is not None:
            assert igrads_result['lhs'].stype == lhs_grad_stype
        if rhs_grad_stype is not None:
            assert igrads_result['rhs'].stype == rhs_grad_stype

        if skip_gradient_check is not True:
            check_numeric_gradient(test, location,
                                   #numeric_eps=numeric_eps,
                                   grad_stype_dict=grad_stypes)

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
        if lstype == 'default' and rstype == 'default':
            return 'default'
        elif rstype != 'default':
            return rstype
        return lstype

    def check_elemwise_binary_ops(lhs_stype, rhs_stype, shape,
                                  lhs_grad_stype=None, rhs_grad_stype=None,
                                  lhs_density=.5, rhs_density=.5,
                                  force_lr_overlap=False,
                                  force_grad_overlap=False,
                                  ograd_density=0.0):
        test_elemwise_binary_op("elemwise_add", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_add(l, r),
                                lambda l, r: l + r,
                                lambda outg, l, r: (outg, outg),
                                lhs_grad_stype, rhs_grad_stype,
                                ograd_density=ograd_density,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                verbose=False)

        test_elemwise_binary_op("elemwise_sub", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_sub(l, r),
                                lambda l, r: l - r,
                                lambda outg, l, r: (outg, -outg),
                                lhs_grad_stype, rhs_grad_stype,
                                ograd_density=ograd_density,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density,
                                rhs_density=rhs_density,
                                verbose=False)

        test_elemwise_binary_op("elemwise_mul", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_mul(l, r),
                                lambda l, r: l * r,
                                lambda outg, l, r: (outg * r, outg * l),
                                least_sparse(lhs_stype, rhs_stype),
                                least_sparse(lhs_stype, rhs_stype),
                                expected_result_storage_type=least_sparse(lhs_stype, rhs_stype),
                                ograd_density=ograd_density,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                verbose=False)

        test_elemwise_binary_op("elemwise_div", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.elemwise_div(l, r),
                                lambda l, r: l / r,
                                lambda outg, l, r: (outg * (1/r), outg * (-l/(r*r))),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                ograd_density=ograd_density,
                                expected_result_storage_type='default',
                                skip_gradient_check=True,
                                verbose=False)

        test_elemwise_binary_op("maximum", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.maximum(l, r),
                                lambda l, r: np.maximum(l, r),
                                lambda outg, l, r: (outg * ge(l, r), outg * lt(l, r)),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                skip_gradient_check=True,
                                ograd_density=ograd_density,
                                verbose=False)

        test_elemwise_binary_op("minimum", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.minimum(l, r),
                                lambda l, r: np.minimum(l, r),
                                lambda outg, l, r: (outg * le(l, r), outg * gt(l, r)),
                                lhs_grad_stype, rhs_grad_stype,
                                modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                ograd_density=ograd_density,
                                skip_gradient_check=True,
                                verbose=False)

        test_elemwise_binary_op("hypot", lhs_stype, rhs_stype, shape,
                                lambda l, r: mx.sym.hypot(l, r),
                                lambda l, r: np.hypot(l, r),
                                lambda outg, l, r: (
                                                outg * assign_each2(l, r, lambda a, b: a/np.sqrt(a * a + b * b)),
                                                outg * assign_each2(l, r, lambda a, b: b/np.sqrt(a * a + b * b))
                                            ),
                                lhs_grad_stype, rhs_grad_stype,
                                force_lr_overlap=force_lr_overlap,
                                force_grad_overlap=force_grad_overlap,
                                lhs_density=lhs_density, rhs_density=rhs_density,
                                ograd_density=ograd_density,
                                skip_gradient_check=True,
                                verbose=False)

        # test_elemwise_binary_op("power", lhs_stype, rhs_stype, shape,
        #                         lambda l, r: mx.sym.pow(l, r),
        #                         lambda l, r: np.power(l, r),
        #                         lambda outg, l, r: (
        #                             outg * assign_each2(l, r, lambda a, b: np.power(a, b - 1) * b),
        #                             outg * assign_each2(l, r, lambda a, b: np.power(a, b) * np.log(a))
        #                         ),
        #                         lhs_grad_stype, rhs_grad_stype,
        #                         modifier_func=lambda a: a if abs(a) > 0.25 else abs(a) + 1,
        #                         force_lr_overlap=force_lr_overlap,
        #                         force_grad_overlap=force_grad_overlap,
        #                         lhs_density=lhs_density, rhs_density=rhs_density,
        #                         ograd_density=ograd_density,
        #                         skip_gradient_check=True,
        #                         verbose=False)

    # Run basic tests
    for ii in range(1):
        for lhs_density in [0.0, random.uniform(0, 1), 1.0]:
        #for lhs_density in [0.0]:
            for rhs_density in [0.0, random.uniform(0, 1), 1.0]:
            #for rhs_density in [0.0]:
                for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
                #for ograd_density in [0.0]:
                    for force_lr_overlap in [False, True]:
                    #for force_lr_overlap in [False]:
                      for force_grad_overlap in [False, True]:
                      #for force_grad_overlap in [False]:
                        shape = rand_shape_2d()
                        #shape = (1,1)
                        check_elemwise_binary_ops('default', 'default', shape,
                                                  lhs_density=lhs_density, rhs_density=rhs_density,
                                                  force_lr_overlap=force_lr_overlap,
                                                  force_grad_overlap=force_grad_overlap,
                                                  ograd_density=ograd_density)
                        check_elemwise_binary_ops('default', 'row_sparse', shape,
                                                  lhs_density=lhs_density, rhs_density=rhs_density,
                                                  force_lr_overlap=force_lr_overlap,
                                                  force_grad_overlap=force_grad_overlap,
                                                  ograd_density=ograd_density)
                        check_elemwise_binary_ops('row_sparse', 'default', shape,
                                                  lhs_density=lhs_density, rhs_density=rhs_density,
                                                  force_lr_overlap=force_lr_overlap,
                                                  force_grad_overlap=force_grad_overlap,
                                                  ograd_density=ograd_density)
                        check_elemwise_binary_ops('row_sparse', 'row_sparse', shape,
                                                  lhs_grad_stype='row_sparse', rhs_grad_stype='row_sparse',
                                                  lhs_density=lhs_density, rhs_density=rhs_density,
                                                  force_lr_overlap=force_lr_overlap,
                                                  force_grad_overlap=force_grad_overlap,
                                                  ograd_density=ograd_density)


def as_dense(arr):
  if arr.stype != 'default':
    return mx.nd.cast_storage(arr, stype='default')
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


def check_sparse_mathematical_core(name, stype,
                                   forward_mxnet_call, forward_numpy_call, backward_numpy_call=None,
                                   rhs_arg=None, data_init=9., grad_init=2., output_grad_stype=None,
                                   input_grad_stype=None, force_overlap=False, density=.5,
                                   ograd_density=.5, verbose=False):
  if verbose is True:
    print("TESTING: " + name)

  data = mx.symbol.Variable('data', stype=stype)

  if input_grad_stype is None:
    input_grad_stype = stype

  if rhs_arg is not None:
    if is_scalar(rhs_arg):
      expected_result_type, expected_grad_result_type = \
        get_fw_bw_result_types_with_scalar(forward_numpy_call, stype,
                                           backward_numpy_call, input_grad_stype)
    else:
      expected_result_type, expected_grad_result_type = \
        get_fw_bw_result_types_2(forward_numpy_call, stype,
                                 backward_numpy_call, input_grad_stype)
  else:
    expected_result_type, expected_grad_result_type = \
      get_fw_bw_result_types(forward_numpy_call, stype,
                             backward_numpy_call, input_grad_stype)

  grad_stypes = list(expected_grad_result_type)

  #shape = rand_shape_2d()
  #shape = (3,4)
  #shape = (9,1)
  #shape = (1,1)
  #shape = (10, 8)
  shape = (2, 5)

  if verbose is True:
    print("Shape: ", shape, "density: ", density, "force_overlap", force_overlap)

  if stype == 'default':
    data_tmp = np.zeros(shape)
    if abs(density) >= 1e-4:
      data_tmp[:] = data_init
    arr_data = mx.nd.array(data_tmp)
  else:
    arr_data = create_sparse_array_zd(
      shape, stype, density=density,
      data_init=data_init,
      rsp_indices=gen_rsp_random_indices(
        shape,
        density=density,
        force_indices=[(shape[0]/2)] if force_overlap is True else None
        #force_indices=[(1, 2)]
      )
    )
    data_tmp = arr_data.asnumpy()
    if verbose is True:
      print("arr_data indices", arr_data.indices.asnumpy())

  if verbose is True:
    print("input", data_tmp)

  if backward_numpy_call is None:
    arr_grad = None
  elif expected_grad_result_type == 'default':
    if abs(density) < 1e-4:
      arr_grad = mx.nd.zeros(shape)
    else:
      arr_grad = mx.nd.ones(shape)
  else:
    arr_grad = create_sparse_array_zd(
      shape,
      expected_grad_result_type,
      density=density,
      data_init=1,
      rsp_indices=gen_rsp_random_indices(
        shape,
        density=density,
        force_indices=[(shape[0]/2)] if force_overlap is True else None
        #force_indices=[(1, 2)]
      )
    )

  if rhs_arg is not None:
    test = forward_mxnet_call(data, rhs_arg)
  else:
    test = forward_mxnet_call(data)

  args = list()
  args.append(arr_data)

  if arr_grad is not None:
    exe_test = test.bind(default_context(), args=args, args_grad=[arr_grad])
  else:
    exe_test = test.bind(default_context(), args=args)

  exe_test.forward(is_train=True)
  assert exe_test.outputs[0].stype == expected_result_type
  out = exe_test.outputs[0].asnumpy()

  if rhs_arg is not None:
    npout = forward_numpy_call(data_tmp, rhs_arg)
  else:
    npout = forward_numpy_call(data_tmp)

  if verbose is True:
    print("out", out)
    print("npout", npout)

  assert_almost_equal(out, npout, equal_nan=True)

  if backward_numpy_call is not None:
    if output_grad_stype == 'default' or output_grad_stype is None:
      out_grad = mx.nd.empty(shape)
      out_grad[:] = grad_init
    else:
      out_grad = create_sparse_array_zd(
        shape, output_grad_stype,
        density=density,
        data_init=grad_init,
        rsp_indices=gen_rsp_random_indices(
          shape,
          density=ograd_density,
          force_indices=[(shape[0]/2)] if force_overlap is True else None))

    npout_grad = out_grad.asnumpy()

    if verbose is True:
      print("npout_grad", npout_grad)

    if rhs_arg is not None:
      temp = backward_numpy_call(data_tmp, rhs_arg)
    else:
      temp = backward_numpy_call(data_tmp)
    input_grad = npout_grad * temp

    if verbose is True:
      print(arr_grad.asnumpy())
    exe_test.backward(out_grad)
    if verbose is True:
      print(arr_grad.asnumpy())

    assert arr_grad.stype == expected_grad_result_type

    arr_grad = arr_grad.asnumpy()

    if verbose is True:
      print(name)
      print("arr_grad", arr_grad)
      print("input_grad", input_grad)

    assert_almost_equal(arr_grad, input_grad, equal_nan=True)


def test_sparse_mathematical_core():
  def util_sign(a):
    if np.isclose(a, -0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
      return 0
    elif np.isclose(a, 0, rtol=1.e-3, atol=1.e-3, equal_nan=True):
      return 0
    elif a < 0.0:
      return -1
    else:  # a > 0.0:
      return 1

  # Check scalar binary operators
  def check_binary_op_with_scalar(stype, output_grad_stype=None,
                                  density=.5, ograd_density=.5,
                                  force_overlap=False,):
    # mul_scalar
    check_sparse_mathematical_core("mul_scalar", stype,
                                   lambda x, y: x * y,
                                   lambda x, y: x * y,
                                   lambda input, rhs: rhs,
                                   rhs_arg=5.0,
                                   data_init=2, grad_init=3,
                                   output_grad_stype=output_grad_stype,
                                   density=density, ograd_density=ograd_density,
                                   force_overlap=force_overlap,
                                   verbose=True)

  # Check many basic unary operators
  def check_mathematical_core(stype, output_grad_stype=None, force_overlap=False,
                              density=.5, ograd_density=.5):
    # sqrt
    check_sparse_mathematical_core("sqrt", stype,
                                   lambda x: mx.sym.sqrt(x),
                                   lambda x: np.sqrt(x),
                                   lambda x: 1.0/(2.0 * np.sqrt(x)),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density,
                                   verbose=False)

    # rsqrt
    check_sparse_mathematical_core("rsqrt", stype,
                                   lambda x: mx.sym.rsqrt(x),
                                   lambda x: 1 / np.sqrt(x),
                                   lambda x: -(1.0 / (2.0 * x * np.sqrt(x))),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # square
    check_sparse_mathematical_core("square", stype,
                                   lambda x: mx.sym.square(x),
                                   lambda x: np.square(x),
                                   lambda x: 2 * x,
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # tan
    check_sparse_mathematical_core("tan", stype, lambda x: mx.sym.tan(x), lambda x: np.tan(x), lambda x: np.tan(x) ** 2 + 1,
                                   output_grad_stype=output_grad_stype, density=density,
                                   ograd_density=ograd_density)

    # abs
    check_sparse_mathematical_core("abs", stype,
                                   lambda x: mx.sym.abs(x),
                                   lambda x: np.abs(x),
                                   lambda x: assign_each(x, function=util_sign),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # negative
    check_sparse_mathematical_core("negative", stype, lambda x: mx.sym.negative(x), lambda x: np.negative(x),
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # floor
    check_sparse_mathematical_core("floor", stype, lambda x: mx.sym.floor(x), lambda x: np.floor(x),
                                   force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # ceil
    check_sparse_mathematical_core("ceil", stype, lambda x: mx.sym.ceil(x), lambda x: np.ceil(x),
                                   force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # sign
    check_sparse_mathematical_core("sign", stype, lambda x: mx.sym.sign(x),
                                   lambda x: np.sign(x), lambda x: np.zeros(x.shape),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # cos
    check_sparse_mathematical_core("cos", stype, lambda x: mx.sym.cos(x), lambda x: np.cos(x), lambda x: -np.sin(x),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # sin
    check_sparse_mathematical_core("sin", stype, lambda x: mx.sym.sin(x), lambda x: np.sin(x), lambda x: np.cos(x),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # arcsin
    check_sparse_mathematical_core("arcsin", stype,
                                   lambda x: mx.sym.arcsin(x),
                                   lambda x: np.arcsin(x),
                                   lambda x: 1. / (1. - x ** 2) ** (1. / 2.),
                                   data_init=0.5, grad_init=0.5,
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # arccos
    check_sparse_mathematical_core("arccos", stype, lambda x: mx.sym.arccos(x), lambda x: np.arccos(x),
                                   lambda x: -1. / (1. - x ** 2.) ** (1. / 2.),
                                   data_init=0.5, grad_init=0.5,
                                   output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # arctan
    check_sparse_mathematical_core("arctan", stype, lambda x: mx.sym.arctan(x), lambda x: np.arctan(x),
                                   lambda x: 1. / (x ** 2. + 1.),
                                   data_init=0.5, grad_init=0.5,
                                   output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # degrees
    check_sparse_mathematical_core("degrees", stype,
                                   lambda x: mx.sym.degrees(x),
                                   lambda x: np.degrees(x),
                                   lambda x: assign_each(x, lambda a: 180./np.pi),
                                   data_init=0.5, grad_init=0.5,
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # radians
    check_sparse_mathematical_core("radians", stype,
                                   lambda x: mx.sym.radians(x),
                                   lambda x: np.radians(x),
                                   lambda x: assign_each(x, lambda a: np.pi / 180.),
                                   data_init=0.6, grad_init=1,
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # sinh
    check_sparse_mathematical_core("sinh", stype, lambda x: mx.sym.sinh(x), lambda x: np.sinh(x), lambda x: np.cosh(x),
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # cosh
    check_sparse_mathematical_core("cosh", stype, lambda x: mx.sym.cosh(x),
                                   lambda x: np.cosh(x), lambda x: np.sinh(x),
                                   data_init=5, grad_init=5,
                                   output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                   density=density, ograd_density=ograd_density)

    # tanh
    check_sparse_mathematical_core("tanh", stype,
                                   lambda x: mx.sym.tanh(x),
                                   lambda x: np.tanh(x),
                                   lambda x: 1. - np.tanh(x) ** 2,
                                   data_init=0.5, grad_init=1,
                                   output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density,
                                   ograd_density=ograd_density)

    # arcsinh
    check_sparse_mathematical_core("arcsinh", stype, lambda x: mx.sym.arcsinh(x), lambda x: np.arcsinh(x),
                                   lambda x: 1./(x**2 + 1.)**(1./2.), output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # arccosh
    check_sparse_mathematical_core("arccosh", stype, lambda x: mx.sym.arccosh(x), lambda x: np.arccosh(x),
                                   lambda x: 1./(x**2 - 1.)**(1./2.), output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # arctanh
    check_sparse_mathematical_core("arctanh", stype, lambda x: mx.sym.arctanh(x), lambda x: np.arctanh(x),
                                   lambda x: -1./(x**2 - 1.),
                                   data_init=0.5, output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # log1p
    check_sparse_mathematical_core("log1p", stype, lambda x: mx.sym.log1p(x), lambda x: np.log1p(x),
                                   lambda x: 1. / (1.0 + x),
                                   data_init=0.5, grad_init=0.5,
                                   output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # # exp
    # check_sparse_mathematical_core("exp", stype,
    #                   lambda x: mx.sym.exp(x),
    #                   lambda x: np.exp(x),
    #                   lambda x: x / np.exp(x),
    #                   0.5, 0.5, output_grad_stype=output_grad_stype,
    #                   force_overlap=force_overlap, density=density,
    #                   ograd_density=ograd_density,
    #                   verbose=True)

    # expm1
    check_sparse_mathematical_core("expm1", stype, lambda x: mx.sym.expm1(x), lambda x: np.expm1(x),
                                   lambda x: np.exp(x),
                                   data_init=0.5, grad_init=0.5,
                                   output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # log10
    check_sparse_mathematical_core("log10", stype, lambda x: mx.sym.log10(x), lambda x: np.log10(x),
                                   lambda x: (1 / x), output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # log2
    check_sparse_mathematical_core("log2", stype, lambda x: mx.sym.log2(x), lambda x: np.log2(x),
                                   lambda x: (1 / x), output_grad_stype=output_grad_stype,
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # rint
    check_sparse_mathematical_core("rint", stype, lambda x: mx.sym.rint(x), lambda x: np.rint(x),
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    # fix
    check_sparse_mathematical_core("fix", stype, lambda x: mx.sym.fix(x), lambda x: np.fix(x),
                                   force_overlap=force_overlap, density=density, ograd_density=ograd_density)

    try:
      from scipy import special as scipy_special
      import_succeeded = True
      # gamma
      check_sparse_mathematical_core("gamma", stype,
                                     lambda x: mx.sym.gamma(x),
                                     lambda x: scipy_special.gamma(x),
                                     lambda x: scipy_special.gamma(x) * scipy_special.psi(x),
                                     output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                     density=density, ograd_density=ograd_density)
      # gammaln
      check_sparse_mathematical_core("gammaln", stype,
                                     lambda x: mx.sym.gammaln(x),
                                     lambda x: scipy_special.gammaln(x),
                                     lambda x: scipy_special.psi(x),
                                     output_grad_stype=output_grad_stype, force_overlap=force_overlap,
                                     density=density, ograd_density=ograd_density)

    except:
      if import_succeeded == False:
        print("Could not import scipy. Skipping unit tests for special functions")
      else:
        raise

  for i in range(1):
    print("pass", i)
    #for density in [0.0, random.uniform(0, 1), 1.0]:
    #for density in [1.0]:
    for density in [.5]:
      #for ograd_density in [0.0, random.uniform(0, 1), 1.0]:
      #for ograd_density in [1.0]:
      for ograd_density in [.25]:
        #for force_overlap in [False, True]:
        #for force_overlap in [True]:
        for force_overlap in [False]:
          # Check unary ops (unary fwd, binary bwd)
          # check_mathematical_core('default', force_overlap=force_overlap,
          #                         density=density, ograd_density=ograd_density)
          # check_mathematical_core('row_sparse', force_overlap=force_overlap,
          #                         density=density, ograd_density=ograd_density)
          # check_mathematical_core('row_sparse', output_grad_stype='default',
          #                         force_overlap=force_overlap,
          #                         density=density, ograd_density=ograd_density)
          # check_mathematical_core('row_sparse', output_grad_stype='row_sparse',
          #                         force_overlap=force_overlap,
          #                         density=density, ograd_density=ograd_density)

          # Check binary with scalar ops
          # check_binary_op_with_scalar('default',
          #                             density=density,
          #                             ograd_density=ograd_density,
          #                             force_overlap=force_overlap)
          # check_binary_op_with_scalar('row_sparse',
          #                             density=density,
          #                             ograd_density=ograd_density,
          #                             force_overlap=force_overlap)
          # check_binary_op_with_scalar('row_sparse', output_grad_stype='default',
          #                             density=density,
          #                             ograd_density=ograd_density,
          #                             force_overlap=force_overlap)
          # check_binary_op_with_scalar('row_sparse',
          #                             output_grad_stype='row_sparse',
          #                             density=density, ograd_density=ograd_density,
          #                             force_overlap=force_overlap)
          check_binary_op_with_scalar('csr',
                                      output_grad_stype='csr',
                                      density=density,
                                      ograd_density=ograd_density,
                                      force_overlap=force_overlap)


def check_elemwise_add_ex(lhs_stype, rhs_stype, shape, lhs_grad_stype=None, rhs_grad_stype=None):
    lhs = mx.symbol.Variable('lhs', stype=lhs_stype)
    rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
    lhs_nd = rand_ndarray(shape, lhs_stype)
    rhs_nd = rand_ndarray(shape, rhs_stype)
    lhs_np = lhs_nd.asnumpy()
    rhs_np = rhs_nd.asnumpy()

    out_np = lhs_np + rhs_np
    test = mx.symbol.elemwise_add(lhs, rhs)
    location = {'lhs': lhs_nd, 'rhs': rhs_nd}
    check_symbolic_forward(test, location, [out_np])
    check_numeric_gradient(test, location)
    grad_stypes = {}
    if lhs_grad_stype is not None and lhs_grad_stype != 'default':
        grad_stypes['lhs'] = lhs_grad_stype
    if rhs_grad_stype is not None and rhs_grad_stype != 'default':
        grad_stypes['rhs'] = rhs_grad_stype
    check_symbolic_backward(test, location, [out_np], [out_np, out_np],
                            grad_stypes=grad_stypes)


def test_elemwise_add_ex():
    shapes = [rand_shape_2d(), rand_shape_3d()]
    for shape in shapes:
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
    idx1 = mx.nd.array([0], dtype=np.int64);
    idx2 = mx.nd.array([1], dtype=np.int64);
    sp_nd1 = mx.nd.row_sparse(val1, idx1, shape)
    sp_nd2 = mx.nd.row_sparse(val2, idx2, shape)
    ds_nd = mx.nd.array(ds_np)

    # sparse + sparse = sparse
    sp_data1 = mx.symbol.Variable('sp_data1', stype='row_sparse')
    sp_data2 = mx.symbol.Variable('sp_data2', stype='row_sparse')
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
        dns_out = mx.nd.cast_storage(rsp, stype='default')
        dns_expected = np.zeros(shape, dtype=default_dtype())
        if row_idx is not None:
            for k, v in enumerate(row_idx):
                dns_expected[v, :] = data[k]
        assert same(dns_out.asnumpy(), dns_expected)

    def test_dns_to_rsp(shape):
        dns_in = rand_ndarray(shape, 'default')
        rsp_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), stype='row_sparse')
        ret = mx.nd.cast_storage(rsp_out, stype='default')
        assert same(ret.asnumpy(), dns_in.asnumpy())

    def test_csr_to_dns(shape, density):
        csr_in, (indptr, indices, values) = rand_sparse_ndarray(shape, 'csr', density)
        dns_out = csr_in.todense()
        assert same(csr_in.asnumpy(), dns_out.asnumpy())

    def test_dns_to_csr(shape, density):
        csr_in, (indptr, colidx, data) = rand_sparse_ndarray(shape, 'csr', density)
        dns_in = csr_in.todense()
        csr_out = mx.nd.cast_storage(mx.nd.array(dns_in, dtype=default_dtype()), stype='csr')
        assert same(csr_in.asnumpy(), csr_out.asnumpy())

    shape = rand_shape_2d()
    if default_context().device_type is 'cpu':
        test_rsp_to_dns(shape)
        test_dns_to_rsp(shape)

    density = [1.00, 0.50, 0.10, 0.05, 0.01]
    for d in density:
        test_csr_to_dns((rnd.randint(1, 10), rnd.randint(  1,   64)), d)
        test_dns_to_csr((rnd.randint(1, 10), rnd.randint(  1,   31)), d) # test gpu thread kernel
        test_dns_to_csr((rnd.randint(1, 10), rnd.randint( 32,  512)), d) # test gpu warp   kernel
        test_dns_to_csr((rnd.randint(1, 10), rnd.randint(513, 1024)), d) # test gpu block  kernel


def test_sparse_dot():
    def test_dot_csr(lhs_shape, rhs_shape, rhs_stype, trans_lhs, density=1):
        lhs_nd = rand_ndarray(lhs_shape, 'csr', 1)
        lhs_dns = lhs_nd.todense()
        rhs_nd = rand_ndarray(rhs_shape, rhs_stype, density=density)
        rhs_dns = rhs_nd if rhs_stype == 'default' else rhs_nd.todense()
        out = mx.nd.dot(lhs_nd, rhs_dns, transpose_a=trans_lhs)
        if trans_lhs and default_context().device_type is 'cpu':
            assert out.stype == 'row_sparse'
        else:
            assert out.stype == 'default'
        out_expected = mx.nd.dot(lhs_dns, rhs_dns, transpose_a=trans_lhs)
        out_np = out_expected.asnumpy()
        backward_trans = not trans_lhs
        rhs_backward_grad = mx.nd.dot(lhs_dns, out_expected, transpose_a=backward_trans).asnumpy()
        assert_almost_equal(out.asnumpy(), out_np, rtol=1e-4, atol=1e-5)

        # test symbolic forward
        lhs = mx.symbol.Variable('lhs', stype='csr')
        rhs = mx.symbol.Variable('rhs', stype=rhs_stype)
        test = mx.symbol.dot(lhs, rhs, transpose_a=trans_lhs)
        location = {'lhs': lhs_nd, 'rhs': rhs_nd}
        expected = {'rhs': rhs_backward_grad}
        check_symbolic_forward(test, location, [out_np], rtol=1e-3, atol=1e-4)
        # test symbolic backward
        check_symbolic_backward(test, location, [out_np], expected,
                                grad_req={'lhs': 'null', 'rhs': 'write'},
                                rtol=1e-3, atol=1e-4)

    lhs_shape = rand_shape_2d(50, 200)
    test_dot_csr(lhs_shape, (lhs_shape[1], 1), 'default', False) # test gpu SpMV
    test_dot_csr(lhs_shape, (lhs_shape[0], 1), 'default', True ) # (vector kernel)
    test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(5, 10)), 'default', False) # test gpu SpMM
    test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(5, 10)), 'default', True ) # (scalar kernel)
    if default_context().device_type is 'cpu':
        test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False)
        test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True )
        test_dot_csr(lhs_shape, (lhs_shape[1], rnd.randint(1, 10)), 'row_sparse', False, 0.05)
        test_dot_csr(lhs_shape, (lhs_shape[0], rnd.randint(1, 10)), 'row_sparse', True , 0.05)


def test_sparse_slice():
    def check_csr_slice(shape, slice_input):
        storage_type = 'csr'
        B, _ = rand_sparse_ndarray(shape, storage_type)
        np = B.asnumpy()
        begin = rnd.randint(0, B.shape[0] - 1)
        end = rnd.randint(begin + 1, B.shape[0])
        nd_slice = mx.nd.crop(B, begin=begin, end=end)
        assert same(nd_slice.asnumpy(), np[begin:end]), (nd_slice.asnumpy(), np[begin:end])

    shape = (rnd.randint(7, 15), rnd.randint(1, 10))
    check_csr_slice(shape, True)
    check_csr_slice(shape, False)


def test_sparse_retain():
    def check_sparse_retain(shape):
        num_rows = shape[0]
        rsp, _ = rand_sparse_ndarray(shape=shape, stype='row_sparse', density=0.5)
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

    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    check_sparse_retain(shape)
    check_sparse_retain(shape_3d)

def do_cast(arr, stype):
  if arr.stype != stype:
    return mx.nd.cast_storage(arr, stype=stype)
  return arr

def test_type(arr, stype):
  if stype is not None:
    assert arr.stype == stype
  else:
    assert arr.stype == 'default'

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
                          backward_numpy_call, output_grad_stype=None,
                          backward_is_use_output=False):
    if output_grad_stype is None:
      output_grad_stype = stype

    expected_result_type, expected_grad_result_type = \
      get_fw_bw_result_types_2(forward_numpy_call, stype, backward_numpy_call, output_grad_stype)

    if backward_is_use_output is True:
      expected_grad_result_type = expected_result_type

    shape = (3, 4)
    data = mx.symbol.Variable("data")

    grad_stypes = list()
    grad_stypes.append(expected_grad_result_type)

    y = mxnet_func(data)
    if stype == 'default':
      xa = np.random.uniform(low=-1.0, high=1.0, size=shape)
      xa_np = xa
    else:
      xa = create_sparse_array(shape, stype, data_init=None, rsp_indices=[1],
                               modifier_func=lambda a: a - 0.5)
      xa_np = xa.asnumpy()

    if output_grad_stype != 'default':
      out_grad = create_sparse_array(shape, output_grad_stype, data_init=None,
                                     rsp_indices=[1, 2],
                                     modifier_func=lambda a: a - 0.5)
      out_grad_np = out_grad.asnumpy()
    else:
      out_grad_np = np.ones(xa.shape)
      out_grad = mx.nd.array(out_grad_np)

    output_np = forward_numpy_call(xa_np)
    input_grad_np = backward_numpy_call(output_np, out_grad_np)

    outputs = check_symbolic_forward(y, [xa], [output_np])
    output = outputs[0]

    assert output.stype == expected_result_type

    input_grad_dict = check_symbolic_backward(y, location=[xa], out_grads=[out_grad],
                                              expected=[input_grad_np],
                                              grad_stypes=grad_stypes)
    inp_grad = input_grad_dict["data"]

    assert inp_grad.stype == expected_grad_result_type

  def check_sparse_function(name, mxnet_func, forward_numpy_call, backward_numpy_call,
                            backward_is_use_output=False):
    check_sparse_simple(name, 'default', mxnet_func, forward_numpy_call, backward_numpy_call)
    for output_grad_stype in [None, "row_sparse", "default"]:
      check_sparse_simple(name, 'row_sparse', mxnet_func, forward_numpy_call, backward_numpy_call,
                          output_grad_stype=output_grad_stype,
                          backward_is_use_output=backward_is_use_output)

  check_sparse_function('relu',
                        lambda x: mx.sym.relu(x),
                        lambda x: np.maximum(x, 0.0),
                        lambda input, outg: outg * assign_each(input, lambda x: x > 0.0))

  check_sparse_function('sigmoid',
                        lambda x: mx.sym.sigmoid(x),
                        lambda x: np.divide(1.0, (1.0 + np.exp(-x))),
                        lambda output, outg: outg * assign_each(output, lambda x: x * (1.0 - x)),
                        backward_is_use_output=True)

def test_sparse_nd_zeros():
    def check_sparse_nd_zeros(stype, shape):
        zero = mx.nd.zeros(shape)
        sparse_zero = mx.nd.zeros(shape=shape, stype=stype)
        assert_almost_equal(sparse_zero.asnumpy(), zero.asnumpy())

    shape = rand_shape_2d()
    check_sparse_nd_zeros('row_sparse', shape)
    check_sparse_nd_zeros('csr', shape)
    check_sparse_nd_zeros('default', shape)


def test_sparse_square_sum():
    dim0 = 30
    dim1 = 30
    axes = [0, 1]
    keepdims = [False, True]
    densities = [0, 0.01, 0.1, 0.2, 0.5]
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        rsp = rand_ndarray(shape, 'row_sparse', density)
        dns = rsp.todense()
        for axis in axes:
            for keepdim in keepdims:
                ret = mx.nd._internal._square_sum(rsp, axis=axis, keepdims=keepdim)
                if axis == 1 and keepdim:
                    assert ret.stype == 'row_sparse'
                else:
                    assert ret.stype == 'default'
                ret_expected = mx.nd.sum(dns*dns, axis=axis, keepdims=keepdim)
                # check forward result
                assert same(ret.asnumpy(), ret_expected.asnumpy())

                # check numeric gradient
                data = mx.sym.Variable('data', stype='row_sparse')
                test = mx._symbol_internal._square_sum(data, axis=axis, keepdims=keepdim)
                check_numeric_gradient(test, [rsp], grad_stype_dict={'data': 'row_sparse'},
                                       atol=1e-2, rtol=0.1)


if __name__ == '__main__':
  # import nose
  # nose.runmodule()

  test_sparse_mathematical_core()
  # test_sparse_unary_with_numerics()
  # test_elemwise_binary_ops()
  print("Done")
