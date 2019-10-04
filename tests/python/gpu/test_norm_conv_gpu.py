# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function

import sys
import os
import mxnet as mx
import numpy as np
from mxnet.test_utils import default_context, set_default_context, assert_almost_equal
from mxnet.cuda_utils import get_sm_arch

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

set_default_context(mx.gpu(0))

# Helper function that returns True/False with equal probability.
def _random_boolean():
    return np.random.randint(0,2) == 0

# Helper function to the normalized convolution tests
# Return the indices (along the feature dimension) that have relu inputs near 0.
def _has_near_zero_outputs(x, b, g, eps, threshold):
    ctx = default_context()
    X = mx.sym.Variable('X')
    B = mx.sym.Variable('B')  # beta, i.e. bias
    G = mx.sym.Variable('G')  # gamma, i.e. scale
    MovMean = mx.sym.Variable('MovMean')
    MovVar = mx.sym.Variable('MovVar')
    feature_shape = b.shape
    mov_mean = mx.nd.zeros(feature_shape, dtype=np.float32, ctx=ctx)
    mov_var = mx.nd.ones(feature_shape, dtype=np.float32, ctx=ctx)
    bn_sym = mx.sym.BatchNorm(data=X,  gamma=G, beta=B, act_type=None,
                            moving_mean=MovMean, moving_var=MovVar,
                            eps=eps, momentum=0.9, fix_gamma=False,
                            use_global_stats=False, output_mean_var=False,
                            cudnn_off=False, name=None, axis=-1)
    args_dict = {'X':x, 'B':b, 'G':g,}
    aux_states_dict = {'MovMean':mov_mean, 'MovVar':mov_var}
    grad_req = {'MovMean':'null', 'MovVar':'null',
              'X':'null', 'W':'null', 'G':'null', 'B':'null'}
    bn_exe = bn_sym.bind(ctx=ctx, args=args_dict,
                       aux_states=aux_states_dict, grad_req=grad_req)
    # Execute forward() graph calculation
    # need is_train=True to keep Batchnorm using the mini-batch mean and variance
    bn_outputs = bn_exe.forward(is_train=True)
    out_data = bn_outputs[0].asnumpy()
    out_data_abs = np.abs(out_data)
    not_feature_axes = (0, 1, 2)
    origin_dist_mins = out_data_abs.min(axis=not_feature_axes)
    bad_indices = np.nonzero(origin_dist_mins < threshold)[0]
    return bad_indices

@with_seed()
def test_norm_convolution():
    ctx = default_context()
    min_cuda_arch = 70
    max_cuda_arch = 70
    cuda_arch = get_sm_arch(ctx.device_id)
    if cuda_arch < min_cuda_arch or cuda_arch > max_cuda_arch:
        print('Bypassing normalized convolution test on cuda arch {} ({} <= arch <= {}).'.format(
            cuda_arch, min_cuda_arch, max_cuda_arch))
        return

    # RN50 layer shapes
    nchw_shapes = [
        ( 64,  256,  56,  56),
        ( 64,  128,  28,  28),
        ( 64,  512,  28,  28),
        ( 64,  256,  14,  14),
        ( 64, 1024,  14,  14),
        ( 64,  512,   7,   7),
        ( 64, 2048,   7,   7),
        (128,   64,  56,  56),
        (128,  256,  56,  56),
        (128,  128,  28,  28),
        (128,  512,  28,  28),
        (128,  256,  14,  14),
        (128, 1024,  14,  14),
        (128,  512,   7,   7),
        (128, 2048,   7,   7),
    ]

    # Make dataset stats (to input to BNStatsFinalize)
    def create_input_stats_np(data_np):
        data_fp32_np = data_np.astype(np.float32)
        not_feature_axes = (0, 1, 2)
        feature_sum_np = data_fp32_np.sum(axis=not_feature_axes)
        feature_sum_squares_np = np.square(data_fp32_np).sum(axis=not_feature_axes)
        return (feature_sum_np, feature_sum_squares_np)

    def create_output_stats(data, output_stats):
        if output_stats:
            data_fp32 = mx.sym.cast(data, np.float32)
            not_feature_axes = (0, 1, 2)
            feature_sum = data_fp32.sum(axis=not_feature_axes)
            feature_sum_squares = data_fp32.square().sum(axis=not_feature_axes)
            return mx.sym.Group([data, feature_sum, feature_sum_squares])
        else:
            return data

    def out_shape(nhwc_inshape, num_filters, kernel_shape, stride, pad):
        (n, h, w, _) = nhwc_inshape
        (kernel_h, kernel_w) = kernel_shape
        (stride_h, stride_w) = stride
        (pad_h, pad_w) = pad
        out_shape_h = 1 + ((h + 2 * pad_h - kernel_h) // stride_h)
        out_shape_w = 1 + ((w + 2 * pad_w - kernel_w) // stride_w)
        return (n, out_shape_h, out_shape_w, num_filters)

    # flip a dataset about the 1st dimension
    def flip(data):
        return mx.sym.flip(data, axis=0)

    # return a new symbol that isolates the input symbol's outputs
    def buffer(sym):
        num_outputs = len(sym.list_outputs())
        if num_outputs == 1:
            return flip(flip(sym))
        else:
            flipped_outputs = [ flip(flip(sym[i])) for i in range(num_outputs)]
            return mx.sym.Group(flipped_outputs)

    # Test fused op without input normalization.  Options for activation and output of stats.
    def finalize_norm_conv_test(nchw_inshape, kernel_shape, num_filter, act_type, stride,
                               pad, output_stats, no_norm, no_conv,
                               eps, momentum):

        # If we are disabling the convolution (no_conv = True), then set kernel=1x1 and weights 1
        if no_conv:
            if kernel_shape != (1,1):
                print('Ignoring kernel_shape {}, forcing 1x1 in no_conv mode.'.format(kernel_shape))
            kernel_shape = (1,1)
        (n, c, h, w) = nchw_inshape
        X = mx.sym.Variable('X')
        W = mx.sym.Variable('W')
        SUM = mx.sym.Variable('SUM')
        SUMSQ = mx.sym.Variable('SUMSQ')
        B = mx.sym.Variable('B')  # beta, i.e. bias
        G = mx.sym.Variable('G')  # gamma, i.e. scale
        # randomly insert buffering here to exercise in-place vs. copy of gamma/beta by Finalize
        if _random_boolean():
            B = buffer(B)
            G = buffer(G)
        MovMean = mx.sym.Variable('MovMean')
        MovVar = mx.sym.Variable('MovVar')

        # make 'ground truth' symbol using standard Batchnorm and Convolution

        if no_norm and (act_type is None):
            normalized = X
        elif no_norm:
            normalized = mx.sym.Activation(data=X, act_type=act_type)
        else:
            normalized = mx.sym.BatchNorm(data=X,  gamma=G, beta=B, act_type=act_type,
                                      moving_mean=MovMean, moving_var=MovVar,
                                      eps=eps, momentum=momentum, fix_gamma=False,
                                      use_global_stats=False, output_mean_var=False,
                                      cudnn_off=False, name=None, axis=-1)
        (r, s) = kernel_shape
        layout = 'NHWC'
        conv_args = {'weight':W, 'num_filter':num_filter, 'kernel':kernel_shape,
                     'stride':stride, 'pad':pad, 'layout':layout, 'name':'conv'}
        conv_sym = mx.sym.Convolution(data=normalized, no_bias=True, **conv_args)
        conv_sym = create_output_stats(conv_sym, output_stats)

        # make symbol-under-test using NormConvolution

        if not no_norm:
            # NormConvolution makes use of conv_args but for stats-apply mode has more inputs:
            conv_args.update({'in_sum':SUM, 'in_sum_squares':SUMSQ, 'gamma':G, 'beta':B,
                              'moving_mean':MovMean, 'moving_var':MovVar, 'eps':eps,
                              'momentum':momentum, 'fix_gamma':False,
                              'output_mean_var':True
                              })
        norm_conv_sym = mx.sym.NormConvolution(X, act_type=act_type,
                                               no_norm=no_norm,
                                                     **conv_args)
        if not output_stats:
            # discard sum and sum_squares outputs before binding
            norm_conv_sym = norm_conv_sym[0]

        # make data inputs
        weight_shape = (num_filter, r, s, c)
        data_shape = (n, h, w, c)
        # x_np = np.fromfunction(lambda n, h, w, c: 3*((n+h+w)%2), data_shape)
        # x = mx.nd.array(x_np, dtype=np.float16, ctx=ctx)
        x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
        (feature_sum_np, feature_sum_squares_np) = create_input_stats_np(x.asnumpy())
        sum = mx.nd.array(feature_sum_np, dtype=np.float32)
        sum_squares = mx.nd.array(feature_sum_squares_np, dtype=np.float32)
        equiv_scale_bias_shape = (c,)
        scale_max = 1.25
        bias_max = 1

        # Comparing gradients of two symbols is tricky when a non-smooth function like 'relu'
        # is part of the function.  We ensure that no relu inputs are near 0 (within a threshold)
        # by trying different beta/gamma values as needed.
        b_np = np.zeros(equiv_scale_bias_shape, dtype=np.float32)
        g_np = np.zeros(equiv_scale_bias_shape, dtype=np.float32)
        indices_to_set = np.array(range(c))
        while len(indices_to_set) > 0:
            for index in indices_to_set:
                b_np[index] = np.random.uniform(-bias_max, bias_max)
                g_np[index] = np.random.uniform(1.0/scale_max, scale_max)
            b = mx.nd.array(b_np, dtype=np.float32, ctx=ctx)
            g = mx.nd.array(g_np, dtype=np.float32, ctx=ctx)
            smallest_norm_fp16 = pow(2, -14)
            threshold = smallest_norm_fp16 / 2
            need_data_check = not no_norm and act_type == 'relu'
            if need_data_check:
                indices_to_set = _has_near_zero_outputs(x, b, g, eps, threshold=threshold)
            else:
                indices_to_set = []

        # mov_mean_np = np.zeros(equiv_scale_bias_shape).astype(np.float32)
        # mov_var_np = np.ones(equiv_scale_bias_shape).astype(np.float32)
        mov_mean_np = np.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape)
        mov_var_np = np.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape)
        # since the models change the moving mean and variance, each model gets their own copy
        mov_mean1 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_mean2 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_var1 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        mov_var2 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        if no_conv:
            weights = mx.ndarray.ones(weight_shape, dtype=np.float16, ctx=ctx)
        else:
            weights = mx.ndarray.random.uniform(-0.20, 0.20, weight_shape, dtype=np.float16, ctx=ctx)
        # These are the tensor's that receive the backpropped gradients (so an output of backward())
        # Copy 1 is for 'ground truth' symbol based on BatchNorm/Convolution ops
        d_x_out_gt = mx.ndarray.zeros(data_shape, dtype=np.float16, ctx=ctx)
        d_w_out_gt = mx.ndarray.zeros(weight_shape, dtype=np.float16, ctx=ctx)
        d_gamma_out_gt = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
        d_beta_out_gt = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
        # Copy 2 is for symbol based on BNStatsFinalize/NormConvolution ops (=ones, not zeros)
        d_x_out = mx.ndarray.ones(data_shape, dtype=np.float16, ctx=ctx)
        d_w_out = mx.ndarray.ones(weight_shape, dtype=np.float16, ctx=ctx)
        d_gamma_out = mx.ndarray.ones(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
        d_beta_out = mx.ndarray.ones(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)

        # bind i/o's to symbols to create executors

        grad_req = {'SUM':'null', 'SUMSQ':'null', 'MovMean':'null', 'MovVar':'null',
                    'X':'write', 'W':'write', 'G':'write', 'B':'write'}

        args_grad_dict_gt = {'X':d_x_out_gt, 'W':d_w_out_gt, 'G':d_gamma_out_gt, 'B':d_beta_out_gt}
        args_grad_dict = {'X':d_x_out, 'W':d_w_out, 'G':d_gamma_out, 'B':d_beta_out}

        args_dict = {'X':x, 'W':weights}
        # conv binding does not need SUM, and SUMSQ, but extra items are OK
        if not no_norm:
            args_dict.update({'B':b, 'G':g, 'SUM':sum, 'SUMSQ':sum_squares})
        gt_aux_states_dict = \
            {'MovMean':mov_mean1, 'MovVar':mov_var1}
        finalize_aux_states_dict = \
            {'MovMean':mov_mean2, 'MovVar':mov_var2}

        conv_exe = conv_sym.bind(ctx=ctx, args=args_dict, args_grad=args_grad_dict_gt,
                                 aux_states=gt_aux_states_dict, grad_req=grad_req)
        norm_conv_exe = norm_conv_sym.bind(ctx=ctx, args=args_dict, args_grad=args_grad_dict,
                                           aux_states=finalize_aux_states_dict, grad_req=grad_req)

        # Execute forward() graph calculation
        # need is_train=True to keep Batchnorm using the mini-batch mean and variance
        conv_outputs = conv_exe.forward(is_train=True)
        # need is_train=True to keep stats from being turned off
        norm_conv_outputs = norm_conv_exe.forward(is_train=True)

        # Check forward outputs
        outputs = ['out', 'sum', 'sum_squares']
        # greater atols needs for 'sum' and 'sum_squares', also if input scale/bias is applied
        if no_norm:
            tols = [(1e-2, 1e-2), (1e-2, 2), (1e-2, 2)]
        else:
            # 'sum' seems to have a large span (e.g. -400K -> +400K) so a large absolute tolerance
            # is needed to cover those cases when the result is near 0 and rtol can't help.
            # One possible source of the large sum tolerance is the internal rounding of the
            # mean to fp16.  Any rounding amount will give a bias to the conv inputs and so the sum.
            # 'sum_squares' doesn't have this issue because rtol handles the always-positive result.
            per_element_atol = 5e-3
            sum_atol = n * h * w * per_element_atol
            tols = [(1e-2, 1e-1), (1e-1, sum_atol), (1e-2, 2)]
        num_outputs = 3 if output_stats else 1
        for idx in range(num_outputs):
            out_name = outputs[idx]
            conv_data = conv_outputs[idx].asnumpy()
            norm_conv_data = norm_conv_outputs[idx].asnumpy()
            (rtol, atol) = tols[idx]
            assert_almost_equal(conv_data, norm_conv_data, rtol=rtol, atol=atol,
                                names=('conv_{}'.format(out_name),
                                       'norm_conv_{}'.format(out_name)))
        # Check backward function
        if no_norm and act_type is not None:
            # gradient calculation not supported for this configuration
            return
        # Create backward gradients
        outshape = out_shape(data_shape, num_filter, kernel_shape, stride, pad)
        d_out_in = mx.ndarray.random.uniform(-0.2, 0.2, outshape,
                                                     dtype=np.float16, ctx=ctx)
        # not really needed
        sum_shape = (num_filter,)
        # gradients on these outputs will be summed into the d_out_in for the ground truth
        # symbol, so make sure these are 0.
        d_sum_in = mx.ndarray.zeros(sum_shape, dtype=np.float32, ctx=ctx)
        d_sum_squares_in = mx.ndarray.zeros(sum_shape, dtype=np.float32, ctx=ctx)
        # d_sum_in = mx.ndarray.random.uniform(0.0, 1.0, sum_shape,
        #                                              dtype=np.float32, ctx=ctx)
        # d_sum_squares_in = mx.ndarray.random.uniform(0.0, 1.0, sum_shape,
        #                                              dtype=np.float32, ctx=ctx)
        # Execute backward() graph calculation
        if output_stats:
            conv_outputs = conv_exe.backward([d_out_in, d_sum_in, d_sum_squares_in])
            norm_conv_outputs = norm_conv_exe.backward([d_out_in, d_sum_in, d_sum_squares_in])
        else:
            conv_outputs = conv_exe.backward([d_out_in,])
            norm_conv_outputs = norm_conv_exe.backward([d_out_in,])

        # Check weight gradient
        out_name = 'd_w'
        assert_almost_equal(d_w_out_gt.asnumpy(), d_w_out.asnumpy(), atol=0.3, rtol=0.1,
                            names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))
        # Check data gradient
        # This check is flakey when act_type = relu because if the two models differ on whether
        # the normalized value is above or below 0, then the gradient may or may-not be backpropped.

        # To fix this test, we could run a separate model with relu off, capture the normalized
        # output and then mask off the gradient comparison when the normalized value is near 0.
        out_name = 'd_x'
        if act_type is None:
            assert_almost_equal(d_x_out_gt.asnumpy(), d_x_out.asnumpy(), atol=0.1, rtol=0.1,
                                names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))
        # Check gamma and beta gradients
        out_name = 'd_gamma'
        assert_almost_equal(d_gamma_out_gt.asnumpy(), d_gamma_out.asnumpy(), atol=10, rtol=0.1,
                            names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))
        out_name = 'd_beta'
        assert_almost_equal(d_beta_out_gt.asnumpy(), d_beta_out.asnumpy(), atol=10, rtol=0.1,
                            names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))

    # Test input normalization function only: no_norm = False, 1x1 unity-weights conv
    # Also test with 'relu' activation on and off.
    print('\nTest of input normalization without convolution function.')
    eps = 1e-4
    momentum = 0.9
    for i in range(len(nchw_shapes)):
        inshape = nchw_shapes[i]
        (n, c, h, w) = inshape
        num_filter = 32
        outshape = (n, num_filter, h, w)
        stride = (1,1)
        print('nchw inshape = {}, outshape = {}, stride = {}'.format(inshape, outshape, stride))
        kernel_shape = (1, 1)
        pad = (0, 0)
        output_stats = False
        act_type = 'relu' if _random_boolean() else None
        print('    kernel= {}, pad = {}, output_stats={}, act_type = {}'.format(
            kernel_shape, pad, output_stats, act_type))
        finalize_norm_conv_test(nchw_inshape=inshape, kernel_shape=kernel_shape,
                                num_filter=num_filter, act_type=act_type,
                                stride=stride, pad=pad, output_stats=output_stats,
                                no_norm=False,
                                no_conv=True, eps=eps, momentum=momentum)

    # Test convolution and stats-gen functions, first without, then with, input normalization.
    # Also test with 'relu' activation on and off.
    for no_norm in [True, False]:
        if no_norm:
            print('\nTest of convolution function, without input normalization.')
        else:
            print('\nTest of convolution function with input normalization.')
        for i in range(len(nchw_shapes)):
            inshape = nchw_shapes[i]
            (n, c, h, w) = inshape
            (stride_h, stride_w) = (1,1)
            # Leverage next test case (if available) to determine outshape, strides
            if i == len(nchw_shapes)-1:
                num_filter = nchw_shapes[i][1]
            else:
                num_filter = nchw_shapes[i+1][1]
                if nchw_shapes[i+1][2] < nchw_shapes[i][2]:
                    stride_h = nchw_shapes[i][2] // nchw_shapes[i+1][2]
                if nchw_shapes[i+1][3] < nchw_shapes[i][3]:
                    stride_w = nchw_shapes[i][3] // nchw_shapes[i+1][3]
            stride = (stride_h, stride_w)
            outshape = (n, num_filter, h // stride_h, w // stride_w)
            print('nchw inshape = {}, outshape = {}, stride = {}'.format(inshape, outshape, stride))
            # Only 3x3 kernel supports strides, not 1x1
            # kernel_shapes = [(1, 1),]
            kernel_shapes = [(3, 3),] if stride_h > 1 or stride_w > 1 else [(1, 1), (3, 3)]
            for kernel_shape in kernel_shapes:
                # padding doesn't make sense for a 1x1 kernel
                pads = [(0, 0),] if kernel_shape[0] == 1 or kernel_shape[1] == 1 else [(0, 0), (1, 1)]
                for pad in pads:
                    act_type = 'relu' if _random_boolean() else None
                    output_stats = _random_boolean()
                    print('    kernel= {}, pad = {}, output_stats={}, act_type = {}'.format(
                        kernel_shape, pad, output_stats, act_type))
                    finalize_norm_conv_test(nchw_inshape=inshape, kernel_shape=kernel_shape,
                                           num_filter=num_filter, act_type=act_type,
                                           stride=stride, pad=pad, output_stats=output_stats,
                                           no_norm=no_norm,
                                           no_conv=False, eps=eps, momentum=momentum)


@with_seed()
def test_normalized_convolution():
    ctx = default_context()
    min_cuda_arch = 70
    max_cuda_arch = 70
    cuda_arch = get_sm_arch(ctx.device_id)
    if cuda_arch < min_cuda_arch or cuda_arch > max_cuda_arch:
        print('Bypassing normalized convolution test on cuda arch {} ({} <= arch <= {}).'.format(
            cuda_arch, min_cuda_arch, max_cuda_arch))
        return

    # RN50 layer shapes
    nchw_shapes = [
        ( 64,  256,  56,  56),
        ( 64,  128,  28,  28),
        ( 64,  512,  28,  28),
        ( 64,  256,  14,  14),
        ( 64, 1024,  14,  14),
        ( 64,  512,   7,   7),
        ( 64, 2048,   7,   7),
        (128,   64,  56,  56),
        (128,  256,  56,  56),
        (128,  128,  28,  28),
        (128,  512,  28,  28),
        (128,  256,  14,  14),
        (128, 1024,  14,  14),
        (128,  512,   7,   7),
        (128, 2048,   7,   7),
    ]

    # Make dataset stats (to input to BNStatsFinalize)
    def create_mean_inv_std(data):
        num_features = data.shape[3]
        num_elems_per_feature = np.prod(data.shape) / num_features
        data_fp32 = data.astype(np.float32)
        not_feature_axes = (0, 1, 2)
        feature_sum = data_fp32.sum(axis=not_feature_axes)
        mean = feature_sum / num_elems_per_feature
        squared_error_sum = np.square(data_fp32 - mean).sum(axis=not_feature_axes)
        variance = squared_error_sum / num_elems_per_feature
        inv_std_dev = 1.0 / np.sqrt(variance)
        return (mean, inv_std_dev)

    # Prepare the input for a standard Convolution so it will mimic NormalizedConvolution
    def normalize_input(data, equiv_scale, equiv_bias, act_type, no_equiv_scale_bias):
        normalized = data if no_equiv_scale_bias else \
                             mx.sym.broadcast_add(mx.sym.broadcast_mul(data, equiv_scale),
                                                  equiv_bias)
        return normalized if act_type is None else mx.sym.Activation(normalized, act_type=act_type)

    # Helper function to the normalized convolution tests
    # Return the indices (along the feature dimension) that have relu inputs near 0.
    def has_near_zero_outputs(x, b, g, threshold):
        X = mx.sym.Variable('X')
        B = mx.sym.Variable('B')  # beta, i.e. bias
        G = mx.sym.Variable('G')  # gamma, i.e. scale
        norm_sym = mx.sym.broadcast_add(mx.sym.broadcast_mul(X, G), B)
        args_dict = {'X':x, 'B':b, 'G':g,}
        grad_req = {'X':'null', 'G':'null', 'B':'null'}
        norm_exe = norm_sym.bind(ctx=ctx, args=args_dict, grad_req=grad_req)
        # Execute forward() graph calculation
        # need is_train=True to keep Batchnorm using the mini-batch mean and variance
        norm_outputs = norm_exe.forward(is_train=True)
        out_data = norm_outputs[0].asnumpy()
        out_data_abs = np.abs(out_data)
        not_feature_axes = (0, 1, 2)
        origin_dist_mins = out_data_abs.min(axis=not_feature_axes)
        bad_indices = np.nonzero(origin_dist_mins < threshold)[0]
        return bad_indices

    # Make dataset stats (to augment standard Convolution) to mimic NormalizedConvolution
    def create_output_stats(data, output_stats):
        if output_stats:
            data_fp32 = mx.sym.cast(data, np.float32)
            not_feature_axes = (0, 1, 2)
            feature_sum = data_fp32.sum(axis=not_feature_axes)
            feature_sum_squares = data_fp32.square().sum(axis=not_feature_axes)
            return mx.sym.Group([data, feature_sum, feature_sum_squares])
        else:
            return data

    # Test fused op without input normalization.  Options for activation and output of stats.
    def convolution_stats_test(nchw_inshape, kernel_shape, num_filter, act_type, stride,
                               pad, output_stats, no_equiv_scale_bias, no_conv):

        # If we are disabling the convolution (no_conv = True), then set kernel=1x1 and weights 1
        if no_conv:
            if kernel_shape != (1,1):
                print('Ignoring kernel_shape {}, forcing 1x1 in no_conv mode.'.format(kernel_shape))
            kernel_shape = (1,1)
        (n, c, h, w) = nchw_inshape
        X = mx.sym.Variable('X')
        W = mx.sym.Variable('W')
        EB = mx.sym.Variable('EB')  # equiv_bias
        ES = mx.sym.Variable('ES')  # equiv_scale
        M = mx.sym.Variable('M')  # mean
        V = mx.sym.Variable('V')  # variance (inv_std_dev actually)
        G = mx.sym.Variable('G')  # gamma (dummy, only needed for backward)
        B = mx.sym.Variable('B')  # beta (dummy, only needed for backward)
        (r, s) = kernel_shape
        layout = 'NHWC'
        conv_args = {'weight':W, 'num_filter':num_filter, 'kernel':kernel_shape,
                       'stride':stride, 'pad':pad, 'layout':layout, 'name':'conv'}

        conv_input = normalize_input(data=X, equiv_scale=ES, equiv_bias=EB, act_type=act_type,
                                     no_equiv_scale_bias=no_equiv_scale_bias)
        conv_sym = mx.sym.Convolution(conv_input, no_bias=True, **conv_args)
        conv_sym = create_output_stats(conv_sym, output_stats)
        if not no_equiv_scale_bias:
            conv_args.update({'equiv_bias':EB, 'equiv_scale':ES, 'mean':M, 'var':V, 'gamma':G, 'beta':B})
        norm_conv_sym = mx.sym.NormalizedConvolution(X, act_type=act_type,
                                                     no_equiv_scale_bias=no_equiv_scale_bias,
                                                     **conv_args)
        if not output_stats:
            # discard sum and sum_squares outputs before binding
            norm_conv_sym = norm_conv_sym[0]

        weight_shape = (num_filter, r, s, c)
        data_shape = (n, h, w, c)
        feature_plane_elements = n * h * w
        x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
        (m_np, v_np) = create_mean_inv_std(x.asnumpy())
        m = mx.nd.array(m_np, dtype=np.float32, ctx=ctx)
        v = mx.nd.array(v_np, dtype=np.float32, ctx=ctx)
        if no_conv:
            w = mx.ndarray.ones(weight_shape, dtype=np.float16, ctx=ctx)
        else:
            w = mx.ndarray.random.uniform(-0.5, 0.5, weight_shape, dtype=np.float16, ctx=ctx)
        equiv_scale_bias_shape = (c,)
        scale_max = 1.25
        bias_max = 1

        # Comparing gradients of two symbols is tricky when a non-smooth function like 'relu'
        # is part of the function.  We ensure that no relu inputs are near 0 (within a threshold)
        # by trying different beta/gamma values as needed.
        eb_np = np.zeros(equiv_scale_bias_shape, dtype=np.float16)
        es_np = np.zeros(equiv_scale_bias_shape, dtype=np.float16)
        indices_to_set = np.array(range(c))
        while len(indices_to_set) > 0:
            for index in indices_to_set:
                eb_np[index] = np.random.uniform(-bias_max, bias_max)
                es_np[index] = np.random.uniform(1.0/scale_max, scale_max)
            eb = mx.nd.array(eb_np, dtype=np.float16, ctx=ctx)
            es = mx.nd.array(es_np, dtype=np.float16, ctx=ctx)
            smallest_norm_fp16 = pow(2, -14)
            threshold = smallest_norm_fp16 / 2
            need_data_check = not no_equiv_scale_bias and act_type == 'relu'
            if need_data_check:
                indices_to_set = has_near_zero_outputs(x, eb, es, threshold=threshold)
            else:
                indices_to_set = []

        dummy_g = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                      dtype=np.float32, ctx=ctx)
        dummy_b = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                      dtype=np.float32, ctx=ctx)
        args_dict = {'X':x, 'W':w} if no_equiv_scale_bias else {'X':x, 'W':w,
                                                                'EB':eb, 'ES':es,
                                                                'M':m, 'V':v,
                                                                'G':dummy_g, 'B':dummy_b}
        conv_exe = conv_sym.bind(ctx=ctx, args=args_dict, grad_req='null')
        norm_conv_exe = norm_conv_sym.bind(ctx=ctx, args=args_dict, grad_req='null')

        conv_outputs = conv_exe.forward(is_train=False)
        # need is_train=True to keep stats from being turned off
        norm_conv_outputs = norm_conv_exe.forward(is_train=output_stats)

        outputs = ['out', 'sum', 'sum_squares']
        # greater atols needs for 'sum' and 'sum_squares', also if input scale/bias is applied
        if no_equiv_scale_bias:
            tols = [(1e-2, 1e-2), (1e-2, 2), (1e-2, 2)]
        else:
            # 'sum' seems to have a large span (e.g. -400K -> +400K) so a large absolute tolerance
            # is needed to cover those cases when the result is near 0 and rtol can't help.
            # One possible source of the large sum tolerance is the internal rounding of the
            # mean to fp16.  Any rounding amount will give a bias to the conv inputs and so the sum.
            # 'sum_squares' doesn't have this issue because rtol handles the always-positive result.
            per_element_atol = 5e-3
            sum_atol = feature_plane_elements * per_element_atol
            tols = [(1e-2, 1e-1), (5e-2, sum_atol), (1e-2, 2)]
        num_outputs = 3 if output_stats else 1
        for idx in range(num_outputs):
            out_name = outputs[idx]
            conv_data = conv_outputs[idx].asnumpy()
            norm_conv_data = norm_conv_outputs[idx].asnumpy()
            (rtol, atol) = tols[idx]
            assert_almost_equal(conv_data, norm_conv_data, rtol=rtol, atol=atol,
                                names=('conv_{}'.format(out_name),
                                       'norm_conv_{}'.format(out_name)))

    # Test input normalization function only: no_equiv_scale_bias = False, 1x1 unity-weights conv
    # Also test with 'relu' activation on and off.
    for i in range(len(nchw_shapes)):
        inshape = nchw_shapes[i]
        (n, c, h, w) = inshape
        num_filter = 32
        outshape = (n, num_filter, h, w)
        stride = (1,1)
        print('nchw inshape = {}, outshape = {}, stride = {}'.format(inshape, outshape, stride))
        kernel_shape = (1, 1)
        pad = (0, 0)
        output_stats = False
        act_type = 'relu' if _random_boolean() else None
        print('    kernel= {}, pad = {}, output_stats={}, act_type = {}'.format(
            kernel_shape, pad, output_stats, act_type))
        convolution_stats_test(nchw_inshape=inshape, kernel_shape=kernel_shape,
                               num_filter=num_filter, act_type=act_type,
                               stride=stride, pad=pad, output_stats=output_stats,
                               no_equiv_scale_bias=False, no_conv=True)

    # Test convolution and stats-gen functions, first without, then with, input normalization.
    # Also test with 'relu' activation on and off.
    for no_equiv_scale_bias in [True, False]:
        if no_equiv_scale_bias:
            print('\nTest of convolution function, without input normalization.')
        else:
            print('\nTest of convolution function with input normalization.')
        for i in range(len(nchw_shapes)):
            inshape = nchw_shapes[i]
            (n, c, h, w) = inshape
            (stride_h, stride_w) = (1,1)
            # Leverage next test case (if available) to determine outshape, strides
            if i == len(nchw_shapes)-1:
                num_filter = nchw_shapes[i][1]
            else:
                num_filter = nchw_shapes[i+1][1]
                if nchw_shapes[i+1][2] < nchw_shapes[i][2]:
                    stride_h = nchw_shapes[i][2] // nchw_shapes[i+1][2]
                if nchw_shapes[i+1][3] < nchw_shapes[i][3]:
                    stride_w = nchw_shapes[i][3] // nchw_shapes[i+1][3]
            stride = (stride_h, stride_w)
            outshape = (n, num_filter, h // stride_h, w // stride_w)
            print('nchw inshape = {}, outshape = {}, stride = {}'.format(inshape, outshape, stride))
            # Only 3x3 kernel supports strides, not 1x1
            kernel_shapes = [(3, 3),] if stride_h > 1 or stride_w > 1 else [(1, 1), (3, 3)]
            for kernel_shape in kernel_shapes:
                # padding doesn't make sense for a 1x1 kernel
                pads = [(0, 0),] if kernel_shape[0] == 1 or kernel_shape[1] == 1 else [(0, 0), (1, 1)]
                for pad in pads:
                    act_type = 'relu' if _random_boolean() else None
                    output_stats = _random_boolean()
                    print('    kernel= {}, pad = {}, output_stats={}, act_type = {}'.format(
                        kernel_shape, pad, output_stats, act_type))
                    convolution_stats_test(nchw_inshape=inshape, kernel_shape=kernel_shape,
                                          num_filter=num_filter, act_type=act_type,
                                          stride=stride, pad=pad, output_stats=output_stats,
                                          no_equiv_scale_bias=no_equiv_scale_bias, no_conv=False)


@with_seed()
def test_finalize_with_normalized_convolution():
    ctx = default_context()
    min_cuda_arch = 70
    max_cuda_arch = 70
    cuda_arch = get_sm_arch(ctx.device_id)
    if cuda_arch < min_cuda_arch or cuda_arch > max_cuda_arch:
        print('Bypassing normalized convolution test on cuda arch {} ({} <= arch <= {}).'.format(
            cuda_arch, min_cuda_arch, max_cuda_arch))
        return

    # RN50 layer shapes
    nchw_shapes = [
        ( 64,  256,  56,  56),
        ( 64,  128,  28,  28),
        ( 64,  512,  28,  28),
        ( 64,  256,  14,  14),
        ( 64, 1024,  14,  14),
        ( 64,  512,   7,   7),
        ( 64, 2048,   7,   7),
        (128,   64,  56,  56),
        (128,  256,  56,  56),
        (128,  128,  28,  28),
        (128,  512,  28,  28),
        (128,  256,  14,  14),
        (128, 1024,  14,  14),
        (128,  512,   7,   7),
        (128, 2048,   7,   7),
    ]

    # Make dataset stats (to input to BNStatsFinalize)
    def create_input_stats_np(data_np):
        data_fp32_np = data_np.astype(np.float32)
        not_feature_axes = (0, 1, 2)
        feature_sum_np = data_fp32_np.sum(axis=not_feature_axes)
        feature_sum_squares_np = np.square(data_fp32_np).sum(axis=not_feature_axes)
        return (feature_sum_np, feature_sum_squares_np)

    def create_output_stats(data, output_stats):
        if output_stats:
            data_fp32 = mx.sym.cast(data, np.float32)
            not_feature_axes = (0, 1, 2)
            feature_sum = data_fp32.sum(axis=not_feature_axes)
            feature_sum_squares = data_fp32.square().sum(axis=not_feature_axes)
            return mx.sym.Group([data, feature_sum, feature_sum_squares])
        else:
            return data

    def out_shape(nhwc_inshape, num_filters, kernel_shape, stride, pad):
        (n, h, w, _) = nhwc_inshape
        (kernel_h, kernel_w) = kernel_shape
        (stride_h, stride_w) = stride
        (pad_h, pad_w) = pad
        out_shape_h = 1 + ((h + 2 * pad_h - kernel_h) // stride_h)
        out_shape_w = 1 + ((w + 2 * pad_w - kernel_w) // stride_w)
        return (n, out_shape_h, out_shape_w, num_filters)

    # flip a dataset about the 1st dimension
    def flip(data):
        return mx.sym.flip(data, axis=0)

    # return a new symbol that isolates the input symbol's outputs
    def buffer(sym):
        num_outputs = len(sym.list_outputs())
        if num_outputs == 1:
            return flip(flip(sym))
        else:
            flipped_outputs = [ flip(flip(sym[i])) for i in range(num_outputs)]
            return mx.sym.Group(flipped_outputs)

    # Test fused op without input normalization.  Options for activation and output of stats.
    def finalize_norm_conv_test(nchw_inshape, kernel_shape, num_filter, act_type, stride,
                               pad, output_stats, no_equiv_scale_bias, no_conv,
                               eps, momentum):

        # If we are disabling the convolution (no_conv = True), then set kernel=1x1 and weights 1
        if no_conv:
            if kernel_shape != (1,1):
                print('Ignoring kernel_shape {}, forcing 1x1 in no_conv mode.'.format(kernel_shape))
            kernel_shape = (1,1)
        (n, c, h, w) = nchw_inshape
        X = mx.sym.Variable('X')
        W = mx.sym.Variable('W')
        SUM = mx.sym.Variable('SUM')
        SUMSQ = mx.sym.Variable('SUMSQ')
        B = mx.sym.Variable('B')  # beta, i.e. bias
        G = mx.sym.Variable('G')  # gamma, i.e. scale
        # randomly insert buffering here to exercise in-place vs. copy of gamma/beta by Finalize
        if _random_boolean():
            B = buffer(B)
            G = buffer(G)
        MovMean = mx.sym.Variable('MovMean')
        MovVar = mx.sym.Variable('MovVar')

        # make 'ground truth' symbol using standard Batchnorm and Convolution

        if no_equiv_scale_bias and (act_type is None):
            normalized = X
        elif no_equiv_scale_bias:
            normalized = mx.sym.Activation(data=X, act_type=act_type)
        else:
            normalized = mx.sym.BatchNorm(data=X,  gamma=G, beta=B, act_type=act_type,
                                      moving_mean=MovMean, moving_var=MovVar,
                                      eps=eps, momentum=momentum, fix_gamma=False,
                                      use_global_stats=False, output_mean_var=False,
                                      cudnn_off=False, name=None, axis=-1)
        (r, s) = kernel_shape
        layout = 'NHWC'
        conv_args = {'weight':W, 'num_filter':num_filter, 'kernel':kernel_shape,
                     'stride':stride, 'pad':pad, 'layout':layout, 'name':'conv'}
        conv_sym = mx.sym.Convolution(data=normalized, no_bias=True, **conv_args)
        conv_sym = create_output_stats(conv_sym, output_stats)

        # make symbol-under-test using Finalize and NormalizedConvolution

        if not no_equiv_scale_bias:
            elem_count = np.prod(nchw_inshape) // c
            (equiv_scale, equiv_bias, saved_mean, saved_inv_std, gamma_out, beta_out) = \
                mx.sym.BNStatsFinalize(sum=SUM, sum_squares=SUMSQ, gamma=G, beta=B,
                                       moving_mean=MovMean, moving_var=MovVar, eps=eps,
                                       momentum=momentum, fix_gamma=False,
                                       output_mean_var=True, elem_count=elem_count)
            # NormalizedConvolution makes use of conv_args but for stats-apply mode has more inputs:
            conv_args.update({'equiv_scale':equiv_scale, 'equiv_bias':equiv_bias,
                              'mean':saved_mean, 'var':saved_inv_std,
                              'gamma':gamma_out, 'beta':beta_out})
        norm_conv_sym = mx.sym.NormalizedConvolution(X, act_type=act_type,
                                                     no_equiv_scale_bias=no_equiv_scale_bias,
                                                     **conv_args)
        if not output_stats:
            # discard sum and sum_squares outputs before binding
            norm_conv_sym = norm_conv_sym[0]

        # make data inputs
        weight_shape = (num_filter, r, s, c)
        data_shape = (n, h, w, c)
        # x_np = np.fromfunction(lambda n, h, w, c: 3*((n+h+w)%2), data_shape)
        # x = mx.nd.array(x_np, dtype=np.float16, ctx=ctx)
        x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
        (feature_sum_np, feature_sum_squares_np) = create_input_stats_np(x.asnumpy())
        sum = mx.nd.array(feature_sum_np, dtype=np.float32)
        sum_squares = mx.nd.array(feature_sum_squares_np, dtype=np.float32)
        equiv_scale_bias_shape = (c,)
        scale_max = 1.25
        bias_max = 1

        # Comparing gradients of two symbols is tricky when a non-smooth function like 'relu'
        # is part of the function.  We ensure that no relu inputs are near 0 (within a threshold)
        # by trying different beta/gamma values as needed.
        b_np = np.zeros(equiv_scale_bias_shape, dtype=np.float32)
        g_np = np.zeros(equiv_scale_bias_shape, dtype=np.float32)
        indices_to_set = np.array(range(c))
        while len(indices_to_set) > 0:
            for index in indices_to_set:
                b_np[index] = np.random.uniform(-bias_max, bias_max)
                g_np[index] = np.random.uniform(1.0/scale_max, scale_max)
            b = mx.nd.array(b_np, dtype=np.float32, ctx=ctx)
            g = mx.nd.array(g_np, dtype=np.float32, ctx=ctx)
            smallest_norm_fp16 = pow(2, -14)
            threshold = smallest_norm_fp16 / 2
            need_data_check = not no_equiv_scale_bias and act_type == 'relu'
            if need_data_check:
                indices_to_set = _has_near_zero_outputs(x, b, g, eps, threshold=threshold)
            else:
                indices_to_set = []

        # mov_mean_np = np.zeros(equiv_scale_bias_shape).astype(np.float32)
        # mov_var_np = np.ones(equiv_scale_bias_shape).astype(np.float32)
        mov_mean_np = np.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape)
        mov_var_np = np.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape)
        # since the models change the moving mean and variance, each model gets their own copy
        mov_mean1 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_mean2 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_var1 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        mov_var2 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        if no_conv:
            weights = mx.ndarray.ones(weight_shape, dtype=np.float16, ctx=ctx)
        else:
            weights = mx.ndarray.random.uniform(-0.20, 0.20, weight_shape, dtype=np.float16, ctx=ctx)
        # These are the tensor's that receive the backpropped gradients (so an output of backward())
        # Copy 1 is for 'ground truth' symbol based on BatchNorm/Convolution ops
        d_x_out_gt = mx.ndarray.zeros(data_shape, dtype=np.float16, ctx=ctx)
        d_w_out_gt = mx.ndarray.zeros(weight_shape, dtype=np.float16, ctx=ctx)
        d_gamma_out_gt = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
        d_beta_out_gt = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
        # Copy 2 is for symbol based on BNStatsFinalize/NormalizedConvolution ops (=ones, not zeros)
        d_x_out = mx.ndarray.ones(data_shape, dtype=np.float16, ctx=ctx)
        d_w_out = mx.ndarray.ones(weight_shape, dtype=np.float16, ctx=ctx)
        d_gamma_out = mx.ndarray.ones(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
        d_beta_out = mx.ndarray.ones(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)

        # bind i/o's to symbols to create executors

        grad_req = {'SUM':'null', 'SUMSQ':'null', 'MovMean':'null', 'MovVar':'null',
                    'X':'write', 'W':'write', 'G':'write', 'B':'write'}

        args_grad_dict_gt = {'X':d_x_out_gt, 'W':d_w_out_gt, 'G':d_gamma_out_gt, 'B':d_beta_out_gt}
        args_grad_dict = {'X':d_x_out, 'W':d_w_out, 'G':d_gamma_out, 'B':d_beta_out}

        args_dict = {'X':x, 'W':weights}
        # conv binding does not need SUM, and SUMSQ, but extra items are OK
        if not no_equiv_scale_bias:
            args_dict.update({'B':b, 'G':g, 'SUM':sum, 'SUMSQ':sum_squares})
        gt_aux_states_dict = \
            {'MovMean':mov_mean1, 'MovVar':mov_var1}
        finalize_aux_states_dict = \
            {'MovMean':mov_mean2, 'MovVar':mov_var2}

        conv_exe = conv_sym.bind(ctx=ctx, args=args_dict, args_grad=args_grad_dict_gt,
                                 aux_states=gt_aux_states_dict, grad_req=grad_req)
        norm_conv_exe = norm_conv_sym.bind(ctx=ctx, args=args_dict, args_grad=args_grad_dict,
                                           aux_states=finalize_aux_states_dict, grad_req=grad_req)

        # Execute forward() graph calculation
        # need is_train=True to keep Batchnorm using the mini-batch mean and variance
        conv_outputs = conv_exe.forward(is_train=True)
        # need is_train=True to keep stats from being turned off
        norm_conv_outputs = norm_conv_exe.forward(is_train=True)

        # Check forward outputs
        outputs = ['out', 'sum', 'sum_squares']
        # greater atols needs for 'sum' and 'sum_squares', also if input scale/bias is applied
        if no_equiv_scale_bias:
            tols = [(1e-2, 1e-2), (1e-2, 2), (1e-2, 2)]
        else:
            # 'sum' seems to have a large span (e.g. -400K -> +400K) so a large absolute tolerance
            # is needed to cover those cases when the result is near 0 and rtol can't help.
            # One possible source of the large sum tolerance is the internal rounding of the
            # mean to fp16.  Any rounding amount will give a bias to the conv inputs and so the sum.
            # 'sum_squares' doesn't have this issue because rtol handles the always-positive result.
            per_element_atol = 5e-3
            sum_atol = n * h * w * per_element_atol
            tols = [(1e-2, 1e-1), (1e-1, sum_atol), (1e-2, 2)]
        num_outputs = 3 if output_stats else 1
        for idx in range(num_outputs):
            out_name = outputs[idx]
            conv_data = conv_outputs[idx].asnumpy()
            norm_conv_data = norm_conv_outputs[idx].asnumpy()
            (rtol, atol) = tols[idx]
            assert_almost_equal(conv_data, norm_conv_data, rtol=rtol, atol=atol,
                                names=('conv_{}'.format(out_name),
                                       'norm_conv_{}'.format(out_name)))
        # Check backward function
        if no_equiv_scale_bias and act_type is not None:
            # gradient calculation not supported for this configuration
            return
        # Create backward gradients
        outshape = out_shape(data_shape, num_filter, kernel_shape, stride, pad)
        d_out_in = mx.ndarray.random.uniform(-0.2, 0.2, outshape,
                                                     dtype=np.float16, ctx=ctx)
        # not really needed
        sum_shape = (num_filter,)
        # gradients on these outputs will be summed into the d_out_in for the ground truth
        # symbol, so make sure these are 0.
        d_sum_in = mx.ndarray.zeros(sum_shape, dtype=np.float32, ctx=ctx)
        d_sum_squares_in = mx.ndarray.zeros(sum_shape, dtype=np.float32, ctx=ctx)
        # d_sum_in = mx.ndarray.random.uniform(0.0, 1.0, sum_shape,
        #                                              dtype=np.float32, ctx=ctx)
        # d_sum_squares_in = mx.ndarray.random.uniform(0.0, 1.0, sum_shape,
        #                                              dtype=np.float32, ctx=ctx)
        # Execute backward() graph calculation
        if output_stats:
            conv_outputs = conv_exe.backward([d_out_in, d_sum_in, d_sum_squares_in])
            norm_conv_outputs = norm_conv_exe.backward([d_out_in, d_sum_in, d_sum_squares_in])
        else:
            conv_outputs = conv_exe.backward([d_out_in,])
            norm_conv_outputs = norm_conv_exe.backward([d_out_in,])

        # Check weight gradient
        out_name = 'd_w'
        assert_almost_equal(d_w_out_gt.asnumpy(), d_w_out.asnumpy(), atol=0.3, rtol=0.1,
                            names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))
        # Check data gradient
        # This check is flakey when act_type = relu because if the two models differ on whether
        # the normalized value is above or below 0, then the gradient may or may-not be backpropped.

        # To fix this test, we could run a separate model with relu off, capture the normalized
        # output and then mask off the gradient comparison when the normalized value is near 0.
        out_name = 'd_x'
        if act_type is None:
            assert_almost_equal(d_x_out_gt.asnumpy(), d_x_out.asnumpy(), atol=0.1, rtol=0.1,
                                names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))
        # Check gamma and beta gradients
        out_name = 'd_gamma'
        assert_almost_equal(d_gamma_out_gt.asnumpy(), d_gamma_out.asnumpy(), atol=10, rtol=0.1,
                            names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))
        out_name = 'd_beta'
        assert_almost_equal(d_beta_out_gt.asnumpy(), d_beta_out.asnumpy(), atol=10, rtol=0.1,
                            names=('conv_{}'.format(out_name), 'norm_conv_{}'.format(out_name)))

    # Test input normalization function only: no_equiv_scale_bias = False, 1x1 unity-weights conv
    # Also test with 'relu' activation on and off.
    print('\nTest of input normalization without convolution function.')
    eps = 1e-4
    momentum = 0.9
    for i in range(len(nchw_shapes)):
        inshape = nchw_shapes[i]
        (n, c, h, w) = inshape
        num_filter = 32
        outshape = (n, num_filter, h, w)
        stride = (1,1)
        print('nchw inshape = {}, outshape = {}, stride = {}'.format(inshape, outshape, stride))
        kernel_shape = (1, 1)
        pad = (0, 0)
        output_stats = False
        act_type = 'relu' if _random_boolean() else None
        print('    kernel= {}, pad = {}, output_stats={}, act_type = {}'.format(
            kernel_shape, pad, output_stats, act_type))
        finalize_norm_conv_test(nchw_inshape=inshape, kernel_shape=kernel_shape,
                                num_filter=num_filter, act_type=act_type,
                                stride=stride, pad=pad, output_stats=output_stats,
                                no_equiv_scale_bias=False,
                                no_conv=True, eps=eps, momentum=momentum)

    # Test convolution and stats-gen functions, first without, then with, input normalization.
    # Also test with 'relu' activation on and off.
    # for no_equiv_scale_bias in [False, True]:
    for no_equiv_scale_bias in [True, False]:
        if no_equiv_scale_bias:
            print('\nTest of convolution function, without input normalization.')
        else:
            print('\nTest of convolution function with input normalization.')
        for i in range(len(nchw_shapes)):
            inshape = nchw_shapes[i]
            (n, c, h, w) = inshape
            (stride_h, stride_w) = (1,1)
            # Leverage next test case (if available) to determine outshape, strides
            if i == len(nchw_shapes)-1:
                num_filter = nchw_shapes[i][1]
            else:
                num_filter = nchw_shapes[i+1][1]
                if nchw_shapes[i+1][2] < nchw_shapes[i][2]:
                    stride_h = nchw_shapes[i][2] // nchw_shapes[i+1][2]
                if nchw_shapes[i+1][3] < nchw_shapes[i][3]:
                    stride_w = nchw_shapes[i][3] // nchw_shapes[i+1][3]
            stride = (stride_h, stride_w)
            outshape = (n, num_filter, h // stride_h, w // stride_w)
            print('nchw inshape = {}, outshape = {}, stride = {}'.format(inshape, outshape, stride))
            # Only 3x3 kernel supports strides, not 1x1
            # kernel_shapes = [(1, 1),]
            kernel_shapes = [(3, 3),] if stride_h > 1 or stride_w > 1 else [(1, 1), (3, 3)]
            for kernel_shape in kernel_shapes:
                # padding doesn't make sense for a 1x1 kernel
                pads = [(0, 0),] if kernel_shape[0] == 1 or kernel_shape[1] == 1 else [(0, 0), (1, 1)]
                for pad in pads:
                    act_type = 'relu' if _random_boolean() else None
                    output_stats = _random_boolean()
                    print('    kernel= {}, pad = {}, output_stats={}, act_type = {}'.format(
                        kernel_shape, pad, output_stats, act_type))
                    finalize_norm_conv_test(nchw_inshape=inshape, kernel_shape=kernel_shape,
                                           num_filter=num_filter, act_type=act_type,
                                           stride=stride, pad=pad, output_stats=output_stats,
                                           no_equiv_scale_bias=no_equiv_scale_bias,
                                           no_conv=False, eps=eps, momentum=momentum)


@with_seed()
def test_bn_stats_finalize():
    ctx = default_context()
    min_cuda_arch = 70
    cuda_arch = get_sm_arch(ctx.device_id)
    if cuda_arch < min_cuda_arch:
        print('Bypassing bn_stats_finalize test on cuda arch {} (need >= {}).'.format(
            cuda_arch, min_cuda_arch))
        return

    nchw_shapes = [
        # n*h*w == 2 included to flush out 'sample' vs. 'population' variance issues
        ( 1,  8,  1,  2),
        # RN50 layer shapes
        ( 64,  256,  56,  56),
        ( 64,  128,  28,  28),
        ( 64,  512,  28,  28),
        ( 64,  256,  14,  14),
        ( 64, 1024,  14,  14),
        ( 64,  512,   7,   7),
        ( 64, 2048,   7,   7),
        (128,   64,  56,  56),
        (128,  256,  56,  56),
        (128,  128,  28,  28),
        (128,  512,  28,  28),
        (128,  256,  14,  14),
        (128, 1024,  14,  14),
        (128,  512,   7,   7),
        (128, 2048,   7,   7),
    ]

    # Prepare the input for a standard Convolution so it will mimic NormalizedConvolution
    def normalize_input(data, equiv_scale, equiv_bias, act_type, no_equiv_scale_bias):
        normalized = data if no_equiv_scale_bias else \
                             mx.sym.broadcast_add(mx.sym.broadcast_mul(data, equiv_scale),
                                                  equiv_bias)
        return normalized if act_type is None else mx.sym.Activation(normalized, act_type=act_type)

    # Make dataset stats (to input to BNStatsFinalize)
    def create_output_stats(data):
        data_fp32 = mx.sym.cast(data, np.float32)
        not_feature_axes = (0, 1, 2)
        feature_sum = data_fp32.sum(axis=not_feature_axes)
        feature_sum_squares = data_fp32.square().sum(axis=not_feature_axes)
        return (feature_sum, feature_sum_squares)

    # flip a dataset about the 1st dimension
    def flip(data):
        return mx.sym.flip(data, axis=0)

    # return a new symbol that isolates the input symbol's outputs
    def buffer(sym):
        num_outputs = len(sym.list_outputs())
        if num_outputs == 1:
            return flip(flip(sym))
        else:
            flipped_outputs = [ flip(flip(sym[i])) for i in range(num_outputs)]
            return mx.sym.Group(flipped_outputs)

    # Test of BNStatsFinalize op against a 'ground truth' of Batchnorm and home-grown functions.
    def bn_stats_finalize_test(nchw_inshape, eps, momentum, is_train, test_writeinplace):

        (n, c, h, w) = nchw_inshape
        elem_count = np.prod(nchw_inshape) // c
        X = mx.sym.Variable('X')
        G = mx.sym.Variable('G')  # gamma, i.e. scale
        B = mx.sym.Variable('B')  # beta, i.e. bias
        if (test_writeinplace):
            G = buffer(G)
            B = buffer(B)
        MovMean = mx.sym.Variable('MovMean')
        MovVar = mx.sym.Variable('MovVar')

        # Make ground truth (i.e. 'gt') model using conventional cudnn Batchnorm, which processes
        # the running mean using the 'sample variance' with N = elem_count - 1.  To avoid use of
        # the NHWCBatchnorm, which uses 'population variance', we transpose around the Batchnorm op.

        # The input data 'X' starts in 'NHWC'.

        # For NHWC -> NCHW, axes=(0,3,1,2)
        transposed = mx.sym.transpose(data=X, axes=(0,3,1,2))
        (data, saved_mean, saved_inv_std) = mx.sym.BatchNorm(data=transposed,  gamma=G, beta=B,
                                       moving_mean=MovMean, moving_var=MovVar,
                                       eps=eps, momentum=momentum, fix_gamma=False,
                                       use_global_stats=False, output_mean_var=True,
                                       cudnn_off=False, name=None, axis=1)
        # For NCHW -> NHWC axes=(0,2,3,1)
        data = mx.sym.transpose(data=data, axes=(0,2,3,1))

        equiv_scale_inf_fp32 = G / mx.sym.sqrt(MovVar + eps)
        equiv_scale_inf = mx.sym.cast(equiv_scale_inf_fp32, dtype=np.float16)
        equiv_bias_inf_fp32 = B - G * MovMean / mx.sym.sqrt(MovVar + eps)
        equiv_bias_inf = mx.sym.cast(equiv_bias_inf_fp32, dtype=np.float16)
        (sum, sum_squares) = create_output_stats(X)
        batch_mean_fp32 = sum / elem_count
        batch_variance_fp32 = sum_squares / elem_count - mx.sym.square(batch_mean_fp32)
        equiv_scale_train_fp32 = G / mx.sym.sqrt(batch_variance_fp32 + eps)
        equiv_scale_train = mx.sym.cast(equiv_scale_train_fp32, dtype=np.float16)
        equiv_bias_train_fp32 = B - G * batch_mean_fp32 / mx.sym.sqrt(batch_variance_fp32 + eps)
        equiv_bias_train = mx.sym.cast(equiv_bias_train_fp32, dtype=np.float16)
        # Leave bn data as part of symbol output in case operator doesn't like req[kOut]==kNullOp
        if is_train:
            gt_sym = mx.sym.Group([equiv_scale_train, equiv_bias_train,
                                   saved_mean, saved_inv_std, data])
        else:
            gt_sym = mx.sym.Group([equiv_scale_inf, equiv_bias_inf, data])

        # Make BNStatsFinalize model, uses sum and sum_squares created above based on the data

        finalize_sym = mx.sym.BNStatsFinalize(sum=sum, sum_squares=sum_squares, gamma=G, beta=B,
                                              moving_mean=MovMean, moving_var=MovVar, eps=eps,
                                              momentum=momentum, fix_gamma=False,
                                              output_mean_var=is_train, elem_count=elem_count)
        if (test_writeinplace):
            finalize_sym = buffer(finalize_sym)

        data_shape = (n, h, w, c)
        x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
        equiv_scale_bias_shape = (c,)
        scale_max = 1.25
        bias_max = 1
        b = mx.ndarray.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape,
                                      dtype=np.float32, ctx=ctx)
        g = mx.ndarray.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape,
                                      dtype=np.float32, ctx=ctx)
        mov_mean_np = np.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape)
        mov_var_np = np.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape)
        # since the models change the moving mean and variance, each model gets their own copy
        mov_mean1 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_mean2 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_var1 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        mov_var2 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        args_dict = {'X':x, 'B':b, 'G':g}
        gt_aux_states_dict =\
            {'MovMean':mov_mean1, 'MovVar':mov_var1}
        finalize_aux_states_dict =\
            {'MovMean':mov_mean2, 'MovVar':mov_var2}
        gt_exe = gt_sym.bind(ctx=ctx, args=args_dict,
                             aux_states=gt_aux_states_dict, grad_req='null')
        finalize_exe = finalize_sym.bind(ctx=ctx, args=args_dict,
                                         aux_states=finalize_aux_states_dict, grad_req='null')

        gt_outputs = gt_exe.forward(is_train=is_train)
        finalize_outputs = finalize_exe.forward(is_train=is_train)

        outputs = ['equiv_scale', 'equiv_bias', 'saved_mean', 'saved_var']
        tols = [(1e-2, 1e-2), (1e-2, 1e-2), (1e-2, 1e-2), (1e-2, 1e-2)]
        num_outputs = 4 if is_train else 2
        for idx in range(num_outputs):
            out_name = outputs[idx]
            gt_data = gt_outputs[idx].asnumpy()
            finalize_data = finalize_outputs[idx].asnumpy()
            (rtol, atol) = tols[idx]
            assert_almost_equal(gt_data, finalize_data, rtol=rtol, atol=atol,
                                names=('gt_{}'.format(out_name),
                                       'finalize_{}'.format(out_name)))
        if is_train:
            for aux_name in ['MovMean', 'MovVar']:
                gt_data = gt_exe.aux_dict[aux_name].asnumpy()
                finalize_data = finalize_exe.aux_dict[aux_name].asnumpy()
                assert_almost_equal(gt_data, finalize_data, rtol=rtol, atol=atol,
                                    names=('gt_{}'.format(aux_name),
                                           'finalize_{}'.format(aux_name)))
            # Also test finalize ability to propagate beta and gamma
            # gamma is output index 4
            gamma_out = finalize_outputs[4].asnumpy()
            assert_almost_equal(gamma_out, g.asnumpy(), atol=0.0, rtol=0.0)
            # gamma is output index 5
            beta_out = finalize_outputs[5].asnumpy()
            assert_almost_equal(beta_out, b.asnumpy(), atol=0.0, rtol=0.0)


        # Now test BNStatsFinalize ability to backprop gradient in a training graph
        if is_train:
            S = mx.sym.Variable('S')    # sum
            SS = mx.sym.Variable('SS')  # sum_squares
            finalize_sym = mx.sym.BNStatsFinalize(sum=S, sum_squares=SS, gamma=G, beta=B,
                                                  moving_mean=MovMean, moving_var=MovVar, eps=eps,
                                                  momentum=momentum, fix_gamma=False,
                                                  output_mean_var=is_train, elem_count=elem_count)
            s = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
            ss = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
            b = mx.ndarray.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape,
                                          dtype=np.float32, ctx=ctx)
            g = mx.ndarray.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape,
                                          dtype=np.float32, ctx=ctx)
            args_dict = {'S':s, 'SS':ss, 'B':b, 'G':g}
            grad_req = {'S':'null', 'SS':'null', 'G':'write', 'B':'write'}
            d_gamma_out = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
            d_beta_out = mx.ndarray.zeros(equiv_scale_bias_shape, dtype=np.float32, ctx=ctx)
            args_grad_dict = {'G':d_gamma_out, 'B':d_beta_out}
            if (test_writeinplace):
                finalize_sym = buffer(finalize_sym)
            finalize_exe = finalize_sym.bind(ctx=ctx, args=args_dict, args_grad=args_grad_dict,
                                         aux_states=finalize_aux_states_dict, grad_req=grad_req)
            finalize_exe.forward(is_train=is_train)
            d_equiv_scale_in = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                          dtype=np.float16, ctx=ctx)
            d_equiv_bias_in = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                          dtype=np.float16, ctx=ctx)
            d_mean_in = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                          dtype=np.float32, ctx=ctx)
            d_inv_stddev_in = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                          dtype=np.float32, ctx=ctx)
            d_gamma_in = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                          dtype=np.float32, ctx=ctx)
            d_beta_in = mx.ndarray.random.uniform(0.0, 1.0, equiv_scale_bias_shape,
                                          dtype=np.float32, ctx=ctx)
            finalize_exe.backward([d_equiv_scale_in, d_equiv_bias_in,
                                   d_mean_in, d_inv_stddev_in, d_gamma_in, d_beta_in])
            assert_almost_equal(d_gamma_in.asnumpy(), d_gamma_out.asnumpy(), atol=0.0, rtol=0.0)
            assert_almost_equal(d_beta_in.asnumpy(), d_beta_out.asnumpy(), atol=0.0, rtol=0.0)

    # Test BNStatsFinalize op in both inference and training modes
    for is_train in [False, True]:
        for test_writeinplace in [False, True]:
            # writeinplace test only relevant for training graphs
            if not is_train and test_writeinplace:
                continue
            for i in range(len(nchw_shapes)):
                inshape = nchw_shapes[i]
                eps = 1e-4
                momentum = 0.9
                bn_stats_finalize_test(inshape, eps, momentum, is_train, test_writeinplace)

@with_seed()
def test_norm_convolution_finalize():
    ctx = default_context()
    min_cuda_arch = 70
    max_cuda_arch = 70
    cuda_arch = get_sm_arch(ctx.device_id)
    if cuda_arch < min_cuda_arch or cuda_arch > max_cuda_arch:
        print('Bypassing normalized convolution test on cuda arch {} ({} <= arch <= {}).'.format(
            cuda_arch, min_cuda_arch, max_cuda_arch))
        return

    nchw_shapes = [
        # n*h*w == 2 included to flush out 'sample' vs. 'population' variance issues
        ( 1,  32,  1,  2),
        # RN50 layer shapes
        ( 64,  256,  56,  56),
        ( 64,  128,  28,  28),
        ( 64,  512,  28,  28),
        ( 64,  256,  14,  14),
        ( 64, 1024,  14,  14),
        ( 64,  512,   7,   7),
        ( 64, 2048,   7,   7),
        (128,   64,  56,  56),
        (128,  256,  56,  56),
        (128,  128,  28,  28),
        (128,  512,  28,  28),
        (128,  256,  14,  14),
        (128, 1024,  14,  14),
        (128,  512,   7,   7),
        (128, 2048,   7,   7),
    ]

    # Make dataset stats (to input to BNStatsFinalize)
    def create_output_stats(data):
        data_fp32 = mx.sym.cast(data, np.float32)
        not_feature_axes = (0, 1, 2)
        feature_sum = data_fp32.sum(axis=not_feature_axes)
        feature_sum_squares = data_fp32.square().sum(axis=not_feature_axes)
        return (feature_sum, feature_sum_squares)

    # Test of BNStatsFinalize op against a 'ground truth' of Batchnorm and home-grown functions.
    def bn_stats_finalize_test(nchw_inshape, eps, momentum, is_train):

        (n, c, h, w) = nchw_inshape
        elem_count = np.prod(nchw_inshape) // c
        X = mx.sym.Variable('X')
        G = mx.sym.Variable('G')  # gamma, i.e. scale
        B = mx.sym.Variable('B')  # beta, i.e. bias
        W = mx.sym.Variable('W')  # weight, dummy value to keep NormConvolution happy
        MovMean = mx.sym.Variable('MovMean')
        MovVar = mx.sym.Variable('MovVar')

        # Make ground truth (i.e. 'gt') model using conventional cudnn Batchnorm, which processes
        # the running mean using the 'sample variance' with N = elem_count - 1.  To avoid use of
        # the NHWCBatchnorm, which uses 'population variance', we transpose around the Batchnorm op.

        # The input data 'X' starts in 'NHWC'.

        # For NHWC -> NCHW, axes=(0,3,1,2)
        transposed = mx.sym.transpose(data=X, axes=(0,3,1,2))
        (data, saved_mean, saved_inv_std) = mx.sym.BatchNorm(data=transposed,  gamma=G, beta=B,
                                       moving_mean=MovMean, moving_var=MovVar,
                                       eps=eps, momentum=momentum, fix_gamma=False,
                                       use_global_stats=False, output_mean_var=True,
                                       cudnn_off=False, name=None, axis=1)
        # For NCHW -> NHWC axes=(0,2,3,1)
        data = mx.sym.transpose(data=data, axes=(0,2,3,1))

        equiv_scale_inf_fp32 = G / mx.sym.sqrt(MovVar + eps)
        equiv_scale_inf = mx.sym.cast(equiv_scale_inf_fp32, dtype=np.float16)
        equiv_bias_inf_fp32 = B - G * MovMean / mx.sym.sqrt(MovVar + eps)
        equiv_bias_inf = mx.sym.cast(equiv_bias_inf_fp32, dtype=np.float16)
        (sum, sum_squares) = create_output_stats(X)
        batch_mean_fp32 = sum / elem_count
        batch_variance_fp32 = sum_squares / elem_count - mx.sym.square(batch_mean_fp32)
        equiv_scale_train_fp32 = G / mx.sym.sqrt(batch_variance_fp32 + eps)
        equiv_scale_train = mx.sym.cast(equiv_scale_train_fp32, dtype=np.float16)
        equiv_bias_train_fp32 = B - G * batch_mean_fp32 / mx.sym.sqrt(batch_variance_fp32 + eps)
        equiv_bias_train = mx.sym.cast(equiv_bias_train_fp32, dtype=np.float16)
        # Leave bn data as part of symbol output in case operator doesn't like req[kOut]==kNullOp
        if is_train:
            gt_sym = mx.sym.Group([saved_mean, saved_inv_std,
                                   equiv_scale_train, equiv_bias_train, data])
        else:
            gt_sym = mx.sym.Group([equiv_scale_inf, equiv_bias_inf, data])

        # Make BNStatsFinalize model, uses sum and sum_squares created above based on the data
        num_filter = 32
        (data, _, _, saved_mean, saved_inv_std, equiv_scale, equiv_bias) = \
             mx.sym.NormConvolution(data=X, weight=W, kernel=(1,1), num_filter=num_filter,
                                    in_sum=sum, in_sum_squares=sum_squares, gamma=G, beta=B,
                                    moving_mean=MovMean, moving_var=MovVar, eps=eps,
                                    momentum=momentum, fix_gamma=False,
                                    output_mean_var=True,
                                    no_norm=False,
                                    output_equiv_scale_bias=True, layout='NHWC')
        if is_train:
            finalize_sym = mx.sym.Group([saved_mean, saved_inv_std,
                                         equiv_scale, equiv_bias, data])
        else:
            finalize_sym = mx.sym.Group([equiv_scale, equiv_bias, data])
        data_shape = (n, h, w, c)
        x = mx.ndarray.random.uniform(-0.5, 0.5, data_shape, dtype=np.float16, ctx=ctx)
        equiv_scale_bias_shape = (c,)
        w = mx.nd.zeros((num_filter,1,1,c), dtype=np.float16, ctx=ctx)
        scale_max = 1.25
        bias_max = 1
        b = mx.ndarray.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape,
                                      dtype=np.float32, ctx=ctx)
        g = mx.ndarray.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape,
                                      dtype=np.float32, ctx=ctx)
        mov_mean_np = np.random.uniform(-bias_max, bias_max, equiv_scale_bias_shape)
        mov_var_np = np.random.uniform(1.0/scale_max, scale_max, equiv_scale_bias_shape)
        # since the models change the moving mean and variance, each model gets their own copy
        mov_mean1 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_mean2 = mx.nd.array(mov_mean_np, dtype=np.float32, ctx=ctx)
        mov_var1 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        mov_var2 = mx.nd.array(mov_var_np, dtype=np.float32, ctx=ctx)
        args_dict = {'X':x, 'B':b, 'G':g, 'W':w}
        gt_aux_states_dict =\
            {'MovMean':mov_mean1, 'MovVar':mov_var1}
        finalize_aux_states_dict =\
            {'MovMean':mov_mean2, 'MovVar':mov_var2}
        gt_exe = gt_sym.bind(ctx=ctx, args=args_dict,
                             aux_states=gt_aux_states_dict, grad_req='null')
        finalize_exe = finalize_sym.bind(ctx=ctx, args=args_dict,
                                         aux_states=finalize_aux_states_dict, grad_req='null')

        finalize_outputs = finalize_exe.forward(is_train=is_train)
        gt_outputs = gt_exe.forward(is_train=is_train)

        if is_train:
            outputs = ['saved_mean', 'saved_var', 'equiv_scale', 'equiv_bias']
        else:
            outputs = ['equiv_scale', 'equiv_bias']

        tols = [(1e-2, 1e-2), (1e-2, 1e-2), (1e-2, 1e-2), (1e-2, 1e-2)]
        for idx, out_name in enumerate(outputs):
            finalize_data = finalize_outputs[idx].asnumpy()
            gt_data = gt_outputs[idx].asnumpy()
            (rtol, atol) = tols[idx]
            assert_almost_equal(gt_data, finalize_data, rtol=rtol, atol=atol,
                                names=('gt_{}'.format(out_name),
                                       'finalize_{}'.format(out_name)))
        if is_train:
            for aux_name in ['MovMean', 'MovVar']:
                gt_data = gt_exe.aux_dict[aux_name].asnumpy()
                finalize_data = finalize_exe.aux_dict[aux_name].asnumpy()
                assert_almost_equal(gt_data, finalize_data, rtol=rtol, atol=atol,
                                    names=('gt_{}'.format(aux_name),
                                           'finalize_{}'.format(aux_name)))

    # Test BNStatsFinalize op in both inference and training modes
    for is_train in [False, True]:
        for i in range(len(nchw_shapes)):
            inshape = nchw_shapes[i]
            eps = 1e-4
            momentum = 0.9
            bn_stats_finalize_test(inshape, eps, momentum, is_train)


if __name__ == '__main__':
    import nose
    nose.runmodule()
