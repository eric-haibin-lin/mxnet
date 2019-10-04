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

import itertools
import numpy as np
import mxnet as mx
from mxnet.test_utils import *

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed


# * GroupAdaGrad
class PyGroupAdaGrad(mx.optimizer.Optimizer):
    """The python reference of Group AdaGrad optimizer.

    Parameters
    ----------
    eps: float, optional
        Small value to avoid division by 0.

    """

    def __init__(self, eps=1e-5, **kwargs):
        super(PyGroupAdaGrad, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        assert len(weight.shape) == 2
        history = mx.nd.zeros(
            (weight.shape[0], 1), weight.context, stype=weight.stype)
        return history

    def update(self, index, weight, grad, state):
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        assert wd == 0

        history = state
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        history[:] += mx.nd.mean(mx.nd.square(grad), axis=1, keepdims=True)
        div = lr * grad / mx.nd.sqrt(history + self.float_stable_eps)
        weight[:] -= div


def test_group_adagrad():
    mx.random.seed(0)
    opt1 = PyGroupAdaGrad
    opt2 = mx.optimizer.contrib.GroupAdaGrad
    shape = (3, 4)
    eps_options = [{}, {'eps': 1e-8}]
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    for dtype in [np.float32]:
        for options in itertools.product(eps_options, cg_options, rg_options):
            kwarg = dict(wd=0.0)
            for option in options:
                kwarg.update(option)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                compare_states=False)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                w_stype='row_sparse',
                g_stype='row_sparse',
                compare_states=False)
            compare_optimizer(
                opt1(**kwarg),
                opt2(**kwarg),
                shape,
                dtype,
                g_stype='row_sparse',
                compare_states=False)

@with_seed()
def test_adamw():
    def get_refs(m, v, weight, grad_rescale, beta1, beta2, eta, wd, epsilon):
        m_ref = beta1*m + (1-beta1)*grad_rescale
        v_ref = beta2*v + (1-beta2)*(grad_rescale**2)
        weight_ref = weight - eta * (1 * m_ref / (v_ref.sqrt() + epsilon) + weight * wd)
        return m_ref, v_ref, weight_ref

    shape = (3, 4)
    weight = mx.nd.random.uniform(shape=shape)

    grad = mx.nd.random.uniform(shape=shape)
    m = mx.nd.random.uniform(shape=shape)
    v = mx.nd.random.uniform(shape=shape)
    rescale_grad = mx.nd.array([10])
    eta, lr, wd, epsilon = 1, 1, 0, 1e-8
    beta1, beta2 = 0.9, 0.999
    kwargs = {'eta': eta, 'lr': lr, 'wd': wd, 'epsilon': epsilon,
              'beta1': beta1, 'beta2': beta2}

    # update is skipped for rescale = nan scalar
    tested_grad = [rescale_grad * 0, rescale_grad * np.nan, rescale_grad * np.inf]
    tested_rescaled_grad = [np.nan]
    tested_rescaled_grad.extend(tested_grad)

    weight_ref = weight.copy()
    for rescaled_grad in tested_rescaled_grad:
        mx.nd.contrib.adamw_update(weight, grad, m, v,
                                   rescaled_grad, out=weight, **kwargs)
            # weight remains unchanged
        mx.test_utils.assert_almost_equal(weight_ref, weight)

    weight_ref = weight.copy()
    weight_fp16 = weight.astype('float16')
    grad_fp16 = grad.astype('float16')
    weight_fp16_ref = weight_fp16.copy()
    for rescaled_grad in tested_grad:
        mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                      rescaled_grad, out=weight_fp16, **kwargs)
        # weight remains unchanged
        mx.test_utils.assert_almost_equal(weight_ref, weight)
        mx.test_utils.assert_almost_equal(weight_fp16_ref, weight_fp16)

    # reference normal update
    grad_rescale = rescale_grad * grad
    m_ref,  v_ref, weight_ref = get_refs(m, v, weight, grad_rescale, beta1, beta2, eta, wd, epsilon)
    weight_test = weight.copy()
    # op normal update
    mx.nd.contrib.adamw_update(weight_test, grad, m, v,
                               rescale_grad, out=weight_test, **kwargs)
    mx.test_utils.assert_almost_equal(weight_ref, weight_test, atol=1e-6)
    mx.test_utils.assert_almost_equal(m_ref, m)
    mx.test_utils.assert_almost_equal(v_ref, v, atol=1e-6)

    # reference normal multi-precision update
    grad_rescale = rescale_grad * grad_fp16.astype('float32')
    m_ref,  v_ref, weight_ref = get_refs(m, v, weight, grad_rescale, beta1, beta2, eta, wd, epsilon)
    weight_fp16_ref = weight_ref.astype('float16')
    # op normal multi-precision update
    mx.nd.contrib.mp_adamw_update(weight_fp16, grad_fp16, m, v, weight,
                                  rescale_grad, out=weight_fp16, **kwargs)
    mx.test_utils.assert_almost_equal(m_ref, m)
    mx.test_utils.assert_almost_equal(v_ref, v)
    mx.test_utils.assert_almost_equal(weight_ref, weight, atol=1e-6)
    mx.test_utils.assert_almost_equal(weight_fp16_ref, weight_fp16, rtol=1e-3, atol=1e-6)


if __name__ == '__main__':
    import nose
    nose.runmodule()
