# pylint: skip-file
import numpy as np
import mxnet as mx
import random
from numpy.testing import assert_allclose
from mxnet.test_utils import *

def test_broadcast_add_sparse():
    data = mx.symbol.Variable('data')
    shape = (1, 1)
    data_tmp = np.ones(shape)
    test = mx.symbol.elemwise_add(data, data)
    #check_numeric_gradient(test, [data_tmp])
    # TODO implement symbolic execution
    check_symbolic_forward(test, [data_tmp], [data_tmp + 1])
    #check_symbolic_backward(test, [data_tmp], [np.ones(shape)], [2 * data_tmp])

if __name__ == '__main__':
    test_broadcast_add_sparse()
