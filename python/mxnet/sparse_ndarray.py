# coding: utf-8
# pylint: disable= too-many-lines, redefined-builtin, protected-access
# pylint: disable=import-error, no-name-in-module, undefined-variable
"""NDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division
#try:
#    from __builtin__ import slice as py_slice
#except ImportError:
#    from builtins import slice as py_slice

import ctypes
#import warnings

import os as _os
import sys as _sys

#import operator
import numpy as np
from .base import _LIB#, string_types, numeric_types
from .base import c_array, mx_real_t#, py_str, c_str
from .base import mx_uint, NDArrayHandle, check_call
#from .base import ctypes2buffer
from .context import Context
from . import _ndarray_internal as _internal
from . import ndarray
from .ndarray import _DTYPE_NP_TO_MX#, _DTYPE_MX_TO_NP
from .ndarray import NDArray

# Use different verison of SymbolBase
# When possible, use cython to speedup part of computation.
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.ndarray import NDArrayBase, _init_ndarray_module
    elif _sys.version_info >= (3, 0):
        from ._cy3.ndarray import NDArrayBase, _init_ndarray_module
    else:
        from ._cy2.ndarray import NDArrayBase, _init_ndarray_module
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.ndarray import NDArrayBase, _init_ndarray_module

# pylint: enable= no-member
_STORAGE_TYPE_ID_TO_STR = {
    0 : 'undefined',
    1 : 'default',
    2 : 'row_sparse',
    3 : 'csr',
}

_STORAGE_TYPE_STR_TO_ID = {
    'undefined' : 0,
    'default' : 1,
    'row_sparse' : 2,
    'csr' : 3,
}

#FIXME change default type for aux_type. Make aux type a list
def _new_alloc_handle(storage_type, shape, ctx, delay_alloc=True,
                      dtype=mx_real_t, aux_type=mx_real_t):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    handle
        A new empty ndarray handle
    """
    hdl = NDArrayHandle()
    aux_type_list = [int(_DTYPE_NP_TO_MX[np.dtype(aux_type).type])]
    check_call(_LIB.MXNDArrayCreateSparseEx(
        ctypes.c_int(int(_STORAGE_TYPE_STR_TO_ID[storage_type])),
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        mx_uint(1),
        c_array(ctypes.c_int, aux_type_list),
        ctypes.byref(hdl)))
    return hdl

class SparseNDArray(NDArray):
    ''' sparse ndarray '''
    __slots__ = []

    def __repr__(self):
        """Return a string representation of the array"""
        #TODO also print shape info
        #shape_info = 'x'.join(['%d' % x for x in self.shape])
        return '<%s>' % (self.__class__.__name__)
    def __add__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __iadd__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __radd__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __sub__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __isub__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __rsub__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __mul__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __neg__(self):
        raise Exception('Not implemented for SparseND yet!')
    def __imul__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __rmul__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __div__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __rdiv__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __idiv__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __truediv__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __rtruediv__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __itruediv__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __pow__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __rpow__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __eq__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __ne__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __gt__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __ge__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __lt__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __le__(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def __getstate__(self):
        raise Exception('Not implemented for SparseND yet!')
    def __setstate__(self, state):
        raise Exception('Not implemented for SparseND yet!')
    def __setitem__(self, key, value):
        raise Exception('Not implemented for SparseND yet!')
    def __getitem__(self, key):
        raise Exception('Not implemented for SparseND yet!')
    def _sync_copyfrom(self, source_array):
        raise Exception('Not implemented for SparseND yet!')
    def _slice(self, start, stop):
        raise Exception('Not implemented for SparseND yet!')
    def _at(self, idx):
        raise Exception('Not implemented for SparseND yet!')
    def reshape(self, shape):
        raise Exception('Not implemented for SparseND yet!')
    def broadcast_to(self, shape):
        raise Exception('Not implemented for SparseND yet!')
    def wait_to_read(self):
        raise Exception('Not implemented for SparseND yet!')
    #@property
    #def shape(self):
    #inherited from parent

    @property
    def size(self):
        raise Exception('Not implemented for SparseND yet!')
    @property
    def context(self):
        raise Exception('Not implemented for SparseND yet!')
    @property
    def dtype(self):
        raise Exception('Not implemented for SparseND yet!')
    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        raise Exception('Not implemented for SparseND yet!')
    def asnumpy(self):
        """Return a dense ``numpy.ndarray`` object with value copied from this array
        """
        dense_nd = self.to_dense()
        return dense_nd.asnumpy()
    def asscalar(self):
        raise Exception('Not implemented for SparseND yet!')
    def astype(self, dtype):
        raise Exception('Not implemented for SparseND yet!')
    def copyto(self, other):
        raise Exception('Not implemented for SparseND yet!')
    def copy(self):
        raise Exception('Not implemented for SparseND yet!')
    def as_in_context(self, context):
        raise Exception('Not implemented for SparseND yet!')
    def to_dense(self):
        return to_dense(self)

# pylint: enable= no-member
def row_sparse(values, index, shape, ctx=Context.default_ctx, dtype=mx_real_t):
    ''' constructor '''
    hdl = NDArrayHandle()
    assert(isinstance(values, NDArrayBase))
    assert(isinstance(index, NDArrayBase))
    indices = c_array(NDArrayHandle, [index.handle])
    check_call(_LIB.MXNDArrayCreateSparse(
        values.handle, mx_uint(1), indices,
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(_STORAGE_TYPE_STR_TO_ID['row_sparse']),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(False)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return SparseNDArray(hdl)

def array(values, index_list, storage_type, shape, ctx=None, dtype=mx_real_t):
    # TODO check input array types. Assume NDArray class for now
    # TODO support other types
    assert(storage_type == 'row_sparse')
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    arr = row_sparse(values, index_list[0], shape, ctx=ctx, dtype=dtype)
    return arr

# Temporary function for testing purpose
def to_dense(source):
    hdl = NDArrayHandle()
    check_call(_LIB.MXNDArrayConvert(
        source.handle, _STORAGE_TYPE_STR_TO_ID['default'],
        ctypes.byref(hdl)))
    return ndarray.NDArray(handle=hdl, writable=True)

def zeros(shape, storage_type, ctx=None, dtype=mx_real_t):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array
    storage_type:

    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)

    Returns
    -------
    NDArray
        A created array

    Examples
    --------
    >>> mx.nd.zeros(1).asnumpy()
    array([ 0.], dtype=float32)
    >>> mx.nd.zeros((1,2), mx.gpu(0))
    <NDArray 1x2 @gpu(0)>
    >>> mx.nd.zeros((1,2), mx.gpu(0), 'float16').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """
    if ctx is None:
        ctx = Context.default_ctx
    if storage_type != 'default':
        # pylint: disable= no-member, protected-access
        out = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx))
        return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out)
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype)
    # pylint: enable= no-member, protected-access

_STORAGE_TYPE_TO_ND_CLASS = {
    1 : ndarray.NDArray,
    2 : SparseNDArray,
    3 : SparseNDArray
}
_init_ndarray_module(_STORAGE_TYPE_TO_ND_CLASS, "mxnet")
