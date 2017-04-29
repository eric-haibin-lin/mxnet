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
        from ._ctypes.ndarray import _init_ndarray_module
    elif _sys.version_info >= (3, 0):
        from ._cy3.ndarray import _init_ndarray_module
    else:
        from ._cy2.ndarray import _init_ndarray_module
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.ndarray import _init_ndarray_module

# pylint: enable= no-member
_STORAGE_TYPE_ID_TO_STR = {
    -1 : 'undefined',
    0  : 'default',
    1  : 'row_sparse',
    2  : 'csr',
}

_STORAGE_TYPE_STR_TO_ID = {
    'undefined'  : -1,
    'default'    : 0,
    'row_sparse' : 1,
    'csr'        : 2,
}

_STORAGE_AUX_TYPES = {
    'row_sparse' : [np.int32],
    'csr'        : [np.int32, np.int32]
}

def _new_alloc_handle(storage_type, shape, ctx, delay_alloc=True,
                      dtype=mx_real_t, aux_types=None):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    handle
        A new empty ndarray handle
    """
    hdl = NDArrayHandle()
    aux_type_list = [int(_DTYPE_NP_TO_MX[np.dtype(aux_t).type]) for aux_t in aux_types]
    num_aux = mx_uint(len(aux_types))
    check_call(_LIB.MXNDArrayCreateSparseEx(
        ctypes.c_int(int(_STORAGE_TYPE_STR_TO_ID[storage_type])),
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(delay_alloc)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        num_aux,
        c_array(ctypes.c_int, aux_type_list),
        ctypes.byref(hdl)))
    return hdl

class SparseNDArray(NDArray):
    ''' sparse ndarray '''
    __slots__ = []

    #def __repr__(self):
    def __reduce__(self):
        return (SparseNDArray, (None,), self.__getstate__())
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
        raise Exception('at operator for SparseND is not supported.')
    def reshape(self, shape):
        raise Exception('Not implemented for SparseND yet!')
    def broadcast_to(self, shape):
        raise Exception('Not implemented for SparseND yet!')
    #def wait_to_read(self):
    #@property
    #def shape(self):

    @property
    def size(self):
        raise Exception('Not implemented for SparseND yet!')
    #@property
    #def context(self):
    #@property
    #def dtype(self):
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

#TODO We need a to_dense method to test it
def csr(values, indptr, idx, shape, ctx=Context.default_ctx, dtype=mx_real_t, aux_types=None):
    ''' constructor '''
    hdl = NDArrayHandle()
    #TODO currently only supports NDArray input
    assert(isinstance(values, NDArray))
    assert(isinstance(index, NDArray))
    indices = c_array(NDArrayHandle, [idx.handle, indptr.handle])
    num_aux = mx_uint(2)
    # TODO create an empty handle with specified types, then assign values
    check_call(_LIB.MXNDArrayCreateSparse(
        values.handle, num_aux, indices,
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(_STORAGE_TYPE_STR_TO_ID['csr']),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(False)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return SparseNDArray(hdl)

# pylint: enable= no-member
#TODO(haibin) also specify aux_types
def row_sparse(values, index, shape, ctx=Context.default_ctx, dtype=mx_real_t, aux_types=None):
    ''' constructor '''
    hdl = NDArrayHandle()
    assert(isinstance(values, NDArray))
    assert(isinstance(index, NDArray))
    indices = c_array(NDArrayHandle, [index.handle])
    num_aux = mx_uint(1)
    # TODO create an empty handle with specified types, then assign values
    check_call(_LIB.MXNDArrayCreateSparse(
        values.handle, num_aux, indices,
        c_array(mx_uint, shape),
        mx_uint(len(shape)),
        ctypes.c_int(_STORAGE_TYPE_STR_TO_ID['row_sparse']),
        ctypes.c_int(ctx.device_typeid),
        ctypes.c_int(ctx.device_id),
        ctypes.c_int(int(False)),
        ctypes.c_int(int(_DTYPE_NP_TO_MX[np.dtype(dtype).type])),
        ctypes.byref(hdl)))
    return SparseNDArray(hdl)

def array(values, index_list, storage_type, shape, ctx=None, dtype=mx_real_t, aux_types=None):
    # TODO check input array types. Assume NDArray class for now
    # TODO support other types
    # TODO also specify auxtypes
    assert(storage_type == 'row_sparse')
    if not isinstance(values, NDArray):
        values = ndarray.array(values)
    for i, index in enumerate(index_list):
       if not isinstance(index, NDArray):
           index_list[i] = ndarray.array(index)
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    arr = row_sparse(values, index_list[0], shape, ctx=ctx, dtype=dtype, aux_types=aux_types)
    return arr

def to_dense(source):
    return ndarray.cast_storage(source, storage_type=_STORAGE_TYPE_STR_TO_ID['default'])

def zeros(shape, storage_type, ctx=None, dtype=mx_real_t, aux_types=None):
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
    assert(storage_type == 'row_sparse')
    if aux_types == None:
        aux_types = _STORAGE_AUX_TYPES['row_sparse']
    # pylint: disable= no-member, protected-access
    out = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx,
                                          aux_types=aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out)
    # pylint: enable= no-member, protected-access

_STORAGE_TYPE_TO_ND_CLASS = {
    _STORAGE_TYPE_STR_TO_ID['default']  : ndarray.NDArray,
    _STORAGE_TYPE_STR_TO_ID['row_sparse'] : SparseNDArray,
    _STORAGE_TYPE_STR_TO_ID['csr']        : SparseNDArray,
}
_init_ndarray_module(_STORAGE_TYPE_TO_ND_CLASS, "mxnet")
