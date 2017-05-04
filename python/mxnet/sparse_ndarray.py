# coding: utf-8
# pylint: disable= too-many-lines, redefined-builtin, protected-access
# pylint: disable=import-error, no-name-in-module, undefined-variable
"""NDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division
# try:
#    from __builtin__ import slice as py_slice
# except ImportError:
#    from builtins import slice as py_slice

import ctypes
# import warnings

import os as _os
import sys as _sys

# import operator
import numpy as np
from .base import _LIB  # , string_types, numeric_types
from .base import c_array, mx_real_t  # , py_str, c_str
from .base import mx_uint, NDArrayHandle, check_call
# from .base import ctypes2buffer
from .context import Context
from . import _ndarray_internal as _internal
from . import ndarray
from .ndarray import _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_ID_TO_STR, _STORAGE_TYPE_STR_TO_ID
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

_STORAGE_AUX_TYPES = {
    'row_sparse': [np.int32],
    'csr': [np.int32, np.int32]
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

    # def __repr__(self):
    def __reduce__(self):
        return SparseNDArray, (None,), self.__getstate__()

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

    # def wait_to_read(self):
    # @property
    # def shape(self):
    def aux_type(self, i):
        aux_type = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetAuxType(
                   self.handle, i, ctypes.byref(aux_type)))
        return _DTYPE_MX_TO_NP[aux_type.value]

    @property
    def size(self):
        raise Exception('Not implemented for SparseND yet!')

    # @property
    # def context(self):
    # @property
    # def dtype(self):
    @property
    def num_aux(self):
        num_aux = mx_uint()
        check_call(_LIB.MXNDArrayGetNumAux(self.handle, ctypes.byref(num_aux)))
        return num_aux.value
    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        raise Exception('Not implemented for SparseND yet!')
    # TODO(haibin) Should this be a property?
    def aux_types(self):
        aux_types = []
        num_aux = self.num_aux
        for i in xrange(num_aux):
            aux_types.append(self.aux_type(i))
        return aux_types

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
        if isinstance(other, NDArray):
            if other.handle is self.handle:
                warnings.warn('You are attempting to copy an array to itself', RuntimeWarning)
                return
            return _internal._copyto(self, out=other)
        elif isinstance(other, Context):
            hret = SparseNDArray(_new_alloc_handle(self.storage_type, self.shape, other, True, self.dtype, self.aux_types()))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def copy(self):
        raise Exception('Not implemented for SparseND yet!')

    def as_in_context(self, context):
        raise Exception('Not implemented for SparseND yet!')

    def to_dense(self):
        return to_dense(self)


# TODO We need a to_dense method to test it
def csr(values, idx, indptr, shape, ctx=Context.default_ctx, dtype=mx_real_t, aux_types=None):
    ''' constructor '''
    hdl = NDArrayHandle()
    # TODO currently only supports NDArray input
    assert (isinstance(values, NDArray))
    assert (isinstance(indptr, NDArray))
    assert (isinstance(idx, NDArray))
    assert (isinstance(shape, tuple))
    indices = c_array(NDArrayHandle, [indptr.handle, idx.handle])
    num_aux = mx_uint(2)
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
# TODO(haibin) also specify aux_types
def row_sparse(values, index, shape, ctx=Context.default_ctx, dtype=mx_real_t, aux_types=None):
    ''' rsp constructor which only accepts NDArray as input '''
    hdl = NDArrayHandle()
    assert (isinstance(values, NDArray))
    assert (isinstance(index, NDArray))
    indices = c_array(NDArrayHandle, [index.handle])
    num_aux = mx_uint(1)
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
    ''' constructor '''
    # TODO check input array types. Assume NDArray class for now
    # TODO support other types
    # TODO also specify auxtypes
    assert (storage_type == 'row_sparse' or storage_type == 'csr')
    if aux_types is not None:
        assert isinstance(aux_types, list)
        assert len(aux_types) == len(index_list)
    if not isinstance(values, NDArray):
        values = ndarray.array(values)
    for i, index in enumerate(index_list):
        if not isinstance(index, NDArray):
            index_list[i] = ndarray.array(index, dtype=aux_types[i] if aux_types is not None else None)

    if isinstance(shape, int):
        shape = (shape,)
    if ctx is None:
        ctx = Context.default_ctx
    if storage_type == 'row_sparse':
        arr = row_sparse(values, index_list[0], shape, ctx=ctx, dtype=dtype, aux_types=aux_types)
    elif storage_type == 'csr':
        arr = csr(values, index_list[0], index_list[1], shape, ctx, dtype, aux_types)
    else:
        raise Exception('Not implemented for SparseND yet!')
    return arr


def to_dense(source):
    return ndarray.cast_storage(source, storage_type='default')


def zeros(shape, storage_type, ctx=None, dtype=mx_real_t, aux_types=None):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array
    storage_type:
        'row_sparse', etc
    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)
    aux_types:
        [np.int32], etc

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
    assert (storage_type == 'row_sparse' or storage_type == 'csr')
    if aux_types is None:
        if 'row_sparse' == storage_type:
            aux_types = _STORAGE_AUX_TYPES['row_sparse']
        elif 'csr' == storage_type:
            aux_types = _STORAGE_AUX_TYPES['csr']
    # pylint: disable= no-member, protected-access
    out = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx,
                                          aux_types=aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out)
    # pylint: enable= no-member, protected-access


_STORAGE_TYPE_TO_ND_CLASS = {
    _STORAGE_TYPE_STR_TO_ID['default']: ndarray.NDArray,
    _STORAGE_TYPE_STR_TO_ID['row_sparse']: SparseNDArray,
    _STORAGE_TYPE_STR_TO_ID['csr']: SparseNDArray,
}
_init_ndarray_module(_STORAGE_TYPE_TO_ND_CLASS, "mxnet")
