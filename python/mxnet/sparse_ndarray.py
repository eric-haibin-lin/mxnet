# coding: utf-8
# pylint: disable= too-many-lines, redefined-builtin, protected-access
# pylint: disable=import-error, no-name-in-module, undefined-variable
"""NDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division
try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

import ctypes
# import warnings

import os as _os
import sys as _sys

# import operator
import numpy as np
from .base import _LIB, string_types, numeric_types
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


def _new_alloc_handle(storage_type, shape, ctx, delay_alloc, dtype, aux_types, aux_shapes=None):
    """Return a new handle with specified shape and context.

    Empty handle is only used to hold results

    Returns
    -------
    handle
        A new empty ndarray handle
    """
    hdl = NDArrayHandle()
    aux_type_ids = [int(_DTYPE_NP_TO_MX[np.dtype(aux_t).type]) for aux_t in aux_types]
    aux_shapes = [(0,) for aux_t in aux_types] if aux_shapes is None else aux_shapes
    aux_shape_lens = [len(aux_shape) for aux_shape in aux_shapes]
    aux_shapes = sum(aux_shapes, ())
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
        c_array(ctypes.c_int, aux_type_ids),
        c_array(mx_uint, aux_shape_lens),
        c_array(mx_uint, aux_shapes),
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
        if not self.writable:
            raise ValueError('Failed to assign to a readonly NDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('slicing not supported in SparseNDArray yet')
            if isinstance(value, NDArray):
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                #_internal._set_value(float(value), out=self)
                raise Exception("Not supported yet")
            elif isinstance(value, (np.ndarray, np.generic)):
                # TODO(haibin) this is not very efficient. Implement sync_copyfrom for
                # sparse ndarray to avoid an extra copy
                tmp = ndarray.array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise Exception('SparseNDArray only supports [:] for assignment')

    def __getitem__(self, key):
        raise Exception('getitem Not implemented for SparseND yet!')

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
        check_call(_LIB.MXNDArrayGetAuxType(self.handle, i, ctypes.byref(aux_type)))
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
        ''' The number of aux data used to help store the sparse ndarray '''
        # This is not necessarily the size of aux_handles in the backend NDArray class,
        # since row sparse with zeros will not have any aux_handle initialized.
        return len(_STORAGE_AUX_TYPES[self.storage_type])
    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        raise Exception('Not implemented for SparseND yet!')
    # TODO(haibin) Should this be a property?
    @property
    def aux_types(self):
        aux_types = []
        num_aux = self.num_aux
        for i in xrange(num_aux):
            aux_types.append(self.aux_type(i))
        return aux_types

    def asnumpy(self):
        """Return a dense ``numpy.ndarray`` object with value copied from this array
        """
        return self.to_dense().asnumpy()

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
            hret = SparseNDArray(_new_alloc_handle(self.storage_type, self.shape, other,
                                                   True, self.dtype, self.aux_types))
            return _internal._copyto(self, out=hret)
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def copy(self):
        raise Exception('Not implemented for SparseND yet!')

    def as_in_context(self, context):
        raise Exception('Not implemented for SparseND yet!')

    def to_dense(self):
        return to_dense(self)
    # Get a read-only copy of the aux data associated with the SparseNDArray.
    # If the SparseNDArray is not yet compacted, the returned result may include invalid values
    def _aux_data(self, i, writable=False):
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetAuxNDArray(self.handle, i, ctypes.byref(hdl)))
        nd = NDArray(hdl, writable)
        if nd.ndim == 0:
            return None
        return nd

    # Get a read-only copy of the aux data associated with the SparseNDArray.
    # If the SparseNDArray is not yet compacted, the returned result may include invalid values
    def _data(self, writable=False):
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetDataNDArray(self.handle, ctypes.byref(hdl)))
        nd = NDArray(hdl, writable)
        if nd.ndim == 0:
            return ndarray.array([], dtype=self.dtype)
        return nd

    def compact(self):
        raise Exception("Not implemented yet")

# TODO We need a to_dense method to test it
def csr(values, idx, indptr, shape, ctx=None, dtype=mx_real_t, aux_types=None):
    ''' constructor '''
    hdl = NDArrayHandle()
    ctx = Context.default_ctx if ctx is None else ctx
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

def csr(values, index, indptr, shape, ctx=None, dtype=mx_real_t, aux_types=None):
    ''' shape = (d1, d2)
    values has shape (m, ) where m is the number of non-zeros entries
    '''
    assert(index.ndim == 1)
    storage_type = 'csr'
    if ctx is None:
        ctx = Context.default_ctx
    if aux_types is None:
        aux_types = _STORAGE_AUX_TYPES[storage_type]
    #TODO read aux types from inputs
    aux_shapes = [indptr.shape, index.shape]
    nd = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype, aux_types, aux_shapes))
    nd_data = nd._data(True)
    print(nd_data)
    nd_aux1 = nd._aux_data(0, True)
    nd_aux2 = nd._aux_data(1, True)
    values.copyto(nd_data)
    index.copyto(nd_aux2)
    indptr.copyto(nd_aux1)
    return nd


def row_sparse(values, index, shape, ctx=None, dtype=mx_real_t, aux_types=None):
    ''' shape = (d1, d2 .. dk)
    values is expected to have shape (d1, d2 .. dk)
    index has shape (m, ) where m is the number of rows which contains non-zeros entries
    '''
    assert(values.ndim == len(shape))
    assert(index.ndim == 1)
    storage_type = 'row_sparse'
    if ctx is None:
        ctx = Context.default_ctx
    if aux_types is None:
        aux_types = _STORAGE_AUX_TYPES[storage_type]
    #TODO read aux types from inputs
    aux_shapes = [index.shape]
    nd = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype, aux_types, aux_shapes))
    nd_data = nd._data(True)
    nd_aux = nd._aux_data(0, True)
    values.copyto(nd_data)
    index.copyto(nd_aux)
    return nd

def array(values, indices, storage_type, shape, ctx=None, dtype=mx_real_t, aux_types=None):
    ''' constructor '''
    # TODO check input array types. Assume NDArray class for now
    # TODO support other types
    # TODO also specify auxtypes
    assert (storage_type == 'row_sparse' or storage_type == 'csr')
    if aux_types is not None:
        assert isinstance(aux_types, list)
        assert len(aux_types) == len(indices)
    if not isinstance(values, NDArray):
        values = ndarray.array(values)
    for i, index in enumerate(indices):
        if not isinstance(index, NDArray):
            indices[i] = ndarray.array(index, dtype=aux_types[i] if aux_types is not None else None)

    if isinstance(shape, int):
        shape = (shape,)
    if ctx is None:
        ctx = Context.default_ctx
    if storage_type == 'row_sparse':
        arr = row_sparse(values, indices[0], shape, ctx=ctx, dtype=dtype, aux_types=aux_types)
    elif storage_type == 'csr':
        arr = csr(values, indices[0], indices[1], shape, ctx, dtype, aux_types)
    else:
        raise Exception('Not implemented for SparseND yet!')
    return arr

def to_dense(source):
    return ndarray.cast_storage(source, storage_type='default')

def empty(storage_type, shape, ctx=None, dtype=mx_real_t, aux_types=None):
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if aux_types is None:
        aux_types = _STORAGE_AUX_TYPES[storage_type]
    return SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype, aux_types))

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
    """
    if ctx is None:
        ctx = Context.default_ctx
    assert (storage_type == 'row_sparse' or storage_type == 'csr')
    if aux_types is None:
        if storage_type == 'row_sparse':
            aux_types = _STORAGE_AUX_TYPES['row_sparse']
        elif storage_type == 'csr':
            aux_types = _STORAGE_AUX_TYPES['csr']
    # pylint: disable= no-member, protected-access
    out = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, True, dtype, aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out)
    # pylint: enable= no-member, protected-access


_STORAGE_TYPE_TO_ND_CLASS = {
    _STORAGE_TYPE_STR_TO_ID['default']: ndarray.NDArray,
    _STORAGE_TYPE_STR_TO_ID['row_sparse']: SparseNDArray,
    _STORAGE_TYPE_STR_TO_ID['csr']: SparseNDArray,
}
_init_ndarray_module(_STORAGE_TYPE_TO_ND_CLASS, "mxnet")
