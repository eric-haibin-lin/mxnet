# coding: utf-8
"""SparseNDArray API of mxnet."""
from __future__ import absolute_import
from __future__ import division
try:
    from __builtin__ import slice as py_slice
except ImportError:
    from builtins import slice as py_slice

import ctypes
import warnings

import os as _os
import sys as _sys

# import operator
import numpy as np
from .base import _LIB, numeric_types #string_types
from .base import c_array, mx_real_t  # , py_str, c_str
from .base import mx_uint, NDArrayHandle, check_call
# from .base import ctypes2buffer
from .context import Context
from . import _ndarray_internal as _internal
from . import ndarray
from .ndarray import _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from .ndarray import _STORAGE_TYPE_STR_TO_ID#, _STORAGE_TYPE_ID_TO_STR
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
    """Return a new handle with specified shape, type and context.

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
    """An array object representing a multidimensional, homogeneous array of
fixed-size items, stored in sparse format.

    """
    __slots__ = []

    #def __reduce__(self):
    #    return SparseNDArray, (None,), self.__getstate__()

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
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value.

        Parameters
        ----------
        key : slice
            The indexing key.
        value : NDArray or numpy.ndarray
            The value to set.

        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly NDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slicing not supported in SparseNDArray.')
            if isinstance(value, NDArray):
                # avoid copying to itself
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise Exception("Assigning numeric types to SparseNDArray not supported yet.")
            elif isinstance(value, (np.ndarray, np.generic)):
                # TODO(haibin) this is not efficient. Implement sync_copyfrom for
                # sparse ndarray to avoid an extra copy
                warnings.warn('Assigning non-NDArray object to SparseNDArray is not efficient',
                              RuntimeWarning)
                tmp = ndarray.array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise Exception('SparseNDArray only supports [:] for assignment')

    def __getitem__(self, key):
        stype = self.storage_type
        assert(stype == 'csr'), "getitem for " + str(stype) + " not implemented yet"
        if isinstance(key, int):
            raise Exception("Not implemented yet")
        if isinstance(key, py_slice):
            if key.step is not None:
                raise ValueError('NDArray only supports continuous slicing on axis 0')
            if key.start is not None or key.stop is not None:
                return self._slice(key.start, key.stop)
            else:
                return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')

    def _sync_copyfrom(self, source_array):
        raise Exception('Not implemented for SparseND yet!')

    def _slice(self, start, stop):
        """Returns a read-only sliced SparseNDArray that shares memory with current one.
        For csr SparseNDArray, it only slices the indptr array, and keeps the original values
        and indices.

        The existing slice operation is not very efficient when it's copied, since the indices
        and values are a superset of the sliced region.


        Parameters
        ----------
        start : int
            Starting index of slice.
        stop : int
            Finishing index of slice.

        Example
        ----------
        >>> indptr = np.array([0, 2, 3, 6])
        >>> indices = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> a = mx.sparse_nd.csr(data, indptr, indices, (3, 3))
        >>> a.asnumpy()
        array([[1, 0, 2],
               [0, 0, 3],
               [4, 5, 6]])

        >>> a[1:2].asnumpy()
        array([[0, 0, 3]])

        >>> a[1:2].indptr.asnumpy()
        array([[2, 3]])

        >>> a[1:2].indicies.asnumpy()
        array([0, 2, 2, 0, 1, 2])

        >>> a[1:2].values.asnumpy()
        array([1, 2, 3, 4, 5, 6])

        """
        stype = self.storage_type
        assert(stype == 'csr'), "_slice for " + str(stype) + " not implemented yet"
        warnings.warn('slicing SparseNDArray is not efficient', RuntimeWarning)
        handle = NDArrayHandle()
        start = mx_uint(start) if start else mx_uint(0)
        stop = mx_uint(stop) if stop else mx_uint(self.shape[0])
        check_call(_LIB.MXNDArraySlice(
            self.handle, start, stop, ctypes.byref(handle)))
        return SparseNDArray(handle=handle, writable=False)

    def _at(self, idx):
        raise Exception('at operator for SparseND is not supported.')

    def reshape(self, shape):
        raise Exception('Not implemented for SparseND yet!')

    def broadcast_to(self, shape):
        raise Exception('Not implemented for SparseND yet!')

    def _aux_type(self, i):
        """Data-type of the array’s ith aux data.

        Returns
        -------
        numpy.dtype
            This NDArray's data type.
        """
        aux_type = ctypes.c_int()
        check_call(_LIB.MXNDArrayGetAuxType(self.handle, i, ctypes.byref(aux_type)))
        return _DTYPE_MX_TO_NP[aux_type.value]

    @property
    def values(self):
        return self._data(0)

    @property
    def indices(self):
        stype = self.storage_type
        if stype == 'row_sparse':
            return self._aux_data(0)
        elif stype == 'csr':
            return self._aux_data(1)
        raise Exception("unknown storage type " + stype)

    @property
    def indptr(self):
        stype = self.storage_type
        if stype == 'csr':
            return self._aux_data(0)
        raise Exception("unknown storage type " + stype)

    @property
    def _num_aux(self):
        ''' The number of aux data used to help store the sparse ndarray.
        '''
        return len(_STORAGE_AUX_TYPES[self.storage_type])

    @property
    # pylint: disable= invalid-name, undefined-variable
    def T(self):
        raise Exception('Transpose is not supported for SparseNDArray.')

    @property
    def aux_types(self):
        ''' The data types of the aux data for the SparseNDArray.
        '''
        aux_types = []
        num_aux = self._num_aux
        for i in xrange(num_aux):
            aux_types.append(self._aux_type(i))
        return aux_types

    def asnumpy(self):
        """Return a dense ``numpy.ndarray`` object with value copied from this array

        """
        return self.to_dense().asnumpy()

    def astype(self, dtype):
        raise Exception('Not implemented for SparseND yet!')

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``NDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray
            The copied array. If ``other`` is an ``NDArray``, then the return value
            and ``other`` will point to the same ``NDArray``.
       """
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

    def to_dense(self):
        return to_dense(self)

    def _aux_data(self, i, writable=False):
        """ Get a reference to the i-th aux data associated with the SparseNDArray. If the
        SparseNDArray is not yet compacted, the returned result may include invalid values.

        """
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetAuxNDArray(self.handle, i, ctypes.byref(hdl)))
        return NDArray(hdl, writable)

    def _data(self, writable=False):
        """ Get a reference to the data associated with the SparseNDArray. If the
        SparseNDArray is not yet compacted, the returned result may include invalid values.

        """
        hdl = NDArrayHandle()
        check_call(_LIB.MXNDArrayGetDataNDArray(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, writable)


    def compact(self):
        raise Exception("Not implemented yet")

def _prepare_src_array(src, dtype, default_dtype):
    if isinstance(src, NDArray):
        dtype = src.dtype if dtype is None else dtype
    else:
        dtype = default_dtype if dtype is None else dtype
        if not isinstance(src, np.ndarray):
            try:
                src = np.array(src, dtype=dtype)
            except:
                raise TypeError('values must be array like object')
    return src, dtype

def csr(values, indptr, indices, shape, ctx=None, dtype=None, indptr_type=None, indices_type=None):
    """Creates a 2D array with compressed sparse row format.

    A SparseNDArray with `csr` storage represents a NDArray as three separate arrays: `values`,
    `indptr` and `indices`. It uses the standard CSR representation where the column indices for
    row i are stored in indices[indptr[i]:indptr[i+1]] and their corresponding values are stored
    in values[indptr[i]:indptr[i+1]].

    Parameters
    ----------
    values: array_like
        An object exposing the array interface, with shape [nnz], where D0 is the number of
        non-zero entries.
    indptr: array_like
        An object exposing the array interface, with shape [D0 + 1]. The first element in indptr
        should always be zero.
    indices: array_like
        An object exposing the array interface, with shape [nnz].
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``values.dtype``
        if `values` is an `NDArray`, `float32` otherwise.
    indptr_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indptr.dtype``
        if `indptr` is an `NDArray`, `int32` otherwise.
    indices_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indices.dtype``
        if `indicies` is an `NDArray`, `int32` otherwise.

    Returns
    -------
    SparseNDArray
        An `SparseNDArray` with the `csr` storage representation.
    """
    storage_type = 'csr'
    # context
    if ctx is None:
        ctx = Context.default_ctx
    # prepare src array and types
    values, dtype = _prepare_src_array(values, dtype, mx_real_t)
    indptr, indptr_type = _prepare_src_array(indptr, indptr_type,
                                             _STORAGE_AUX_TYPES[storage_type][0])
    indices, indices_type = _prepare_src_array(indices, indices_type,
                                               _STORAGE_AUX_TYPES[storage_type][1])
    # verify types
    assert('int' in str(indptr_type) or 'long' in str(indptr_type))
    assert('int' in str(indices_type) or 'long' in str(indices_type))
    # verify shapes
    aux_shapes = [indptr.shape, indices.shape]
    assert(values.ndim == 1)
    assert(indptr.ndim == 1)
    assert(indices.ndim == 1)
    assert(len(shape) == 2)
    result = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                             [indptr_type, indices_type], aux_shapes))
    # assign indptr, indices and values
    values_ref = result._data(True)
    indptr_ref = result._aux_data(0, True)
    indices_ref = result._aux_data(1, True)
    values_ref[:] = values
    indptr_ref[:] = indptr
    indices_ref[:] = indices
    return result

def row_sparse(values, indices, shape, ctx=None, dtype=None, indices_type=None):
    """Creates a row sparse array with a set of tensor slices at given indices.

    A SparseNDArray with `row_sparse` storage is typically used to represent a subset of a larger
    NDArray  with `default_storage` of shape [LARGE0, D1, .. , DN] where LARGE0 >> D0. The values
    in indices are the indices in the first dimension of the slices that have been extracted from
    the larger NDArray.

    The corresponding NDArray ``dense`` with `default_storage` represented by a ``rsp``
    SparseNDArray with `row_sparse` storage has

    ``dense[rsp.indices[i], :, :, :, ...] = rsp.values[i, :, :, :, ...]``

    `row_sparse` SparseNDArray is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. SparseEmbedding).

    Parameters
    ----------
    values: array_like
        An object exposing the array interface, with shape [D0, D1, .. Dn], where D0 is
        the number of rows with non-zeros entries.
    indices: array_like
        An object exposing the array interface, with shape [D0].
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``values.dtype``
        if `values` is an `NDArray`, `float32` otherwise.
    indices_type: str or numpy.dtype, optional
        The data type of the indices array. The default dtype is ``indices.dtype``
        if `indicies` is an `NDArray`, `int32` otherwise.

    Returns
    -------
    SparseNDArray
        An `SparseNDArray` with the `row_sparse` storage representation.
    """
    storage_type = 'row_sparse'
    # context
    if ctx is None:
        ctx = Context.default_ctx
    # prepare src array and types
    values, dtype = _prepare_src_array(values, dtype, mx_real_t)
    indices, indices_type = _prepare_src_array(indices, indices_type,
                                               _STORAGE_AUX_TYPES[storage_type][0])
    # verify types
    assert('int' in str(indices_type) or 'long' in str(indices_type))
    # verify shapes
    assert(values.ndim == len(shape))
    assert(indices.ndim == 1)
    result = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                             [indices_type], [indices.shape]))
    # assign indices and values
    values_ref = result._data(True)
    indices_ref = result._aux_data(0, True)
    values_ref[:] = values
    indices_ref[:] = indices
    return result

def to_dense(source):
    """ Return a dense array representation of this SparseNDArray.

    Returns
    -------
    SparseNDArray
        The dense array with default storage
    """
    return ndarray.cast_storage(source, storage_type='default_storage')

def zeros(storage_type, shape, ctx=None, dtype=None, aux_types=None):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of int
        The shape of the empty array
    storage_type: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)
    aux_types: list of numpy.dtype, optional
        An optional type for the aux data for SparseNDArray (default values depends
        on the storage type)

    Returns
    -------
    SparseNDArray
        A created array
    """
    if ctx is None:
        ctx = Context.default_ctx

    dtype = mx_real_t if dtype is None else dtype
    if aux_types is None:
        if storage_type == 'row_sparse' or storage_type == 'csr':
            aux_types = _STORAGE_AUX_TYPES[storage_type]
        else:
            raise Exception("unknown storage type")
    assert(len(aux_types) == len(_STORAGE_AUX_TYPES[storage_type]))
    out = SparseNDArray(_new_alloc_handle(storage_type, shape, ctx, True, dtype, aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out)

_STORAGE_TYPE_TO_ND_CLASS = {
    _STORAGE_TYPE_STR_TO_ID['default_storage']: ndarray.NDArray,
    _STORAGE_TYPE_STR_TO_ID['row_sparse']: SparseNDArray,
    _STORAGE_TYPE_STR_TO_ID['csr']: SparseNDArray,
}

_init_ndarray_module(_STORAGE_TYPE_TO_ND_CLASS, "mxnet")
