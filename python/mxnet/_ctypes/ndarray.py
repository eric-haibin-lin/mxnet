# coding: utf-8
# pylint: disable=invalid-name, protected-access, too-many-arguments,  global-statement
"""Symbolic configuration API."""
from __future__ import absolute_import as _abs

import ctypes
import sys as _sys
import numpy as np

from ..base import _LIB
from ..base import c_array, py_str, c_str, mx_uint
from ..base import NDArrayHandle, OpHandle
from ..base import check_call
from ..ndarray_doc import _build_doc

_ndarray_cls_map = {}

class NDArrayBase(object):
    """Base data structure for ndarray"""
    __slots__ = ["handle", "writable"]
    # pylint: disable= no-member
    def __init__(self, handle, writable=True):
        """initialize a new NDArray

        Parameters
        ----------
        handle : NDArrayHandle
            NDArray handle of C API
        """
        if handle is not None:
            assert isinstance(handle, NDArrayHandle)
        self.handle = handle
        self.writable = writable

    def __del__(self):
        check_call(_LIB.MXNDArrayFree(self.handle))

    def __reduce__(self):
        return (_ndarray_cls[1], (None,), self.__getstate__())

    @property
    def shape(self):
        """Tuple of array dimensions.

        Examples
        --------
        >>> x = mx.nd.array([1, 2, 3, 4])
        >>> x.shape
        (4,)
        >>> y = mx.nd.zeros((2, 3, 4))
        >>> y.shape
        (2, 3, 4)
        """
        ndim = mx_uint()
        pdata = ctypes.POINTER(mx_uint)()
        check_call(_LIB.MXNDArrayGetShape(
            self.handle, ctypes.byref(ndim), ctypes.byref(pdata)))
        return tuple(pdata[:ndim.value])

# pylint: disable=too-many-locals, invalid-name
def _make_ndarray_function(handle, name):
    """Create a NDArray function from the FunctionHandle."""
    real_name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()
    key_var_num_args = ctypes.c_char_p()
    ret_type = ctypes.c_char_p()

    check_call(_LIB.MXSymbolGetAtomicSymbolInfo(
        handle, ctypes.byref(real_name), ctypes.byref(desc),
        ctypes.byref(num_args),
        ctypes.byref(arg_names),
        ctypes.byref(arg_types),
        ctypes.byref(arg_descs),
        ctypes.byref(key_var_num_args),
        ctypes.byref(ret_type)))
    narg = int(num_args.value)
    func_name = name
    key_var_num_args = py_str(key_var_num_args.value)
    ret_type = py_str(ret_type.value) if ret_type.value is not None else ''
    doc_str = _build_doc(func_name,
                         py_str(desc.value),
                         [py_str(arg_names[i]) for i in range(narg)],
                         [py_str(arg_types[i]) for i in range(narg)],
                         [py_str(arg_descs[i]) for i in range(narg)],
                         key_var_num_args,
                         ret_type)
    arguments = []
    for i in range(num_args.value):
        dtype = py_str(arg_types[i])
        if not (dtype.startswith('NDArray') or
                dtype.startswith('Symbol') or
                dtype.startswith('ndarray-or-symbol')):
            arguments.append(py_str(arg_names[i]))

    # Definition of internal functions.
    def generic_ndarray_function(*args, **kwargs):
        """Invoke this function by passing in parameters

        Parameters
        ----------
        *args
            Positional arguments of input scalars and NDArray
        out : NDArray or tuple of NDArray, optional
            Output NDArray, used to hold the output result.

        Returns
        -------
        out : NDArray
            The result NDArray(tuple) of result of computation.
        """
        ndargs = []
        pos_args = []
        for i in args:
            if isinstance(i, NDArrayBase):
                ndargs.append(i)
            else:
                pos_args.append(str(i))

        if len(pos_args) > len(arguments):
            raise ValueError("Too many positional arguments")
        kwargs.update(zip(arguments[:len(pos_args)], pos_args))
        if 'dtype' in kwargs:
            kwargs['dtype'] = np.dtype(kwargs['dtype']).name

        original_output = None
        if 'out' in kwargs:
            output_vars = kwargs['out']
            original_output = output_vars
            del kwargs['out']
            if isinstance(output_vars, NDArrayBase):
                output_vars = (output_vars,)
            num_output = ctypes.c_int(len(output_vars))
            output_vars = c_array(NDArrayHandle, [v.handle for v in output_vars])
            output_vars = ctypes.cast(output_vars, ctypes.POINTER(NDArrayHandle))
        else:
            output_vars = ctypes.POINTER(NDArrayHandle)()
            num_output = ctypes.c_int(0)

        check_call(_LIB.MXImperativeInvoke(
            handle,
            ctypes.c_int(len(ndargs)),
            c_array(NDArrayHandle, [i.handle for i in ndargs]),
            ctypes.byref(num_output),
            ctypes.byref(output_vars),
            ctypes.c_int(len(kwargs)),
            c_array(ctypes.c_char_p, [c_str(key) for key in kwargs.keys()]),
            c_array(ctypes.c_char_p, [c_str(str(i)) for i in kwargs.values()])))
        if original_output is not None:
            return original_output
        ret_list = []
        for i in xrange(num_output.value):
            chunk_type = ctypes.c_int(0)
            check_call(_LIB.MXNDArrayGetChunkType(ctypes.cast(output_vars[i], NDArrayHandle), ctypes.byref(chunk_type)))
            ret_list.append(_ndarray_cls_map[chunk_type.value](ctypes.cast(output_vars[i], NDArrayHandle)))
        if num_output.value == 1:
            return ret_list[0]
        return ret_list

    # End of function declaration
    # All operators are registered under mxnet.ndarray namespace
    generic_ndarray_function.__name__ = func_name
    generic_ndarray_function.__doc__ = doc_str
    generic_ndarray_function.__module__ = 'mxnet.ndarray'
    return generic_ndarray_function


def _set_chunk_nd_map(chunk_nd_map):
    """Set the symbolic class to be cls"""
    global _ndarray_cls_map 
    _ndarray_cls_map = chunk_nd_map


# pylint: enable=too-many-locals, invalid-name
def _init_ndarray_module(chunk_nd_map, root_namespace):
    """List and add all the ndarray functions to current module."""
    _set_chunk_nd_map(chunk_nd_map)
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    module_obj = _sys.modules["%s.ndarray" % root_namespace]
    module_internal = _sys.modules["%s._ndarray_internal" % root_namespace]
    for name in op_names:
        hdl = OpHandle()
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        function = _make_ndarray_function(hdl, name)
        if function.__name__.startswith('_'):
            setattr(module_internal, function.__name__, function)
        else:
            setattr(module_obj, function.__name__, function)
