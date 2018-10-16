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

# coding: utf-8
# pylint: disable=
"""fold for dynamic batching."""
__all__ = ['batching', 'is_batching', 'NDArrayFuture', 'create_ndarray_future']

import numpy as np
import ctypes
import random
import sys
from collections import namedtuple, defaultdict

from ... import ndarray
from ...base import check_call, _LIB, py_str
from ...ndarray import NDArray

# TODO
# - improve ndarray future
# - op signature
# - infer batch axis
# - multiple outputs
# - optimize batch algorithm

_OpRecord = namedtuple('OpRecord', ['op_name', 'future', 'inputs', 'attrs'])

def _flatten(args):
    if isinstance(args, (list, tuple)):
        flat = []
        fmts = []
        for i in args:
            arg, fmt = _flatten(i)
            flat.extend(arg)
            fmts.append(fmt)
        return tuple(flat), tuple(fmts)
    else:
        return (args,), int(0)



class NDArrayFutureManager(object):
    def __init__(self):
        self.table = {}

    def instantiate(self, future, arr):
        assert future.key not in self.table
        self.table[future.key] = arr

    def look(self, future):
        assert future.key in self.table
        return self.table[future.key]

_CurrentFutureManager = NDArrayFutureManager()

def get_future_manager():
    return _CurrentFutureManager

def get_output_futures(op_name):
    num_outputs = 1
    if op_name == 'split':
        num_outputs = 2
    if num_outputs > 1:
        futures = tuple([create_ndarray_future() for i in range(num_outputs)])
    else:
        futures = create_ndarray_future()
    return futures


class NDArrayFuture(NDArray):
    allowed = ['__init__', 'key', 'instantiated', 'manager',
               'instantiate', 'handle', 'writable']

    def __init__(self):
        NDArray.__init__(self, None, True)
        self.key = random.randint(0, sys.maxint)
        self.instantiated = False
        self.manager = get_future_manager()

    def __del__(self):
        # do nothing to avoid double free
        pass

    def __repr__(self):
        if self.instantiated:
            return NDArray.__repr__(self)
        else:
            return '<NDArrayFuture (uninstantiated) @ {0}>'.format(self.key)

    def instantiate(self, arr):
        assert isinstance(arr, NDArray)
        self.manager.instantiate(self, arr)
        NDArray.__init__(self, arr.handle, True)
        self.instantiated = True

    def __getattribute__(self, attr):
        if attr in NDArrayFuture.allowed:
            return NDArray.__getattribute__(self, attr)

        if not self.instantiated:
            try:
                arr = self.manager.look(self)
            except KeyError:
                raise ValueError, "attribute '{0}' is not allowed for " \
                    "uninstantiated ndarray future.".format(attr)
            NDArray.__init__(self, arr.handle, True)
            self.instantiated = True
            return NDArray.__getattribute__(self, attr)
        else:
            return NDArray.__getattribute__(self, attr)

def create_ndarray_future(arr=None):
    future = NDArrayFuture()
    if arr is not None:
        future.instantiate(arr)
    return future

import collections
OpSig = collections.namedtuple('OpSig', ['op', 'fmt', 'batch_axis'])

def calculate_signature(op_name, args, kwargs):
    batch_axis = 0
    flat_args, fmt = _flatten((args, kwargs))
    op_sig = OpSig(op_name, fmt, batch_axis)
    return hash(op_sig) # MAGIC!


# algorithm part for dynamic batching
class Fold(object):
    def __init__(self):
        self.max_step = 0
        self.steps = defaultdict(lambda: defaultdict(list))
        self.depths = {}
        self.exec_list = []

    def record(self, op_name, future, inputs, attrs):
        print('record {0}'.format(op_name))
        # setup input depth
        for arg in inputs:
            if arg not in self.depths:
                # NDArray or instantiated NDArrayFuture
                if isinstance(arg, NDArrayFuture):
                    assert arg.instantiated
                self.depths[arg] = 0

        step = max([self.depths[arg] for arg in inputs]) + 1
        self.max_step = max(step, self.max_step)
        self.depths[future] = step

        op_sig = calculate_signature(op_name, inputs, attrs)
        self.steps[step][op_sig].append(_OpRecord(op_name, future, inputs, attrs))

    def _concat_inputs(self, inputs):
        concat_attrs = {'dim': 0}
        concat_out = create_ndarray_future()
        concat_op = _OpRecord('concat', concat_out, inputs, concat_attrs)
        return concat_out, concat_op

    def _split_output(self, new_op_out, futures, concat_arrs):
        split_attrs = {'concat_arrs': concat_arrs, 'futures': futures, 'inputs': [new_op_out]}
        # placeholder for split
        split_op = _OpRecord('deferred_split', None, None, split_attrs)
        return split_op

    def batch(self):
        print('max step: {0}'.format(self.max_step))
        for step in range(self.max_step + 1):
            for op_sig in self.steps[step]:
                old_ops = self.steps[step][op_sig]
                op_name = old_ops[0].op_name
                futures = [op.future for op in old_ops]
                inputs = [op.inputs[0] for op in old_ops]
                params = old_ops[0].inputs[1:]
                attrs = old_ops[0].attrs

                concat_out, concat_op = self._concat_inputs(inputs)
                self.exec_list.append(concat_op)

                new_op_in = (concat_out, ) + params
                # possibly multiple outputs
                new_op_out = get_output_futures(op_name)
                new_op = _OpRecord(op_name, new_op_out, new_op_in, attrs)
                self.exec_list.append(new_op)

                deferred_split_op = self._split_output(new_op_out, futures, inputs)
                self.exec_list.append(deferred_split_op)

    def execute(self):
        def execute_record(record):
            print('executing %s'%record.op_name)
            op = getattr(ndarray, record.op_name)
            out = op(*record.inputs, **record.attrs)
            if isinstance(out, (list, tuple)):
                num_outs = len(out)
                assert len(out) == len(record.future)
                for i in range(num_outs):
                    record.future[i].instantiate(out[i])
            else:
                record.future.instantiate(out)
            print('done %s'%record.op_name)

        for record in self.exec_list:
            # handle deferred_split
            if record.op_name == "deferred_split":
                concat_arrs = record.attrs['concat_arrs']
                futures = record.attrs['futures']
                inputs = record.attrs['inputs']

                indices = [0]
                acc = 0
                for num_batch in [arr.shape[0] for arr in concat_arrs]:
                    acc += num_batch
                    indices.append(acc)
                split_ops = []
                for i in range(1, len(indices)):
                    split_attrs = {'axis': 0, 'begin': indices[i - 1], 'end': indices[i]}
                    num_outputs = len(futures[i - 1]) if isinstance(futures[i - 1], (tuple, list)) else 1
                    if num_outputs == 1:
                        # futures[i - 1] stores the output future for (i-1)th sample
                        split_op = _OpRecord('slice_axis', futures[i - 1], inputs, split_attrs)
                        execute_record(split_op)
                    else:
                        assert len(futures[i - 1]) == len(inputs[0])
                        for j in range(num_outputs):
                            split_op = _OpRecord('slice_axis', futures[i - 1][j], (inputs[0][j],), split_attrs)
                            execute_record(split_op)
                break

            execute_record(record)

    def clear(self):
        pass


_CurrentBatchingScope = None

def _current_batching_scope():
    return _CurrentBatchingScope

def _set_current_batching_scope(scope):
    global _CurrentBatchingScope
    _CurrentBatchingScope = scope

def batching(batch_size=64):
    return _BatchingScope(batch_size=batch_size)

def is_batching():
    return (_current_batching_scope() is not None)

class _BatchingScope(object):
    """batching scope manager """
    def __init__(self, batch_size):
        self._fold = Fold()
        self._batch_size = batch_size

    def __enter__(self):
        if _current_batching_scope() is not None:
            raise ValueError, "nested batching scope is not allowed"
        _set_current_batching_scope(self)

    def __exit__(self, ptype, value, trace):
        # real execution happens
        # compute all NDArrayFuture
        self._fold.batch()
        self._fold.execute()
        self._fold.clear()
        _set_current_batching_scope(None)

    def record(self, op, out, args, kwargs):
        self._fold.record(op, out, args, kwargs)

def _make_op_func(name, func_name):
    """Create a NDArray function from the FunctionHandle."""
    code = """
def {0}(*args, **kwargs):
    batching = _current_batching_scope()
    op_name = sys._getframe().f_code.co_name
    futures = get_output_futures(op_name)
    batching.record(op_name, futures, args, kwargs)
    return futures
    """.format(func_name)
    doc_str = ""

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    op_function = local[func_name]
    op_function.__name__ = func_name
    op_function.__doc__ = doc_str
    op_function.__module__ = 'mxnet.gluon.fold.op'
    return op_function

def _init_ops():
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.MXListAllOpNames(ctypes.byref(size),
                                     ctypes.byref(plist)))
    op_names = []
    for i in range(size.value):
        op_names.append(py_str(plist[i]))

    for name in op_names:
        func_name = name
        function = _make_op_func(name, func_name)
        cur_module = sys.modules['mxnet.gluon.fold.op']
        function.__module__ = cur_module
        setattr(cur_module, function.__name__, function)
        cur_module.__all__.append(function.__name__)

_init_ops()
