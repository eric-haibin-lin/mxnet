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

"""Register backend ops in mxnet.ndarray namespace"""
from ..base import _init_op_module
from .fold import _current_batching_scope, get_num_outputs, create_ndarray_future

def _make_op_func(hdl, name, func_name):
    """Create a NDArray function from the FunctionHandle."""
    code = """
def {0}(*args, **kwargs):
    batching = _current_batching_scope()
    num_outputs = get_num_outputs("{1}", args, kwargs)
    futures = tuple([create_ndarray_future() for _ in range(num_outputs)])
    batching.record("{1}", futures, args, kwargs)
    if num_outputs == 1:
        return futures[0]
    return futures
    """.format(func_name, name)
    doc_str = ""

    local = {}
    exec(code, None, local)  # pylint: disable=exec-used
    op_function = local[func_name]
    op_function.__name__ = func_name
    op_function.__doc__ = doc_str
    # op_function.__module__ = 'mxnet.fold.op'
    return op_function


_init_op_module('mxnet', 'fold', _make_op_func)
