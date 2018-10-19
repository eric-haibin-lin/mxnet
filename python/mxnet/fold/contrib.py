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
# pylint: disable=wildcard-import, unused-wildcard-import,redefined-outer-name
"""Contrib NDArray API of MXNet."""
import sys
from .fold import _current_batching_scope, get_num_outputs, create_ndarray_future

__all__ = ["foreach"]

def foreach(*args, **kwargs):
    batching = _current_batching_scope()
    num_outputs = get_num_outputs('_contrib_foreach', args, kwargs)
    futures = tuple([create_ndarray_future() for _ in range(num_outputs)])
    batching.record('_contrib_foreach', futures, args, kwargs)
    if num_outputs == 1:
        return futures[0]
    return futures
