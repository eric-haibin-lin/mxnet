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

import os, gzip, sys
import mxnet as mx
import numpy as np
import data_utils

class MultiSentenceIter(mx.io.DataIter):
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, data_file, vocab, batch_size, bptt):
        super(MultiSentenceIter, self).__init__()
        self.batch_size = batch_size
        self.bptt = bptt
        self.provide_data = [('data', (batch_size, bptt), np.int32), ('mask', (batch_size, bptt))]
        self.provide_label = [('label', (batch_size, bptt))]
        self.vocab = vocab
        self.data_file = data_file
        self._dataset = data_utils.Dataset(self.vocab, data_file, deterministic=True)
        self._iter = self._dataset.iterate_once(batch_size, bptt)

    def iter_next(self):
        data = self._iter.next()
        if data is None:
            return False
        self._next_data = mx.nd.array(data[0], dtype=np.int32)
        self._next_label = mx.nd.array(data[1])
        self._next_mask = mx.nd.array(data[2])
        self._next_mask[:] = 1
        return True

    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:
            raise StopIteration

    def reset(self):
        print('reset')
        self._dataset = data_utils.Dataset(self.vocab, self.data_file, deterministic=False)
        self._iter = self._dataset.iterate_once(self.batch_size, self.bptt)
        self._next_data = None
        self._next_label = None
        self._next_mask = None

    def getdata(self):
        return [self._next_data, self._next_mask]

    def getlabel(self):
        return [self._next_label]
