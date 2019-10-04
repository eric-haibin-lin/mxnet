"""
SQuAD with Bidirectional Encoder Representations from Transformers

=========================================================================================

BERT base model exporter

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming- \
      Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

# coding=utf-8

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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import argparse
import collections
import json
import logging
import os
import io
import copy
import random
import time
import warnings
import ctypes

os.environ["MXNET_GPU_WORKER_NTHREADS"] = "1"
os.environ["MXNET_COPY_WORKER_NTHREADS"] = "1"

import numpy as np
import mxnet as mx

import gluonnlp as nlp
from gluonnlp.data import SQuAD
from data.qa import SQuADTransform, preprocess_dataset
from bert_qa_evaluate import get_F1_EM, predict, PredResult
from export import hybrid_bert, hybrid_bert_old

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

send_lib = ctypes.CDLL('./libsend_data.so')

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


parser = argparse.ArgumentParser(description='BERT QA model exporter for inference.')

parser.add_argument('--dtype',
                    type=str,
                    help='Type to use. Currently only float16 is supported.')

parser.add_argument('--model_parameters',
                    type=str,
                    default=None,
                    help='Model parameter file')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. Currently only bert_12_768_12 is supported.')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--max_seq_length',
                    type=int,
                    default=384,
                    help='The maximum total input sequence length after WordPiece tokenization.'
                    'Sequences longer than this will be truncated, and sequences shorter '
                    'than this will be padded. default is 384')

parser.add_argument('--doc_stride',
                    type=int,
                    default=128,
                    help='When splitting up a long document into chunks, how much stride to '
                    'take between chunks. default is 128')

parser.add_argument('--max_query_length',
                    type=int,
                    default=64,
                    help='The maximum number of tokens for the question. Questions longer than '
                    'this will be truncated to this length. default is 64')

parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help='which gpu to use. GPU(0) is used if not set.')

parser.add_argument('--sentencepiece',
                    type=str,
                    default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')

args = parser.parse_args()

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)

log.info(args)

model_name = args.bert_model
dataset_name = 'book_corpus_wiki_en_uncased'
model_parameters = args.model_parameters
lower = args.uncased

test_batch_size = 1
ctx = mx.gpu(args.gpu)

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length

if max_seq_length <= max_query_length + 3:
    raise ValueError('The max_seq_length (%d) must be greater than max_query_length '
                     '(%d) + 3' % (max_seq_length, max_query_length))

# vocabulary and tokenizer
if args.sentencepiece:
    logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
    if dataset_name:
        warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                      'The vocabulary will be loaded based on --sentencepiece.')
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)
    dataset_name = None
else:
    vocab = None

use_fp16 = (args.dtype == 'float16')
if not use_fp16:
    raise ValueError("Currently only float16 is supported.")

hybrid_bert_old.init_fast_softmax()
orig_bert, _ = hybrid_bert_old.get_hybrid_model(
    name=model_name,
    dataset_name=dataset_name,
    vocab=vocab,
    pretrained=False,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
    seq_length=args.max_seq_length,
    use_FP16=use_fp16)
orig_net = hybrid_bert_old.HybridBERTForQA(bert=orig_bert, use_FP16=use_fp16)
if model_parameters:
    nlp.utils.load_parameters(orig_net, model_parameters, ctx=ctx, cast_dtype=None)

hybrid_bert.init_fast_multiheadattn_and_softmax()
bert, vocab = hybrid_bert.get_hybrid_model(
    name=model_name,
    dataset_name=dataset_name,
    vocab=vocab,
    pretrained=False,
    ctx=ctx,
    use_pooler=False,
    use_decoder=False,
    use_classifier=False,
    seq_length=args.max_seq_length,
    use_FP16=use_fp16)


def convert_arg_params(net_arg_params, loaded_arg_params, num_heads):
    for k, v in net_arg_params.items():
        k = k.replace('hybridbertencoder1', 'hybridbertencoder0')
        k = k.replace('hybridbertmodel1', 'hybridbertmodel0')
        k = k.replace('hybridbertforqa1', 'hybridbertforqa0')
        if k.endswith('proj_inweight'):
            assert k[:-13] + 'query_weight' in loaded_arg_params
            assert k[:-13] + 'key_weight' in loaded_arg_params
            assert k[:-13] + 'value_weight' in loaded_arg_params
            q_weight = loaded_arg_params[k[:-13] + 'query_weight'].data().reshape(shape=(num_heads, -1, 0),
                                                                                  reverse=True)
            k_weight   = loaded_arg_params[k[:-13] + 'key_weight'].data().reshape(shape=(num_heads, -1, 0),
                                                                                  reverse=True)
            v_weight = loaded_arg_params[k[:-13] + 'value_weight'].data().reshape(shape=(num_heads, -1, 0),
                                                                                  reverse=True)
            all_weight = mx.nd.concat(q_weight, k_weight, v_weight, dim=-2)
            all_weight = mx.nd.reshape(all_weight, shape=(-1, 0), reverse=True)
            v.set_data(all_weight.astype(v.dtype))
        elif k.endswith('proj_inbias'):
            assert k[:-11] + 'query_bias' in loaded_arg_params
            assert k[:-11] + 'key_bias' in loaded_arg_params
            assert k[:-11] + 'value_bias' in loaded_arg_params
            q_bias = loaded_arg_params[k[:-11] + 'query_bias'].data().reshape(shape=(num_heads, -1),
                                                                              reverse=True)
            k_bias   = loaded_arg_params[k[:-11] + 'key_bias'].data().reshape(shape=(num_heads, -1),
                                                                              reverse=True)
            v_bias = loaded_arg_params[k[:-11] + 'value_bias'].data().reshape(shape=(num_heads, -1),
                                                                              reverse=True)
            all_bias = mx.nd.stack(q_bias, k_bias, v_bias, axis=1)
            all_bias = mx.nd.reshape(all_bias, shape=(-1))
            v.set_data(all_bias.astype(v.dtype))
        else:
            v.set_data(loaded_arg_params[k].data().astype(v.dtype))

net = hybrid_bert.HybridBERTForQA(bert=bert, use_FP16=use_fp16)
net.initialize(mx.init.Xavier(), ctx=ctx)
net.hybridize(static_alloc=True, static_shape=True)
if use_fp16:
    net.cast('float16')
if model_parameters:
    convert_arg_params(net.collect_params(), orig_net.collect_params(), 12)
if args.sentencepiece:
    tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, lower=lower)
else:
    tokenizer = nlp.data.BERTTokenizer(vocab=vocab, lower=lower)

def _transposed_pad_arrs_to_max_length(arrs, pad_axis, pad_val, use_shared_mem, dtype):
    if isinstance(arrs[0], mx.nd.NDArray):
        dtype = arrs[0].dtype if dtype is None else dtype
        arrs = [arr.asnumpy() for arr in arrs]
    elif not isinstance(arrs[0], np.ndarray):
        arrs = [np.asarray(ele) for ele in arrs]
    else:
        dtype = arrs[0].dtype if dtype is None else dtype

    original_length = [ele.shape[pad_axis] for ele in arrs]
    max_size = max(original_length)

    ret_shape = list(arrs[0].shape)
    ret_shape[pad_axis] = max_size
    ret_shape = (ret_shape[0], len(arrs)) + tuple(ret_shape[1:])
    ret = np.full(shape=ret_shape, fill_value=pad_val, dtype=dtype)
    for i, arr in enumerate(arrs):
        if arr.shape[pad_axis] == max_size:
            ret[:,i] = arr
        else:
            slices = [slice(None) for _ in range(arr.ndim)]
            slices[pad_axis] = slice(0, arr.shape[pad_axis])
            if slices[pad_axis].start != slices[pad_axis].stop:
                slices = [slice(i, i + 1)] + slices
                ret[:, tuple(slices)] = arr
    ctx = mx.Context('cpu_shared', 0) if use_shared_mem else mx.cpu()
    ret = mx.nd.array(ret, ctx=ctx, dtype=dtype)
    original_length = mx.nd.array(original_length, ctx=ctx, dtype=np.int32)

    return ret, original_length

class TransposedPad(object):
    def __init__(self, axis=0, pad_val=None, ret_length=False, dtype=None):
        self._axis = axis
        assert isinstance(axis, int), 'axis must be an integer! ' \
                                      'Received axis=%s, type=%s.' % (str(axis),
                                                                      str(type(axis)))
        self._pad_val = 0 if pad_val is None else pad_val
        self._ret_length = ret_length
        self._dtype = dtype
        self._warned = False
        if pad_val is None:
            warnings.warn("padding value is not given and will be set automatically to 0")

    def __call__(self, data):
        if isinstance(data[0], mx.nd.NDArray) and not self._warned:
            self._warned = True
            warnings.warn("Using Pad with NDArrays is discouraged for speed reasons...")

        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _transposed_pad_arrs_to_max_length(data, self._axis,
                                                                  self._pad_val, True,
                                                                  self._dtype)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr
        else:
            raise NotImplementedError

batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Stack(),
    TransposedPad(axis=0, pad_val=vocab[vocab.padding_token]),
    TransposedPad(axis=0, pad_val=vocab[vocab.padding_token]),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'),
    nlp.data.batchify.Stack('float32'))

def send_data(arr_cpu, arr_gpu):
    cpu_handle = arr_cpu.handle
    gpu_handle = arr_gpu.handle
    if len(arr_cpu.shape) == 1:
        nbytes = 4 * arr_cpu.shape[0]
    elif len(arr_cpu.shape) == 2:
        nbytes = 4 * arr_cpu.shape[0] * arr_cpu.shape[1]
    ptr_cpu = ctypes.c_void_p()
    ptr_gpu = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr_cpu.handle, ctypes.byref(ptr_cpu))
    mx.base._LIB.MXNDArrayGetData(arr_gpu.handle, ctypes.byref(ptr_gpu))
    send_lib.send_data(ptr_cpu, ptr_gpu, ctypes.c_size_t(nbytes))

def send_data_to_GPU(input_cpu, token_types_cpu, valid_length_cpu, input_gpu, token_types_gpu, valid_length_gpu):
    # Send data to the GPU using stream 0 in order to avoid
    # cudaStreamSynchronize call overhead that would happen
    # when naively copying the data to GPU via MXNet
    send_data(input_cpu, input_gpu)
    send_data(token_types_cpu, token_types_gpu)
    send_data(valid_length_cpu, valid_length_gpu)

def recv_data(arr_gpu, arr_cpu, nbytes, sync=False):
    gpu_handle = arr_gpu.handle
    ptr_gpu = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr_gpu.handle, ctypes.byref(ptr_gpu))
    if sync:
        send_lib.recv_data_sync(arr_cpu, ptr_gpu, ctypes.c_size_t(nbytes))
    else:
        send_lib.recv_data_async(arr_cpu, ptr_gpu, ctypes.c_size_t(nbytes))

def recv_data_from_GPU(start_gpu, end_gpu, start_cpu, end_cpu):
    if use_fp16:
        nbytes = 2 * start_gpu.shape[0] * start_gpu.shape[1]
    else:
        nbytes = 4 * start_gpu.shape[0] * start_gpu.shape[1]
    recv_data(start_gpu, start_cpu, nbytes)
    recv_data(end_gpu, end_cpu, nbytes, sync=True)

def export():
    """Evaluate the model on validation dataset.
    """
    log.info('Loading dev data...')
    dev_data = SQuAD('dev', version='1.1')
    log.info('Number of records in dev data:{}'.format(len(dev_data)))

    dev_dataset = dev_data.transform(
        SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=False,
            is_training=False)._transform, lazy=False)

    dev_data_transform, _ = preprocess_dataset(
        dev_data, SQuADTransform(
            copy.copy(tokenizer),
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            is_pad=True,
            is_training=False))
    log.info('The number of examples after preprocessing:{}'.format(
        len(dev_data_transform)))

    dev_dataloader = mx.gluon.data.DataLoader(
        dev_data_transform,
        batchify_fn=batchify_fn,
        num_workers=4, batch_size=test_batch_size,
        shuffle=False, last_batch='discard')

    total_iters=0
    total_time=0.0
    total_len = 0

    inputs_GPU = mx.nd.zeros((max_seq_length, test_batch_size), ctx=ctx)
    token_types_GPU = mx.nd.zeros((max_seq_length, test_batch_size), ctx=ctx)
    valid_length_GPU = mx.nd.zeros((test_batch_size,), ctx=ctx)

    if use_fp16:
        pred_start = np.zeros((test_batch_size, max_seq_length), dtype=np.float16)
        pred_end = np.zeros((test_batch_size, max_seq_length), dtype=np.float16)
    else:
        raise ValueError
        pred_start = np.zeros((test_batch_size, max_seq_length), dtype=np.float32)
        pred_end = np.zeros((test_batch_size, max_seq_length), dtype=np.float32)

    start_ptr = pred_start.ctypes.data_as(ctypes.c_void_p)
    end_ptr = pred_end.ctypes.data_as(ctypes.c_void_p)

    for data in dev_dataloader:
        example_ids, inputs, token_types, valid_length, _, _ = data
        mx.nd.waitall()
        tic = time.time()

        send_data_to_GPU(inputs, token_types, valid_length, inputs_GPU, token_types_GPU, valid_length_GPU)
        out1, out2 = net(inputs_GPU,
                         token_types_GPU,
                         valid_length_GPU)
        mx.nd.waitall()
        recv_data_from_GPU(out1, out2, start_ptr, end_ptr)

        toc = time.time()

        net.export('fp16_bert_' + str(max_seq_length), 0)
        break

if __name__ == '__main__':
    if model_parameters:
        export()
    else:
        raise RuntimeError("Need to provide model_parameters option")
