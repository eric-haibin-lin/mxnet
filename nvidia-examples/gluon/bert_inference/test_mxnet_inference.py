"""
SQuAD with Bidirectional Encoder Representations from Transformers

=========================================================================================

Example of inference using BERT base model

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
from export import hybrid_bert

np.random.seed(6)
random.seed(6)
mx.random.seed(6)

send_lib = ctypes.CDLL('./libsend_data.so')

log = logging.getLogger('gluonnlp')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(levelname)s:%(name)s:%(asctime)s %(message)s', datefmt='%H:%M:%S')


parser = argparse.ArgumentParser(description='BERT QA inference example.')

parser.add_argument('--dtype',
                    type=str,
                    help='Type to use. Currently only float16 is supported.')

parser.add_argument('--bert_model',
                    type=str,
                    default='bert_12_768_12',
                    help='BERT model name. Currently only bert_12_768_12 is supported.')

parser.add_argument('--uncased',
                    action='store_false',
                    help='if not set, inputs are converted to lower case.')

parser.add_argument('--test_batch_size',
                    type=int,
                    default=24,
                    help='Test batch size. default is 24')

parser.add_argument('--log_interval',
                    type=int,
                    default=50,
                    help='report interval. default is 50')

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

parser.add_argument('--n_best_size',
                    type=int,
                    default=20,
                    help='The total number of n-best predictions to generate in the '
                    'nbest_predictions.json output file. default is 20')

parser.add_argument('--max_answer_length',
                    type=int,
                    default=30,
                    help='The maximum length of an answer that can be generated. This is needed '
                    'because the start and end predictions are not conditioned on one another.'
                    ' default is 30')

parser.add_argument('--null_score_diff_threshold',
                    type=float,
                    default=0.0,
                    help='If null_score - best_non_null is greater than the threshold predict null.'
                    'Typical values are between -1.0 and -5.0. default is 0.0')

parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help='which gpu to use for finetuning. GPU(0) is used if not set.')

parser.add_argument('--sentencepiece',
                    type=str,
                    default=None,
                    help='Path to the sentencepiece .model file for both tokenization and vocab.')

parser.add_argument('--profile',
                    action='store_true',
                    help='Run 50 iterations for profiling')


args = parser.parse_args()

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
log.addHandler(console)

log.info(args)

model_name = args.bert_model
dataset_name = 'book_corpus_wiki_en_uncased'
lower = args.uncased

test_batch_size = args.test_batch_size
ctx = mx.gpu(args.gpu)

log_interval = args.log_interval

null_score_diff_threshold = args.null_score_diff_threshold

max_seq_length = args.max_seq_length
doc_stride = args.doc_stride
max_query_length = args.max_query_length
n_best_size = args.n_best_size
max_answer_length = args.max_answer_length

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

use_fp16 = args.dtype == 'float16'
if not use_fp16:
    raise ValueError("Currently only float16 is supported.")

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

if use_fp16:
    sym, arg_params, aux_params = mx.model.load_checkpoint('fp16_bert_'+str(max_seq_length), 0)
else:
    sym, arg_params, aux_params = mx.model.load_checkpoint('fp32_bert_'+str(max_seq_length), 0)
executor = sym.simple_bind(ctx=ctx, data0=(max_seq_length, test_batch_size), data1=(max_seq_length, test_batch_size), data2=(test_batch_size,), type_dict={'data0' : 'float32', 'data1' : 'float32', 'data2' : 'float32'}, grad_req='null')
executor.copy_params_from(arg_params, aux_params)
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

def get_MXNet_ptr(arr):
    handle = arr.handle
    ptr = ctypes.c_void_p()
    mx.base._LIB.MXNDArrayGetData(arr.handle, ctypes.byref(ptr))
    return ptr

def send_data(ptr_cpu, ptr_gpu, nbytes):
    send_lib.send_data(ptr_cpu, ptr_gpu, ctypes.c_size_t(nbytes))

def recv_data(arr_gpu, arr_cpu, nbytes, sync=False):
    if sync:
        send_lib.recv_data_sync(arr_cpu, arr_gpu, ctypes.c_size_t(nbytes))
    else:
        send_lib.recv_data_async(arr_cpu, arr_gpu, ctypes.c_size_t(nbytes))

def recv_data_from_GPU(start_gpu, end_gpu, start_cpu, end_cpu, nbytes):
    recv_data(start_gpu, start_cpu, nbytes)
    recv_data(end_gpu, end_cpu, nbytes, sync=True)

def evaluate():
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

    log.info('start prediction')

    all_results = collections.defaultdict(list)

    warmup=10
    total_iters=0
    total_time=0.0
    total_len = 0

    # get input pointers
    inputs_GPU = executor.arg_dict['data0']
    inputs_GPU_ptr = get_MXNet_ptr(inputs_GPU)
    token_types_GPU = executor.arg_dict['data1']
    token_types_GPU_ptr = get_MXNet_ptr(token_types_GPU)
    valid_length_GPU = executor.arg_dict['data2']
    valid_length_GPU_ptr = get_MXNet_ptr(valid_length_GPU)
    in_nbytes = 4 * inputs_GPU.shape[0] * inputs_GPU.shape[1]
    vl_nbytes = 4 * valid_length_GPU.shape[0]

    # get output pointers
    out1 = executor.outputs[0]
    out1_ptr = get_MXNet_ptr(out1)
    out2 = executor.outputs[1]
    out2_ptr = get_MXNet_ptr(out2)
    if use_fp16:
        out_nbytes = 2 * out1.shape[0] * out1.shape[1]
    else:
        out_nbytes = 4 * out1.shape[0] * out1.shape[1]

    if use_fp16:
        pred_start = np.zeros((test_batch_size, max_seq_length), dtype=np.float16)
        pred_end = np.zeros((test_batch_size, max_seq_length), dtype=np.float16)
    else:
        raise ValueError
        pred_start = np.zeros((test_batch_size, max_seq_length), dtype=np.float32)
        pred_end = np.zeros((test_batch_size, max_seq_length), dtype=np.float32)

    start_ptr = pred_start.ctypes.data_as(ctypes.c_void_p)
    end_ptr = pred_end.ctypes.data_as(ctypes.c_void_p)

    for counter, data in enumerate(dev_dataloader):
        example_ids, inputs, token_types, valid_length, _, _ = data
        # In order to minimize Python overheads
        # that skew the timing results,
        # we take pointers from NDArrays outside
        # the timing region
        inputs_ptr = get_MXNet_ptr(inputs)
        token_types_ptr = get_MXNet_ptr(token_types)
        valid_length_ptr = get_MXNet_ptr(valid_length)
        mx.nd.waitall()
        tic = time.time()

        # Send data to the GPU using stream 0 in order to avoid
        # cudaStreamSynchronize call overhead that would happen
        # when naively copying the data to GPU via MXNet
        send_data(inputs_ptr, inputs_GPU_ptr, in_nbytes)
        send_data(token_types_ptr, token_types_GPU_ptr, in_nbytes)
        send_data(valid_length_ptr, valid_length_GPU_ptr, vl_nbytes)
        executor.forward()
        mx.nd.waitall()
        recv_data_from_GPU(out1_ptr, out2_ptr, start_ptr, end_ptr, out_nbytes)

        toc = time.time()

        example_ids = example_ids.asnumpy()
        example_ids = example_ids.tolist()
        for example_id, start, end in zip(example_ids, pred_start, pred_end):
            all_results[example_id].append(PredResult(start=np.copy(start), end=np.copy(end)))

        if not warmup:
            total_time = total_time + (toc-tic)
            total_iters=total_iters+1
            total_len += inputs.shape[1]  # inputs = (seq_len, batch_size)
        else:
            warmup=warmup-1
        if args.profile:
            if counter == 50:
                break

    log.info('Time cost={:.2f} s, Throughput={:.2f} samples/s'.format(
        total_time, total_len/total_time))
    log.info('Average Latency using BatchSize {}: {:.4f} ms ...after {} iterations'.format(
        test_batch_size,(total_time/total_iters)*1000,total_iters))
    log.info('Get prediction results...')

    all_predictions = collections.OrderedDict()

    for features in dev_dataset:
        results = all_results[features[0].example_id]
        example_qas_id = features[0].qas_id

        prediction, _ = predict(
            features=features,
            results=results,
            tokenizer=nlp.data.BERTBasicTokenizer(lower=lower),
            max_answer_length=max_answer_length,
            null_score_diff_threshold=null_score_diff_threshold,
            n_best_size=n_best_size,
            version_2=False)

        all_predictions[example_qas_id] = prediction

    with io.open('predictions.json', 'w', encoding='utf-8') as fout:
        data = json.dumps(all_predictions, ensure_ascii=False)
        fout.write(data)

    F1_EM = get_F1_EM(dev_data, all_predictions)
    log.info(F1_EM)


if __name__ == '__main__':
    evaluate()
