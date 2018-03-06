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

import numpy as np
import mxnet as mx
import mxnet.symbol as S
import run_utils
from data import MultiSentenceIter, Vocabulary
from model import *
from sparse_module import CustomModule
import os, math, logging, sys

if __name__ == '__main__':
    parser = run_utils.get_parser()
    args = parser.parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.INFO, format=head)
    logging.info(args)
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')] if args.gpus else [mx.gpu()]
    ngpus = len(ctx)
    nsamples = args.k
    rescale_loss = args.bptt
    logging.debug(sys.argv)

    # data
    vocab = Vocabulary.from_file(args.vocab)
    ntokens = vocab.num_tokens
    train_data = mx.io.PrefetchingIter(MultiSentenceIter(args.data, vocab,
                                       args.batch_size * ngpus, args.bptt))
    # rnn model
    rnn_out, last_states, lstm_args, state_names = rnn(args.bptt, ntokens, args.emsize,
                                                       args.nhid, args.nlayers, args.dropout,
                                                       args.num_proj, args.batch_size)
    # decoder weight and bias
    decoder_w = mx.sym.var("decoder_weight", stype='row_sparse')
    decoder_b = mx.sym.var("decoder_bias", shape=(ntokens, 1))

    # sampled softmax for training
    sample = S.var('sample', shape=(nsamples,))
    prob_sample = S.var("prob_sample", shape=(nsamples,))
    prob_target = S.var("prob_target")
    logits, new_targets = sampled_softmax(ntokens, args.k, args.num_proj,
                                          rnn_out, decoder_w, decoder_b,
                                          [sample, prob_sample, prob_target])
    train_loss = CrossEntropyLoss(rescale_loss=rescale_loss)(logits, new_targets)
    train_loss_and_states = mx.sym.Group(last_states + [train_loss])

    # full softmax for testing
    #TODO same stype for decoder_b
    decoder_b = S.reshape(decoder_b, (-1,))
    eval_logits = mx.sym.FullyConnected(data=rnn_out, weight=decoder_w,
                                        num_hidden=ntokens, name='decode_fc', bias=decoder_b)
    label = mx.sym.Variable('label')
    label = mx.sym.reshape(label, shape=(-1,))
    decoder_b = mx.sym.reshape(decoder_b, shape=(-1,))
    eval_loss = CrossEntropyLoss()(eval_logits, label)
    eval_loss_and_states = mx.sym.Group(last_states + [eval_loss])

    # training module
    data_names, label_names = ['data', 'mask'], ['label']
    eval_state_names = state_names
    extra_states_names = ['sample', 'prob_sample', 'prob_target']
    num_extra_states = len(extra_states_names)
    train_state_names = state_names + extra_states_names

    module = CustomModule(symbol=train_loss_and_states, context=ctx,
                          state_names=train_state_names,
                          data_names=data_names, label_names=label_names)
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier(factor_type='out'))

    kvstore = None if args.kvstore is None else mx.kv.create(args.kvstore)
    require_rsp_pull = kvstore and not args.dense
    optimizer = mx.optimizer.create('adagrad', learning_rate=args.lr, \
                                    rescale_grad=1.0/ngpus, eps=args.eps)

    module.init_optimizer(optimizer=optimizer, kvstore=kvstore)
    num_words_per_batch = args.batch_size * ngpus * args.bptt
    speedometer = mx.callback.Speedometer(num_words_per_batch, args.log_interval)

    if args.profile:
        config = ['dense', args.dense, 'ngpus', ngpus]
        config_str = map(lambda x: str(x), config)
        filename = '-'.join(config_str) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=filename)
        mx.profiler.profiler_set_state('run')

    # train

    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        total_L = mx.nd.array([0.0])
        nbatch = 0
        module.set_states(value=0)
        state_cache = module.get_states(merge_multi_context=False)[:-num_extra_states]
        next_batch = train_data.next()
        next_sampled_values = generate_samples(next_batch.label[0], ngpus, args.k, ntokens)
        stop_iter = False
        while not stop_iter:
            batch = next_batch
            lists = next_sampled_values
            state_cache += lists
            module.set_states(states=state_cache)
            if require_rsp_pull:
                data = batch.data[0]
                target_ids = [batch.label[0]]
                sampled_ids = lists[0]
                param_rowids = {'encoder_weight': data,
                                'decoder_weight': sampled_ids + target_ids}
                module.prepare_sparse_params(param_rowids)
            module.forward(batch)
            try:
                next_batch = train_data.next()
                next_sampled_values = generate_samples(next_batch.label[0], ngpus, args.k, ntokens)
            except StopIteration:
                stop_iter = True
            outputs = module.get_outputs(merge_multi_context=False)
            state_cache = outputs[:-1]
            module.backward()
            for g in range(ngpus):
                total_L += outputs[-1][g].copyto(mx.cpu()) / ngpus

            # update all parameters (including the weight parameter)
            module.rescale_grad(args.rescale_embed, 'encoder_weight')
            norm = module.clip_by_global_norm_per_ctx(max_norm=args.clip, param_names=lstm_args)
            module.update()
            speedometer_param = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                       eval_metric=None, locals=locals())

            speedometer(speedometer_param)
            # update training metric
            if nbatch % args.log_interval == 0 and nbatch > 0:
                cur_L = total_L.asscalar() / args.log_interval / rescale_loss
                try:
                    ppl = math.exp(cur_L)
                except OverflowError:
                    ppl = 1e36
                logging.info('Iter[%d] Batch [%d] \tloss %.7f, ppl %.7f'%(epoch, nbatch, cur_L, ppl))
                total_L[:] = 0.0
            nbatch += 1

        # run evaluation with full softmax on cpu
        module.save_checkpoint(args.checkpoint_dir, epoch, save_optimizer_states=False)
        cpu_train_mod = CustomModule.load(args.checkpoint_dir, epoch, context=mx.cpu(),
                                          state_names=train_state_names,
                                          data_names=data_names, label_names=label_names)
        eval_data = mx.io.PrefetchingIter(MultiSentenceIter(args.test, vocab,
                                          args.batch_size, args.bptt))
        cpu_train_mod.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label)

        # eval module
        eval_module = CustomModule(symbol=eval_loss_and_states, context=mx.cpu(), data_names=data_names,
                                   label_names=label_names, state_names=eval_state_names)
        eval_module.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label,
                         shared_module=cpu_train_mod, for_training=False)
        val_L = run_utils.evaluate(eval_module, eval_data, epoch, 2)
        train_data.reset()
    logging.info("Training completed. ")
    if args.profile:
        mx.profiler.profiler_set_state('stop')
