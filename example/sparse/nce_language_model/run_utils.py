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

import argparse, time, logging

def get_parser():
    parser = argparse.ArgumentParser(description='Language Model on GBW')
    parser.add_argument('--data', type=str, default='./data/ptb.train.txt',
                        help='location of the data corpus')
    parser.add_argument('--test', type=str, default='./data/ptb.train.txt',
                        help='such as /home/ubuntu/gbw-validation/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050')
    parser.add_argument('--vocab', type=str, default='./data/1b_word_vocab.txt',
                        help='location of the corpus vocab')
    parser.add_argument('--emsize', type=int, default=512,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=2048,
                        help='number of hidden units per layer')
    parser.add_argument('--num_proj', type=int, default=512,
                        help='number of projection units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--epochs', type=int, default=8,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size per gpu')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--eps', type=float, default=0.0001,
                        help='eps for adagrad')
    parser.add_argument('--bptt', type=int, default=20,
                        help='sequence length')
    parser.add_argument('--k', type=int, default=8192,
                        help='number of noise samples to estimate')
    parser.add_argument('--gpus', type=str,
                        help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
    parser.add_argument('--dense', action='store_true',
                        help='use dense embedding instead of sparse embedding')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='report interval')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--profile', action='store_true',
                        help='whether to use profiler')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='type of kv-store to use')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoint/',
                        help='dir for checkpoint')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping by global norm. Done per context.')
    parser.add_argument('--rescale-embed', type=float, default=None,
                        help='rescale-embedding-grad')
    # TODO mod.reshapet diff
    parser.add_argument('--eval-batch-size', type=int, default=32,
                        help='batch size for eval')
    return parser

def evaluate(mod, data_iter, epoch, log_interval):
    """ Run evaluation on cpu"""
    start = time.time()
    total_L = 0.0
    nbatch = 0
    mod.set_states(value=0)
    for batch in data_iter:
        mod.forward(batch, is_train=False)
        outputs = mod.get_outputs(merge_multi_context=False)
        states = outputs[:-1]
        total_L += outputs[-1][0].asscalar()
        mod.set_states(states=states)
        nbatch += 1
        if (nbatch + 1) % log_interval == 0:
            logging.info("eval batch %d : %.7f" % (nbatch, total_L / nbatch))
    data_iter.reset()
    loss = total_L / nbatch
    try:
        ppl = math.exp(loss)
    except Exception:
        ppl = 1e37
    end = time.time()
    logging.info('Iter[%d]\t\t CE loss %.7f, ppl %.7f. Eval duration = %.2f seconds'%(epoch, loss, ppl, end - start))
    return loss

