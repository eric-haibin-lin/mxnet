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

# pylint: skip-file
from __future__ import print_function
import mxnet as mx
import numpy as np
import logging
import data
from autoencoder import AutoEncoderModel
import model

train = True
predict = True

if __name__ == '__main__':
    data_file = "matrix_30_40_0.8_20_25_0.85_10_15_0.9_250_350_0.06.npy"
    X = data.get_amzn(data_file)
    train_X = X
    val_X = X
    num_sample, num_feature = X.shape

    # set to INFO to see less information during training
    logging.basicConfig(level=logging.INFO)
    dims = [num_feature, 128, 4]
    post_fix = '_'.join(map(str, dims))
    ae_model = AutoEncoderModel(mx.cpu(0), dims, #pt_dropout=0.2,
        internal_act='relu', output_act='relu')
    ae_model.loss.save("symbol" + post_fix + ".json")
    #X, _ = data.get_mnist()
    #train_X = X[:60000]
    #val_X = X[60000:]

    #np.set_printoptions(threshold=np.nan)
    #print(X.shape)
    #from matplotlib import pyplot as plt
    #plt.imshow(X, interpolation='nearest')
    #plt.show()
    monitor_count = 100
    if train:
        ae_model.layerwise_pretrain(train_X, num_sample, 5000, 'sgd', l_rate=0.1, decay=0.0,
                                    lr_scheduler=mx.misc.FactorScheduler(20000,0.1),
                                    monitor_count=monitor_count)
        ae_model.finetune(train_X, num_sample, 10000, 'sgd', l_rate=0.1, decay=0.0,
                          lr_scheduler=mx.misc.FactorScheduler(20000,0.1),
                          monitor_count=monitor_count)
        ae_model.save('mnist_pt' + post_fix + '.arg')

    if predict:
        #ae_model.load('mnist_pt.arg')
        ae_model.load('mnist_pt' + post_fix + '.arg')
        #print("Training error:", ae_model.eval(train_X))
        #print("Validation error:", ae_model.eval(val_X))
        batch_size = num_sample / 10
        #print(X.shape)
        data_iter = mx.io.NDArrayIter({'data': X}, batch_size=batch_size, shuffle=False,
                                      last_batch_handle='pad')
        Y = list(model.extract_feature(ae_model.decoder, ae_model.args, ae_model.auxs, data_iter,
                                 X.shape[0], ae_model.xpu).values())[0]
        np.save('prediction' + post_fix + '.npy', Y)
        Z = list(model.extract_feature(ae_model.encoder, ae_model.args, ae_model.auxs, data_iter,
                                 X.shape[0], ae_model.xpu).values())[0]
        np.save('encode' + post_fix + '.npy', Z)
