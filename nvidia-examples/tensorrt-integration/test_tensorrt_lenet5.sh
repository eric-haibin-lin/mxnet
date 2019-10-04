#!/bin/bash

EPOCH=10
MODEL_PREFIX="lenet5"
SYMBOL="${MODEL_PREFIX}-symbol.json"
PARAMS="${MODEL_PREFIX}-$(printf "%04d" $EPOCH).params"

if [[ ! -f $SYMBOL || ! -f $PARAMS ]]; then
  echo -e "\nTrained model does not exist. Training - please wait...\n"
  python $MXNET_HOME/tests/python/tensorrt/lenet5_train.py
else
   echo "Pre-trained model exists. Skipping training."
fi

echo "Running inference script."

python test_tensorrt_lenet5.py
