# BERT inference example using MXNet
## Overview

This directory contains example of using MXNet to perform BERT QA inference on SQuAD dataset.

## Getting started

To use this example you will need to have trained BERT parameters for QA task, which you can obtain by [finetuning BERT on SQuAD dataset](https://gluon-nlp.mxnet.io/model_zoo/bert/index.html#bert-for-question-answering-on-squad). As a result you will get the `.param` file containing pretrained parameters.

To run inference, run the test_bert_inference script in this directory, providing desired max sequence length and pretrained parameters:

```
./test_bert_inference <sequence_length> <path to your parameters> float16
```
e.g.
```
./test_bert_inference 128 squad.params float16
```

This example will benchmark inference on a series of batch sizes and return achieved latency, throughput and accuracy.
