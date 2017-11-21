```
ubuntu@ip-172-31-37-65:~/haibin/example/model-parallel$ PYTHONPATH=../../python python autoencoder.py  --num-gpus=4
2017-11-21 06:07:28,210 Namespace(batch_size=128, num_epochs=5, num_gpus=4, num_hidden=512)
2017-11-21 06:07:33,676 Training started ...
2017-11-21 06:07:38,289 Iter[0] Batch [10]      Speed: 934.13 samples/sec
2017-11-21 06:07:39,807 Iter[1] Batch [10]      Speed: 937.70 samples/sec
2017-11-21 06:07:41,342 Iter[2] Batch [10]      Speed: 927.03 samples/sec
2017-11-21 06:07:42,859 Iter[3] Batch [10]      Speed: 937.54 samples/sec
2017-11-21 06:07:44,385 Iter[4] Batch [10]      Speed: 933.13 samples/sec
ubuntu@ip-172-31-37-65:~/haibin/example/model-parallel$ PYTHONPATH=../../python python autoencoder.py  --num-gpus=1
2017-11-21 06:07:47,012 Namespace(batch_size=128, num_epochs=5, num_gpus=1, num_hidden=512)
2017-11-21 06:07:48,699 Training started ...
2017-11-21 06:07:55,045 Iter[0] Batch [10]      Speed: 501.06 samples/sec
2017-11-21 06:07:57,875 Iter[1] Batch [10]      Speed: 501.89 samples/sec
2017-11-21 06:08:00,702 Iter[2] Batch [10]      Speed: 505.43 samples/sec
2017-11-21 06:08:03,532 Iter[3] Batch [10]      Speed: 505.08 samples/sec
2017-11-21 06:08:06,348 Iter[4] Batch [10]      Speed: 505.15 samples/sec
```
