#!/bin/bash
python ../benchmark.py --executable ../runner -n 1,2,4,8 -b 64,128,192 -i 2000 -e 2 -w 1 --num-examples 38400 --mode train -o report.json
python check_perf.py benchmark_baselines/dgx1v16g_train_fp16_dali.json report.json --metrics train.total_ips
