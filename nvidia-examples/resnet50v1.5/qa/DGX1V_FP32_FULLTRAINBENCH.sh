#!/bin/bash
python ../benchmark.py --executable ../runner -n 1,2,4,8 -b 32,64,96 -i 2000 -e 2 -w 1 --num-examples 38400 --dtype=float32 --mode train -o report.json
python check_perf.py benchmark_baselines/dgx1v16g_train_fp32_dali.json report.json --metrics train.total_ips
