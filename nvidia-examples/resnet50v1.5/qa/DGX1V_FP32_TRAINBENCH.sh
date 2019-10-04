#!/bin/bash
python ../benchmark.py --executable ../runner -n 1,8 -b 64,96 -i 250 -e 2 -w 1 --dtype=float32 --data-backend synthetic --mode train -o report.json
python check_perf.py benchmark_baselines/dgx1v16g_train_fp32_synth.json report.json --metrics train.total_ips
