#!/bin/bash
python ../benchmark.py --executable ../runner -n 1 -b 1,2,4,8,16,32,64,128 -i 800 -e 2 -w 1 --num-examples 38400 --dtype=float32 --mode val -o report.json
python check_perf.py benchmark_baselines/dgx1v16g_infer_fp32_dali.json report.json --metrics val.total_ips
