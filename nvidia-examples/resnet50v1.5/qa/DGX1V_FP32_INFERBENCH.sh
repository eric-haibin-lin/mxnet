#!/bin/bash
python ../benchmark.py --executable ../runner -n 1 -b 1,2,4,8,64,128 -i 600 -e 2 -w 1 --dtype=float32 --data-backend synthetic --mode val -o report.json
python check_perf.py benchmark_baselines/dgx1v16g_infer_fp32_synth.json report.json --metrics val.total_ips
