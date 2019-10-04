#!/bin/bash
../runner -n 8 -b 192 --seed 0 --report report.json
python check_curves.py curve_baselines/dgx1v16g_8gpu_fp16.json report.json &&
python check_duration.py curve_baselines/dgx1v16g_8gpu_fp16.json report.json
