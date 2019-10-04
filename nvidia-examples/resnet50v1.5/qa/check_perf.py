import argparse
import json
from math import isfinite

def str_list(x):
    return x.split(',')

parser = argparse.ArgumentParser(description='Performace Tests')
parser.add_argument('baseline', metavar='DIR', help='path to baseline')
parser.add_argument('report', metavar='DIR', help='path to report')
parser.add_argument('--metrics', default=['train.total_ips', 'val.total_ips'], type=str_list)
parser.add_argument('--tolerance', default=0.15, type=float)
args = parser.parse_args()

def check(baseline, report):

    allright = True

    for m in args.metrics:
        for ngpus in report['metrics']:
            for bs in report['metrics'][ngpus]:
                minv = baseline['metrics'][ngpus][bs][m] * (1 - args.tolerance)
                r = report['metrics'][ngpus][bs][m]

                if not isfinite(r) or r < minv:
                    allright = False
                    print("Result value doesn't match baseline: {} ngpus {} batch-size {}, allowed min: {}, result: {}".format(
                          m, ngpus, bs, minv, r))

    return allright


with open(args.report, 'r') as f:
    report_json = json.load(f)

with open(args.baseline, 'r') as f:
    baseline_json = json.load(f)

if check(baseline_json, report_json):
    print("&&&& PERF PASSED")
    exit(0)
else:
    print("&&&& PERF FAILED")
    exit(1)
