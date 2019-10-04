import argparse
import json
from math import isfinite

def str_list(x):
    return x.split(',')

parser = argparse.ArgumentParser(description='Convergence Tests')
parser.add_argument('baseline', metavar='DIR', help='path to baseline')
parser.add_argument('report', metavar='DIR', help='path to report')
parser.add_argument('--metrics', default=['train.loss', 'train.top1', 'train.top5', 'val.top1', 'val.top5'], type=str_list)
parser.add_argument('--epochs', default=90, type=int)
args = parser.parse_args()

def check(baseline, report):

    allright = True

    for m in args.metrics:
        for epoch in range(args.epochs):
            minv = baseline['metrics'][m][epoch][0]
            maxv = baseline['metrics'][m][epoch][1]
            r = report['metrics'][m][epoch]

            if not isfinite(r) or (not (r > minv and r < maxv)):
                allright = False
                print("Result value doesn't match baseline: {} epoch {}, allowed min: {}, allowed max: {}, result: {}".format(
                    m, epoch, minv, maxv, r))

    return allright


with open(args.report, 'r') as f:
    report_json = json.load(f)

with open(args.baseline, 'r') as f:
    baseline_json = json.load(f)

if check(baseline_json, report_json):
    print("&&&& CURVES PASSED")
    exit(0)
else:
    print("&&&& CURVES FAILED")
    exit(1)
