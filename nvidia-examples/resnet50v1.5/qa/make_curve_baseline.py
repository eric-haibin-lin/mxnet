import json
import numpy as np
import sys
from collections import OrderedDict

metrics = []

durations = []
for file in sys.argv[1:]:
    with open(file, 'r') as f:
        data = json.load(f)

    metrics.append(data['metrics'])
    durations.append(data['total_duration'])

ret = OrderedDict()
ret['total_duration'] = np.mean(durations)
ret['metric_keys'] = []
ret['metrics'] = OrderedDict()

for name in ['train.loss', 'train.top1', 'train.top5', 'val.top1', 'val.top5']:
    from_all_runs = []
    for metric in metrics:
        from_all_runs.append(metric[name][:90])

    data = np.array(from_all_runs).transpose()
    avg = np.mean(data, axis=1)
    std = np.std(data, axis=1, ddof=1)
    min = np.min(data, axis=1)
    max = np.max(data, axis=1)

    std2 = []
    for i in range(90):
        std2.append(np.mean(std[np.maximum(0, i - 4) : np.minimum(i + 4, 90)]))
    std = np.array(std2)

    low  = np.minimum(avg - std ** 0.5 * 0.8, avg - std * 10)
    high = np.maximum(avg + std ** 0.5 * 0.8, avg + std * 10)

    print(name, np.min([min - low, high - max], axis=0), file=sys.stderr)

    ret['metric_keys'].append(name)
    ret['metrics'][name] = []
    for i in range(90):
        ret['metrics'][name].append([float(low[i]), float(high[i])])

print(json.dumps(ret, indent=4))
