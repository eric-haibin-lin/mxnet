import os
import random
from mxnet.test_utils import *


def get_data(data_dir, data_name, url, data_origin_name):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        import urllib
        zippath = os.path.join(data_dir, data_origin_name)
        urllib.urlretrieve(url, zippath)
        os.system("bzip2 -d %r" % data_origin_name)
    os.chdir("..")


def estimate_density(DATA_PATH, feature_size):
    """sample 10 times of a size of 1000 for estimating the density of the sparse dataset"""
    if not os.path.exists(DATA_PATH):
        raise Exception("Data is not there!")
    density = []
    P = 0.01
    for _ in xrange(10):
        num_non_zero = 0
        num_sample = 0
        with open(DATA_PATH) as f:
            for line in f:
                if (random.random() < P):
                    num_non_zero += len(line.split(" ")) - 1
                    num_sample += 1
        density.append(num_non_zero * 1.0 / (feature_size * num_sample))
    return sum(density) / len(density)


def _get_uniform_dataset(num_rows, num_cols, density=0.1):
    """Returns CSRNDArray with uniform distribution
    """
    if (num_rows <= 0 or num_cols <= 0):
        raise ValueError("num_rows or num_cols should be greater than 0")

    if (density < 0 or density > 1):
        raise ValueError("density has to be between 0 and 1")

    return mx.nd.array(sp.rand(num_rows, num_cols, density).toarray())._to_csr()


