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


def _get_nonuniform_dataset(num_rows, num_cols, density=0.1):
    """Returns CSRNDArray with nonuniform distribution(for lesser densities),
    with exponentially increasing number of non zeros in each row.
    For dense matrices it returns a CSRNDArray with random distribution
    """
    def check_nnz(totalnnz):
        return (totalnnz >= 0)

    if (num_rows <= 0 or num_cols <= 0):
        raise ValueError("num_rows or num_cols should be greater than 0")

    if (density < 0 or density > 1):
        raise ValueError("density has to be between 0 and 1")

    totalnnz = num_rows * num_cols * density
    unusednnz = totalnnz
    x = np.zeros((num_rows, num_cols))
    # Start with ones on each row so that no row is empty
    for i in range(num_rows):
        x[i][0] = random.uniform(0, 1)
        unusednnz = unusednnz - 1
        if not check_nnz(unusednnz):
            return mx.nd.array(x)._to_csr()

    # Populate rest of matrix with 2^i items in ith row.
    # if we have used all total nnz return the sparse matrix
    # else if we reached max column size then fill up full columns unit we use all nnz
    j = 2
    for i in range(num_rows):
        col_limit = min(num_cols, j)
        # In case col_limit reached assign same value to all elements, which is much faster
        if (col_limit == num_cols) and unusednnz > col_limit:
            x[i] = random.uniform(0, 1)
            unusednnz = unusednnz - col_limit
            if not check_nnz(unusednnz):
                return mx.nd.array(x)._to_csr()
        for k in range(1, col_limit):
            x[i][k] = random.uniform(0, 1)
            unusednnz = unusednnz - 1
            if not check_nnz(unusednnz):
                return mx.nd.array(x)._to_csr()
        j = j * 2

    if unusednnz > 0:
        return mx.nd.array(sp.random(num_rows, num_cols, density).toarray())._to_csr()
    else:
        return mx.nd.array(x)._to_csr()


def get_synthetic_dataset(num_rows, num_cols, density=0.1, dataset_type="uniform"):
    """Get uniform dataset.
    Parameters
    ----------
    num_rows: int, number of rows
    num_cols: int, number of columns
    density: 0 <= density <= 1, Indicates density or sparsity of the matrix
    dataset_type: "uniform" or "nonuniform"
    Returns
    -------
    CSRNDArray object with specified dimensions, distribution and sparsity.
    """
    if dataset_type == "uniform":
        return _get_uniform_dataset(num_rows, num_cols, density)
    elif dataset_type == "nonuniform":
        return _get_nonuniform_dataset(num_rows, num_cols, density)
    else:
        raise ValueError("non valid dataset_type: {}".format(dataset_type))
