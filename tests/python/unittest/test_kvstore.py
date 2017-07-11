# pylint: skip-file
import mxnet as mx
import numpy as np
from mxnet.test_utils import rand_ndarray, assert_almost_equal

shape = (4, 4)
keys = [5, 7, 11]
str_keys = ['b', 'c', 'd']

def init_kv(stype='default'):
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init(3, mx.nd.zeros(shape=shape, storage_type=stype))
    # list
    kv.init(keys, [mx.nd.zeros(shape=shape, storage_type=stype)] * len(keys))
    return kv

def init_kv_with_str(stype='default'):
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init('a', mx.nd.zeros(shape))
    # list
    kv.init(str_keys, [mx.nd.zeros(shape=shape, storage_type=stype)] * len(keys))
    return kv

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0)


def test_single_kv_pair():
    """single key-value pair push & pull"""
    def check_single_kv_pair(kv, key):
        kv.push(key, mx.nd.ones(shape))
        val = mx.nd.empty(shape)
        kv.pull(key, out = val)
        check_diff_to_scalar(val, 1)

    check_single_kv_pair(init_kv(), 3)
    check_single_kv_pair(init_kv_with_str(), 'a')

def test_row_sparse_pull():
    kv = init_kv_with_str('row_sparse')
    kv.init('e', mx.nd.ones(shape)._to_rsp())

    def check_row_sparse_pull(kv, count):
        num_rows = shape[0]
        vals = []
        row_ids = []
        all_row_ids = np.arange(num_rows)
        for i in range(count):
            vals.append(mx.nd.zeros(shape)._to_rsp())
            row_id = np.random.randint(num_rows, size=num_rows)
            row_ids.append(mx.nd.array(row_id, dtype='int64'))
        row_ids_to_pull = row_ids[0] if len(row_ids) == 1 else row_ids
        vals_to_pull = vals[0] if len(vals) == 1 else vals

        kv.row_sparse_pull('e', out=vals_to_pull, row_ids=row_ids_to_pull)
        for val, row_id in zip(vals, row_ids):
            retained = val.asnumpy()
            excluded_row_ids = np.setdiff1d(all_row_ids, row_id.asnumpy())
            for row in range(num_rows):
                expected_val = np.zeros_like(retained[row])
                expected_val += 0 if row in excluded_row_ids else 1
                assert_almost_equal(retained[row], expected_val)

    check_row_sparse_pull(kv, 1)
    check_row_sparse_pull(kv, 4)

def test_init():
    """test init"""
    def check_init(kv, key):
        kv.init(key, mx.nd.ones(shape)*4)
        a = mx.nd.zeros(shape)
        kv.pull(key, out=a)
        check_diff_to_scalar(a, 4)

    check_init(mx.kv.create(), 3)
    check_init(mx.kv.create(), 'a')


def test_list_kv_pair():
    """list key-value pair push & pull"""
    def check_list_kv_pair(kv, key):
        kv.push(key, [mx.nd.ones(shape)*4] * len(key))
        val = [mx.nd.empty(shape)] * len(key)
        kv.pull(key, out = val)
        for v in val:
            check_diff_to_scalar(v, 4)

    check_list_kv_pair(init_kv(), keys)
    check_list_kv_pair(init_kv_with_str(), str_keys)


def test_aggregator():
    """aggregate value on muliple devices"""

    def check_aggregator(kv, key, key_list):
        # devices
        num_devs = 4
        devs = [mx.Context('cpu', i) for i in range(num_devs)]

        # single
        vals = [mx.nd.ones(shape, d) for d in devs]

        kv.push(key, vals)
        kv.pull(key, out = vals)

        for v in vals:
            check_diff_to_scalar(v, num_devs)

        # list
        vals = [[mx.nd.ones(shape, d)*2.0 for d in devs]] * len(key_list)
        kv.push(key_list, vals)
        kv.pull(key_list, out = vals)

        for vv in vals:
            for v in vv:
                check_diff_to_scalar(v, num_devs * 2.0)

    check_aggregator(init_kv(), 3, keys)
    check_aggregator(init_kv_with_str(), 'a', str_keys)


def test_sparse_aggregator():
    """aggregate sparse ndarray on muliple devices"""

    stype = 'row_sparse'
    kv = init_kv(stype)

    # devices
    num_devs = 4
    devs = [mx.Context('cpu', i) for i in range(num_devs)]

    # single
    vals = [rand_ndarray(shape, stype).copyto(devs[i]) for i in range(num_devs)]
    expected_sum = np.zeros(shape)
    for v in vals:
        expected_sum += v.asnumpy()

    kv.push(3, vals)
    kv.pull(3, out = vals)
    result_sum = np.zeros(shape)
    for v in vals:
        result_sum += v.asnumpy()
    assert_almost_equal(result_sum, expected_sum * num_devs)

    # list
    vals = [[rand_ndarray(shape, stype).copyto(devs[i]) for i in range(num_devs)]] * len(keys)
    expected_sum = np.zeros(shape)
    for v in vals[0]:
        expected_sum += v.asnumpy()

    kv.push(keys, vals)
    kv.pull(keys, out = vals)
    for vv in vals:
        result_sum = np.zeros(shape)
        for v in vv:
            result_sum += v.asnumpy()
        assert_almost_equal(result_sum, expected_sum * num_devs)


def updater(key, recv, local):
    """use updater: +="""
    local += recv


def test_updater(dev = 'cpu'):
    """updater"""

    def check_updater(kv, key, key_list):
        # devices
        num_devs = 4
        devs = [mx.Context(dev, i) for i in range(num_devs)]

        # single
        vals = [mx.nd.ones(shape, d) for d in devs]

        kv.push(key, vals)
        kv.pull(key, out = vals)

        for v in vals:
            check_diff_to_scalar(v, num_devs)

        # list
        vals = [[mx.nd.ones(shape, d) for d in devs]] * len(key_list)

        num_push = 4
        for i in range(num_push):
            kv.push(key_list, vals)

        kv.pull(key_list, out = vals)

        for vv in vals:
            for v in vv:
                check_diff_to_scalar(v, num_devs * num_push)

    kv = init_kv()
    kv._set_updater(updater)
    check_updater(kv, 3, keys)

    str_kv = init_kv_with_str()
    str_kv._set_updater(updater)
    check_updater(str_kv, 'a', str_keys)



def test_get_type():
    kvtype = 'local_allreduce_cpu'
    kv = mx.kv.create(kvtype)
    assert kv.type == kvtype

if __name__ == '__main__':
    test_init()
    test_get_type()
    test_single_kv_pair()
    test_list_kv_pair()
    test_sparse_aggregator()
    test_aggregator()
    test_updater()
    test_row_sparse_pull()
