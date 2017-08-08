# pylint: skip-file
import mxnet as mx
import numpy as np
from mxnet.test_utils import assert_almost_equal, default_context

shape = (4, 4)
keys = [5, 7, 11]
str_keys = ['b', 'c', 'd']


def init_kv_with_str(stype='default'):
    """init kv """
    kv = mx.kv.create()
    # single
    kv.init('a', mx.nd.zeros(shape, stype=stype))
    # list
    kv.init(str_keys, [mx.nd.zeros(shape=shape, stype=stype)] * len(keys))
    return kv


def test_row_sparse_pull():
    kv = init_kv_with_str('row_sparse')
    kv.init('e', mx.nd.ones(shape)._to_rsp())

    def check_row_sparse_pull(kv, count, ctx=default_context()):
        num_rows = shape[0]
        vals = []
        row_ids = []
        all_row_ids = np.arange(num_rows)
        for i in range(count):
            vals.append(mx.nd.zeros(shape, ctx=ctx)._to_rsp())
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

    check_row_sparse_pull(kv, 1, mx.gpu(0))
    check_row_sparse_pull(kv, 4, mx.gpu(0))


if __name__ == '__main__':
    test_row_sparse_pull()
