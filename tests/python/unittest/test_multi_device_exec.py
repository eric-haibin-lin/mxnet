import os
import numpy as np
import mxnet as mx

def test_ctx_group():
    with mx.AttrScope(ctx_group='stage1'):
        data = mx.symbol.Variable('data')
        fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
        act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")

    set_stage1 = set(act1.list_arguments())
    with mx.AttrScope(ctx_group='stage2'):
        fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
        act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
        fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
        fc3 = mx.symbol.BatchNorm(fc3)
        mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

    set_stage2 = set(mlp.list_arguments()) - set_stage1

    group2ctx = {
        'stage1' : mx.cpu(1),
        'stage2' : mx.cpu(2)
    }

    texec = mlp.simple_bind(mx.cpu(0),
                            group2ctx=group2ctx,
                            data=(1,200))

    for arr, name in zip(texec.arg_arrays, mlp.list_arguments()):
        if name in set_stage1:
            assert arr.context == group2ctx['stage1']
        else:
            assert arr.context == group2ctx['stage2']
'''
This tests the simple bind function
'''
def test_ctx_group_sparse(mode='dense_sparse'):
    # Input Data
    dense_np = np.array([[1,2],[3,4],[5,6]])
    sparse_np1 = np.array([[5,10],[0,0],[0,0]])
    dense_nd = mx.nd.array(dense_np)
    val = mx.nd.array([5, 10]);
    idx = mx.nd.array([0], dtype=np.int32);
    sparse_nd1 = mx.sparse_nd.row_sparse(val, idx, (3,2))
    sparse_nd2 = mx.sparse_nd.row_sparse(val, idx, (3,2))

    # Symbols
    if mode == 'dense_dense':
      data1 = mx.symbol.Variable('data1')
      data2 = mx.symbol.Variable('data2')
      
    elif mode == 'dense_sparse':
      data1 = mx.symbol.Variable('data1')
      #data1 = mx.symbol.Variable('data1', storage_type='row_sparse')
      data2 = mx.symbol.Variable('data2', storage_type='row_sparse')

    mlp  = mx.symbol.elemwise_add(data1, data2, name='plus')
    texec = mlp.simple_bind(mx.cpu(0), data1=(3,2), data2=(3,2))
    print("Done simple_bind")
    
    #texec.arg_dict['data1'] = sparse_nd1
    #texec.arg_dict['data1'] = dense_nd
    #texec.arg_dict['data2'] = sparse_nd2
    #texec.outputs = texec._get_outputs()

    print("Done data preparation")
    output = texec.forward()

    print(output[0].asnumpy())
    for arr, name in zip(texec.arg_arrays, mlp.list_arguments()):
         pass
if __name__ == '__main__':
    #test_ctx_group()
    test_ctx_group_sparse('dense_sparse')
