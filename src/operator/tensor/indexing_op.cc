/*!
 * Copyright (c) 2017 by Contributors
 * \file indexing_op.cc
 * \brief
 * \author Siyi Li, Chi Zhang
*/

#include "./indexing_op.h"
namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(EmbeddingParam);
DMLC_REGISTER_PARAMETER(TakeParam);
DMLC_REGISTER_PARAMETER(OneHotParam);

NNVM_REGISTER_OP(Embedding)
.describe(R"code(Maps integer indices to vector representations (embeddings).

This operator maps words to real-valued vectors in a high-dimensional space,
called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
For example, it has been noted that in the learned embedding spaces, similar words tend
to be close to each other and dissimilar words far apart.

For an input array of shape (d1, ..., dK),
the shape of an output array is (d1, ..., dK, output_dim).
All the input values should be integers in the range [0, input_dim).

If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
(ip0, op0).

By default, if any index mentioned is too large, it is replaced by the index that addresses
the last vector in an embedding matrix.

Examples::

  input_dim = 4
  output_dim = 5

  // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
  y = [[  0.,   1.,   2.,   3.,   4.],
       [  5.,   6.,   7.,   8.,   9.],
       [ 10.,  11.,  12.,  13.,  14.],
       [ 15.,  16.,  17.,  18.,  19.]]

  // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
  x = [[ 1.,  3.],
       [ 0.,  2.]]

  // Mapped input x to its vector representation y.
  Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
                            [ 15.,  16.,  17.,  18.,  19.]],

                           [[  0.,   1.,   2.,   3.,   4.],
                            [ 10.,  11.,  12.,  13.,  14.]]]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", EmbeddingOpShape)
.set_attr<nnvm::FInferType>("FInferType", EmbeddingOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_Embedding", n, ograds,
                               {n->inputs[0]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array to the embedding operator.")
.add_argument("weight", "NDArray-or-Symbol", "The embedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Embedding)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", EmbeddingOpBackward<cpu>);

NNVM_REGISTER_OP(SparseEmbedding)
.describe(R"doc(Represents words or other sparse inputs by dense continuous vectors.
It assumes that the input is in one-hot form. E.g., for a vocabulary size of 10,000,
 each input vector is expected to have dimension 10,000.
The index of the non-zero entry is the index of the word or item it represents.

The corresponding embedding vectors are stored as rows of a matrix.
Hence, mapping an input word to its embedding is implemented as a matrix product.

The gradient of an embedding matrix has the form of gradient vectors that are only
 non-zero for words seen in a minibatch.
)doc" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(ParamParser<EmbeddingParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "weight"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", SparseEmbeddingForwardShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", SparseEmbeddingForwardStorageType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FComputeEx>(FCOMP_EX_CPU, SparseEmbeddingForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_SparseEmbedding", n, ograds,
                               {n->inputs[0]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol",
              "The input array to the sparse embedding operator.")
.add_argument("weight", "NDArray-or-Symbol", "The embedding weight matrix.")
.add_arguments(EmbeddingParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_SparseEmbedding)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", SparseEmbeddingBackwardStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseEmbeddingBackwardEx<cpu>);

NNVM_REGISTER_OP(take)
.describe(R"code(Takes elements from an input array along the given axis.

This function slices the input array along a particular axis with the provided indices.

Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the output
will have shape ``(i0, i1, d1, d2)``, computed by::

  output[i,j,:,:] = input[indices[i,j],:,:]

.. note::
   - `axis`- Only slicing along axis 0 is supported for now.
   - `mode`- Only `clip` mode is supported for now.

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  // takes elements with specified indices along axis 0
  take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
                             [ 3.,  4.]],

                            [[ 3.,  4.],
                             [ 5.,  6.]]]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr_parser(TakeParamParser<TakeParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", TakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", TakeOpType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<FCompute>("FCompute<cpu>", TakeOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n,  const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_take", n, ograds,
                               {n->inputs[1]}, n->attrs.dict);
  })
.add_argument("a", "NDArray-or-Symbol", "The input array.")
.add_argument("indices", "NDArray-or-Symbol", "The indices of the values to be extracted.")
.add_arguments(TakeParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_take)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", TakeOpBackward<cpu>);


NNVM_REGISTER_OP(batch_take)
.describe(R"code(Takes elements from a data batch.

.. note::
  `batch_take` is deprecated. Use `pick` instead.

Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
an output array of shape ``(i0,)`` with::

  output[i] = input[i, indices[i]]

Examples::

  x = [[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.]]

  // takes elements with specified indices
  batch_take(x, [0,1,0]) = [ 1.  4.  5.]

)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", BatchTakeOpShape)
.set_attr<nnvm::FInferType>("FInferType", BatchTakeOpType)
.set_attr<FCompute>("FCompute<cpu>", BatchTakeOpForward<cpu>)
.add_argument("a", "NDArray-or-Symbol", "The input array")
.add_argument("indices", "NDArray-or-Symbol", "The index array");

NNVM_REGISTER_OP(one_hot)
.describe(R"code(Returns a one-hot array.

The locations represented by `indices` take value `on_value`, while all
other locations take value `off_value`.

`one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
in an output array of shape ``(i0, i1, d)`` with::

  output[i,j,:] = off_value
  output[i,j,indices[i,j]] = on_value

Examples::

  one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
                           [ 1.  0.  0.]
                           [ 0.  0.  1.]
                           [ 1.  0.  0.]]

  one_hot([1,0,2,0], 3, on_value=8, off_value=1,
          dtype='int32') = [[1 8 1]
                            [8 1 1]
                            [1 1 8]
                            [8 1 1]]

  one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
                                      [ 1.  0.  0.]]

                                     [[ 0.  1.  0.]
                                      [ 1.  0.  0.]]

                                     [[ 0.  0.  1.]
                                      [ 1.  0.  0.]]]
)code" ADD_FILELINE)
.set_num_outputs(1)
.set_num_inputs(1)
.set_attr_parser(ParamParser<OneHotParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", OneHotOpShape)
.set_attr<nnvm::FInferType>("FInferType", OneHotOpType)
.set_attr<FCompute>("FCompute<cpu>", OneHotOpForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("indices", "NDArray-or-Symbol", "array of locations where to set on_value")
.add_arguments(OneHotParam::__FIELDS__());

NNVM_REGISTER_OP(sparse_retain)
.describe(R"code(pick rows specified by user input index array from a row sparse matrix
and save them in the output sparse matrix.

Example::

  data = [[1, 2], [3, 4], [5, 6]]
  indices = [0, 1, 3]
  shape = (4, 2)
  rsp_in = row_sparse(data, indices)
  to_retain = [0, 3]
  rsp_out = sparse_retain(rsp_in, to_retain)
  rsp_out.values = [[1, 2], [5, 6]]
  rsp_out.indices = [0, 3]

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "indices"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", SparseRetainOpShape)
.set_attr<nnvm::FInferType>("FInferType", SparseRetainOpType)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", SparseRetainForwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseRetainOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode("_backward_sparse_retain", n, ograds,
                               {n->inputs[sr::kIdx]}, n->attrs.dict);
  })
.add_argument("data", "NDArray-or-Symbol", "The input array for sparse_retain operator.")
.add_argument("indices", "NDArray-or-Symbol", "The index array of rows ids that will be retained.");

NNVM_REGISTER_OP(_backward_sparse_retain)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", SparseRetainBackwardInferStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", SparseRetainOpBackwardEx<cpu>);

}  // namespace op
}  // namespace mxnet
