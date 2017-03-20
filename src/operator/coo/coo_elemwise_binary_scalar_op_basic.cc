/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
//#include "./elemwise_unary_op.h"
#include "./../tensor/elemwise_binary_op.h"
#include "./coo_elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_COO_BINARY_SCALAR(_coo_plus_scalar)
.set_attr<FComputeNDArray>("FComputeNDArray<cpu>", COOBinaryScalarComputeNDArray<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.add_alias("COOPlusScalar");

//.set_attr<FCompute>("FCompute<cpu>", COOBinaryScalarCompute<cpu, mshadow::op::plus>)
}  // namespace op
}  // namespace mxnet
