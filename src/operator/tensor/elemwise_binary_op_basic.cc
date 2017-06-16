/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY(elemwise_add)
.add_alias("_add").add_alias("_plus").add_alias("_Plus")
.describe("Adds arguments element-wise.")
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::Launch<cpu, mshadow::op::plus>)
.set_attr<FComputeEx>(FCOMP_EX_CPU, BinaryOp::LaunchEx<cpu, mshadow::op::plus>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_add"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY(_grad_add)
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::Launch<cpu, mshadow::op::plus>)
.set_attr<FComputeEx>(FCOMP_EX_CPU, BinaryOp::LaunchEx<cpu, mshadow::op::plus>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>);

NNVM_REGISTER_OP(_backward_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::BinaryBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::identity>)
.set_attr<FComputeEx>(FCOMP_EX_CPU,
                      BinaryOp::BinaryBackwardUseNoneEx<cpu, mshadow_op::identity,
                        mshadow_op::identity>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 2>);

MXNET_OPERATOR_REGISTER_BINARY(elemwise_sub)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus")
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::Launch<cpu, mshadow::op::minus>)
.set_attr<FComputeEx>(FCOMP_EX_CPU, BinaryOp::LaunchEx<cpu, mshadow::op::minus>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::BinaryBackwardUseNone<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FComputeEx>(FCOMP_EX_CPU, BinaryOp::BinaryBackwardUseNoneEx<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 2>);

MXNET_OPERATOR_REGISTER_BINARY(elemwise_mul)
.add_alias("_mul").add_alias("_Mul")
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::Launch<cpu, mshadow::op::mul>)
.set_attr<FComputeEx>(FCOMP_EX_CPU, BinaryOp::LaunchEx<cpu, mshadow::op::mul>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::BinaryBackwardUseIn<cpu, mshadow_op::right,
    mshadow_op::left>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryOp::BinaryBackwardUseInEx<cpu, mshadow_op::right,
    mshadow_op::left>);
;

// For divide, we will always auto-convert to dense since the sparse 0's will generate nans
MXNET_OPERATOR_REGISTER_BINARY(elemwise_div)
  .add_alias("_div").add_alias("_Div")
  .set_attr<FCompute>("FCompute<cpu>", BinaryOp::Launch<cpu, mshadow::op::div>)
  .set_attr<FComputeEx>(FCOMP_EX_CPU, BinaryOp::LaunchAsDense<cpu, mshadow::op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<cpu>", BinaryOp::BinaryBackwardUseIn<cpu, mshadow_op::div_grad,
  mshadow_op::div_rgrad>)
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryOp::BinaryBackwardUseInExDense<cpu, mshadow_op::div_grad,
  mshadow_op::div_rgrad>)
;
}  // namespace op
}  // namespace mxnet
