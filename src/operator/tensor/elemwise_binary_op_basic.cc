/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(elemwise_add, mshadow::op::plus)
.add_alias("_add").add_alias("_plus").add_alias("_Plus")
.describe("Adds arguments element-wise.")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_add"});

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_grad_add, mshadow::op::plus);

NNVM_REGISTER_OP(_backward_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BinaryBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::identity>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      ElemwiseBinaryOp::BinaryBackwardUseNoneEx<cpu, mshadow_op::identity,
                        mshadow_op::identity>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 2>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(elemwise_sub, mshadow::op::minus)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BinaryBackwardUseNone<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BinaryBackwardUseNoneEx<cpu,
  mshadow_op::identity, mshadow_op::negation>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 2>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DENSE_RVALUE(elemwise_mul, mshadow::op::mul)
.add_alias("_mul").add_alias("_Mul")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BinaryBackwardUseIn<cpu, mshadow_op::right,
    mshadow_op::left>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BinaryBackwardUseInEx<cpu, mshadow_op::right,
    mshadow_op::left>);
;

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(elemwise_div, mshadow::op::div)
.add_alias("_div").add_alias("_Div")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::BinaryBackwardUseIn<cpu, mshadow_op::div_grad,
  mshadow_op::div_rgrad>)
.set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::BinaryBackwardUseInExDense<cpu, mshadow_op::div_grad,
  mshadow_op::div_rgrad>)
;
}  // namespace op
}  // namespace mxnet
