/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow_op::power>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::LaunchEx<gpu, mshadow_op::power>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>);

NNVM_REGISTER_OP(_backward_power)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::power_grad,
  mshadow_op::power_rgrad>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::BinaryBackwardUseInEx<gpu, mshadow_op::power_grad,
  mshadow_op::power_rgrad>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>);

NNVM_REGISTER_OP(_maximum)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow_op::maximum>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::LaunchEx<gpu, mshadow_op::maximum>);

NNVM_REGISTER_OP(_backward_maximum)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::ge,
  mshadow_op::lt>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::BinaryBackwardUseInEx<gpu, mshadow_op::ge,
  mshadow_op::lt>);

NNVM_REGISTER_OP(_minimum)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow_op::minimum>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::LaunchEx<gpu, mshadow_op::minimum>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>);

NNVM_REGISTER_OP(_backward_minimum)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::le,
  mshadow_op::gt>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::BinaryBackwardUseInEx<gpu, mshadow_op::le,
  mshadow_op::gt>);

NNVM_REGISTER_OP(_hypot)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow_op::hypot>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::LaunchEx<gpu, mshadow_op::hypot>)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>);

NNVM_REGISTER_OP(_backward_hypot)
.set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<3, 2>)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::hypot_grad_left,
  mshadow_op::hypot_grad_right>)
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryOp::BinaryBackwardUseInEx<gpu,
  mshadow_op::hypot_grad_left, mshadow_op::hypot_grad_right>);

}  // namespace op
}  // namespace mxnet
