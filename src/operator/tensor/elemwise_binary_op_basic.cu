/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
NNVM_REGISTER_OP(elemwise_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow::op::plus>)
.set_attr<FComputeEx>(FCOMP_EX_GPU, BinaryOp::LaunchEx<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_grad_add)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_add)
.set_attr<FCompute>("FCompute<gpu>",
                    BinaryOp::BinaryBackwardUseNone<gpu,
                    mshadow_op::identity, mshadow_op::identity>)
.set_attr<FComputeEx>(FCOMP_EX_GPU,
  BinaryOp::BinaryBackwardUseNoneEx<gpu, mshadow_op::identity, mshadow_op::identity>);

NNVM_REGISTER_OP(_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow::op::minus>);

NNVM_REGISTER_OP(_backward_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseNone<gpu, mshadow_op::identity,
                                                                    mshadow_op::negation>);

NNVM_REGISTER_OP(_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow::op::mul>);

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::right,
                                                                  mshadow_op::left>);

NNVM_REGISTER_OP(_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::Launch<gpu, mshadow::op::div>);

NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::div_grad,
                                                                  mshadow_op::div_rgrad>);

}  // namespace op
}  // namespace mxnet
