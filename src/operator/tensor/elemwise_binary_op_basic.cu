/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_LAUNCH(elemwise_add, gpu, mshadow::op::plus);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH(_grad_add, gpu, mshadow::op::plus);

NNVM_REGISTER_OP(_backward_add)
.set_attr<FCompute>("FCompute<gpu>",
                    BinaryOp::BinaryBackwardUseNone<gpu,
                    mshadow_op::identity, mshadow_op::identity>)
.set_attr<FComputeEx>("FComputeEx<gpu>",
  BinaryOp::BinaryBackwardUseNoneEx<gpu, mshadow_op::identity, mshadow_op::identity>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH(elemwise_sub, gpu, mshadow::op::minus)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus");

NNVM_REGISTER_OP(_backward_sub)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseNone<gpu, mshadow_op::identity,
                                                                    mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH(elemwise_mul, gpu, mshadow::op::mul)
.add_alias("_mul").add_alias("_Mul");

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::right,
                                                                  mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH(elemwise_div, gpu, mshadow::op::div)
.add_alias("_div").add_alias("_Div");

NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>", BinaryOp::BinaryBackwardUseIn<gpu, mshadow_op::div_grad,
                                                                  mshadow_op::div_rgrad>);

}  // namespace op
}  // namespace mxnet
