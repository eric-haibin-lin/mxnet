/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file elemwise_binary_scalar_op.cu
 * \brief GPU Implementation of unary function.
 */
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_WITH_HALF2(elemwise_add, mshadow::op::plus);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_WITH_HALF2(_grad_add, mshadow::op::plus);

NNVM_REGISTER_OP(_backward_add)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BinaryBackwardUseNoneWithHalf2<gpu,
  mshadow_op::identity, mshadow_op::identity>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_WITH_HALF2(elemwise_sub, mshadow::op::minus)
.add_alias("_sub").add_alias("_minus").add_alias("_Minus");

NNVM_REGISTER_OP(_backward_sub)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BinaryBackwardUseNoneWithHalf2<gpu,
  mshadow_op::identity, mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_WITH_HALF2_DENSE_LRVALUE(elemwise_mul, mshadow::op::mul)
.add_alias("_mul").add_alias("_Mul");

NNVM_REGISTER_OP(_backward_mul)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BinaryBackwardUseInWithHalf2<gpu,
  mshadow_op::right, mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_WITH_HALF2_CUDA_DR(elemwise_div, mshadow::op::div)
.add_alias("_div").add_alias("_Div");
NNVM_REGISTER_OP(_backward_div)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BinaryBackwardUseInWithHalf2<gpu,
  mshadow_op::div_grad, mshadow_op::div_rgrad>);

NNVM_REGISTER_OP(_mod)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::LaunchWithHalf2<gpu, mshadow_op::mod>);

NNVM_REGISTER_OP(_backward_mod)
.set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::BinaryBackwardUseInWithHalf2<
  gpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet
