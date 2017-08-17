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
 * \file elemwise_unary_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {

// relu
MXNET_OPERATOR_REGISTER_UNARY_LAUNCH(relu, cpu, kernel_launch_op::relu)
.describe(R"code(Computes rectified linear.

.. math::
   max(features, 0)

The storage type of ``relu`` output depends upon the input storage type:

  relu(default) = default
  relu(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_relu"});


MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_relu, kernel_launch_op::relu_grad);

// sigmoid
MXNET_OPERATOR_REGISTER_UNARY_LAUNCH_DR(sigmoid, cpu, kernel_launch_op::sigmoid)
.describe(R"code(Computes sigmoid of x element-wise.

.. math::
   y = 1 / (1 + exp(-x))

The storage type of ``sigmoid`` output is always denseThe storage type of ``sigmoid`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_sigmoid, kernel_launch_op::sigmoid_grad);

// copy
MXNET_OPERATOR_REGISTER_UNARY(_copy)
.MXNET_DESCRIBE("Returns a copy of the input.")
.add_alias("identity")
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"});

NNVM_REGISTER_OP(_backward_copy)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  });

MXNET_OPERATOR_REGISTER_UNARY(BlockGrad)
.add_alias("stop_gradient")
.describe(R"code(Stops gradient computation.

Stops the accumulated gradient of the inputs from flowing through this operator
in the backward direction. In other words, this operator prevents the contribution
of its inputs to be taken into account for computing gradients.

Example::

  v1 = [1, 2]
  v2 = [0, 1]
  a = Variable('a')
  b = Variable('b')
  b_stop_grad = stop_gradient(3 * b)
  loss = MakeLoss(b_stop_grad + a)

  executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  executor.forward(is_train=True, a=v1, b=v2)
  executor.outputs
  [ 1.  5.]

  executor.backward()
  executor.grad_arrays
  [ 0.  0.]
  [ 1.  1.]

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

MXNET_OPERATOR_REGISTER_UNARY(make_loss)
.describe(R"code(Stops gradient computation.
.. note:: ``make_loss`` is deprecated, use ``MakeLoss``.

The storage type of ``make_loss`` output depends upon the input storage type:

  make_loss(default) = default
  make_loss(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"loss"};
  })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeEx<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient",
  [](const nnvm::NodePtr& n, const std::vector<nnvm::NodeEntry>& ograds) {
    auto p = MakeNode("ones_like", n->attrs.name + "_backward",
                      &(n->inputs), nullptr, &n);
    std::vector<nnvm::NodeEntry> ret;
    ret.emplace_back(nnvm::NodeEntry{p, 0, 0});
    return ret;
  });

// identity output as first input, but attributes are constrained to be like rhs
NNVM_REGISTER_OP(_identity_with_attr_like_rhs)
.set_num_inputs(2)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"lhs", "rhs"};
  })
.set_attr<nnvm::FInplaceOption>(
    "FInplaceOption", [](const NodeAttrs& attrs) {
      return std::vector<std::pair<int, int> >{{0, 0}};
    })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
    [](const NodeAttrs& attrs){
      return std::vector<bool>{true};
    })
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 1); })
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::IdentityComputeFirstItemsEx<cpu>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", IdentityAttrLikeRhsStorageType)
.set_attr<nnvm::FGradient>(
    "FGradient",  [](const nnvm::NodePtr& n,
                     const std::vector<nnvm::NodeEntry>& ograds) {
      auto lhs = MakeNonlossGradNode(
          "_backward_copy", n, ograds, {},
          std::unordered_map<std::string, std::string>());
      auto ng = MakeNode("zeros_like", n->attrs.name + "rhs_backward",
                         {n->inputs[1]}, nullptr, &n);
      lhs.push_back(nnvm::NodeEntry{ng, 0, 0});
      return lhs;
    })
.add_argument("lhs", "NDArray-or-Symbol", "First input.")
.add_argument("rhs", "NDArray-or-Symbol", "Second input.");

DMLC_REGISTER_PARAMETER(CastParam);
NNVM_REGISTER_OP(Cast)
.add_alias("cast")
.describe(R"code(Casts all elements of the input to a new type.

.. note:: ``Cast`` is deprecated. Use ``cast`` instead.

Example::

   cast([0.9, 1.3], dtype='int32') = [0, 1]
   cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
   cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]

)code" ADD_FILELINE)
.set_attr_parser(ParamParser<CastParam>)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", CastType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", CastCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_cast"})
.add_argument("data", "NDArray-or-Symbol", "The input.")
.add_arguments(CastParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_cast)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<FCompute>("FCompute<cpu>", CastCompute<cpu>);

// negative
MXNET_OPERATOR_REGISTER_UNARY(negative)
.describe(R"code(Numerical negative of the argument, element-wise.

The storage type of ``negative`` output depends upon the input storage type:

  negative(default) = default
  negative(row_sparse) = row_sparse
  negative(csr) = csr

)code")
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::negation>)
.set_attr<FComputeEx>("FComputeEx<cpu>", UnaryOp::ComputeEx<cpu, mshadow_op::negation>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"negative"});

// reciprocal
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(reciprocal, cpu, mshadow_op::reciprocal)
.describe(R"code(Returns the reciprocal of the argument, element-wise.

Calculates 1/x.

Example::

    reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]

The storage type of ``reciprocal`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_reciprocal"});

MXNET_OPERATOR_REGISTER_BINARY(_backward_reciprocal)
.set_attr<FCompute>("FCompute<cpu>",
  ElemwiseBinaryOp::Launch<cpu, unary_bwd<mshadow_op::reciprocal_grad> >);

// abs
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(abs, cpu, mshadow_op::abs)
.describe(R"code(Returns element-wise absolute value of the input.

Example::

   abs([-2, 0, 3]) = [2, 0, 3]

The storage type of ``abs`` output depends upon the input storage type:

  abs(default) = default
  abs(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_abs"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_abs, unary_bwd<mshadow_op::sign>);

// sign
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(sign, cpu, mshadow_op::sign)
.describe(R"code(Returns element-wise sign of the input.

Example::

   sign([-2, 0, 3]) = [-1, 0, 1]

The storage type of ``sign`` output depends upon the input storage type:

  sign(default) = default
  sign(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_sign"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_sign, unary_bwd<mshadow_op::sign_grad>);

// round
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(round, cpu, mshadow_op::round)
.describe(R"code(Returns element-wise rounded value to the nearest integer of the input.

Example::

   round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]

The storage type of ``round`` output depends upon the input storage type:

  round(default) = default
  round(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes);

// rint
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(rint, cpu, mshadow_op::rint)
.describe(R"code(Returns element-wise rounded value to the nearest integer of the input.

.. note::
   - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
   - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.

Example::

   rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]

The storage type of ``rint`` output depends upon the input storage type:

  rint(default) = default
  rint(row_sparse) = row_sparse

)code" ADD_FILELINE);

// ceil
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(ceil, cpu, mshadow_op::ceil)
.describe(R"code(Returns element-wise ceiling of the input.

The ceil of the scalar x is the smallest integer i, such that i >= x.

Example::

   ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]

The storage type of ``ceil`` output depends upon the input storage type:

  ceil(default) = default
  ceil(row_sparse) = row_sparse

)code" ADD_FILELINE);

// floor
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(floor, cpu, mshadow_op::floor)
.describe(R"code(Returns element-wise floor of the input.

The floor of the scalar x is the largest integer i, such that i <= x.

Example::

   floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]

The storage type of ``floor`` output depends upon the input storage type:

  floor(default) = default
  floor(row_sparse) = row_sparse

)code" ADD_FILELINE);

// trunc
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(trunc, cpu, mshadow_op::trunc)
.describe(R"code(Return the element-wise truncated value of the input.

The truncated value of the scalar x is the nearest integer i which is closer to
zero than x is. In short, the fractional part of the signed number x is discarded.

Example::

   trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]

The storage type of ``trunc`` output depends upon the input storage type:

  trunc(default) = default
  trunc(row_sparse) = row_sparse

)code" ADD_FILELINE);

// fix
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(fix, cpu, mshadow_op::fix)
.describe(R"code(Returns element-wise rounded value to the nearest integer towards zero of the input.

Example::

   fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]

The storage type of ``fix`` output depends upon the input storage type:

  fix(default) = default
  fix(row_sparse) = row_sparse

)code" ADD_FILELINE);

// square
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(square, cpu, mshadow_op::square)
.describe(R"code(Returns element-wise squared value of the input.

.. math::
   square(x) = x^2

Example::

   square([2, 3, 4]) = [4, 9, 16]

The storage type of ``square`` output depends upon the input storage type:

  square(default) = default
  square(row_sparse) = row_sparse
  square(csr) = csr

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_square"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_square, unary_bwd<mshadow_op::square_grad>);

// sqrt
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(sqrt, cpu, mshadow_op::square_root)
.describe(R"code(Returns element-wise square-root value of the input.

.. math::
   \textrm{sqrt}(x) = \sqrt{x}

Example::

   sqrt([4, 9, 16]) = [2, 3, 4]

The storage type of ``sqrt`` output depends upon the input storage type:

  sqrt(default) = default
  sqrt(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sqrt"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_sqrt,
                                             unary_bwd<mshadow_op::square_root_grad>);

// rsqrt
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(rsqrt, cpu, mshadow_op::reciprocal_square_root)
.describe(R"code(Returns element-wise inverse square-root value of the input.

.. math::
   rsqrt(x) = 1/\sqrt{x}

Example::

   rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]

The storage type of ``rsqrt`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_rsqrt"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_rsqrt,
                                             unary_bwd<mshadow_op::reciprocal_square_root_grad>);

// exp
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(exp, cpu, mshadow_op::exp)
.describe(R"code(Returns element-wise exponential value of the input.

.. math::
   exp(x) = e^x \approx 2.718^x

Example::

   exp([0, 1, 2]) = [inf, 1, 0.707]

The storage type of ``exp`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_mul"});

// log
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(log, cpu, mshadow_op::log)
.describe(R"code(Returns element-wise Natural logarithmic value of the input.

The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``

The storage type of ``log`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log10
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(log10, cpu, mshadow_op::log10)
.describe(R"code(Returns element-wise Base-10 logarithmic value of the input.

``10**log10(x) = x``

The storage type of ``log10`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

// log2
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(log2, cpu, mshadow_op::log2)
.describe(R"code(Returns element-wise Base-2 logarithmic value of the input.

``2**log2(x) = x``

The storage type of ``log2`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_log, unary_bwd<mshadow_op::log_grad>);

// sin
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(sin, cpu, mshadow_op::sin)
.describe(R"code(Computes the element-wise sine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]

The storage type of ``sin`` output depends upon the input storage type:

  sin(default) = default
  sin(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sin" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_sin, unary_bwd<mshadow_op::sin_grad>);

// log1p
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(log1p, cpu, mshadow_op::log1p)
.describe(R"code(Returns element-wise ``log(1 + x)`` value of the input.

This function is more accurate than ``log(1 + x)``  for small ``x`` so that
:math:`1+x\approx 1`

The storage type of ``log1p`` output depends upon the input storage type:

  log1p(default) = default
  log1p(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_log1p"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_log1p, unary_bwd<mshadow_op::log1p_grad>);

// expm1
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(expm1, cpu, mshadow_op::expm1)
.describe(R"code(Returns ``exp(x) - 1`` computed element-wise on the input.

This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.

The storage type of ``expm1`` output depends upon the input storage type:

  expm1(default) = default
  expm1(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_expm1"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_expm1, unary_bwd<mshadow_op::exp>);

// cos
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(cos, cpu, mshadow_op::cos)
.describe(R"code(Computes the element-wise cosine of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]

The storage type of ``cos`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_cos"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_cos, unary_bwd<mshadow_op::cos_grad>);

// tan
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(tan, cpu, mshadow_op::tan)
.describe(R"code(Computes the element-wise tangent of the input array.

The input should be in radians (:math:`2\pi` rad equals 360 degrees).

.. math::
   tan([0, \pi/4, \pi/2]) = [0, 1, -inf]

The storage type of ``tan`` output depends upon the input storage type:

  tan(default) = default
  tan(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tan" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_tan, unary_bwd<mshadow_op::tan_grad>);

// arcsin
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(arcsin, cpu, mshadow_op::arcsin)
.describe(R"code(Returns element-wise inverse sine of the input array.

The input should be in the range `[-1, 1]`.
The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].

.. math::
   arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]

The storage type of ``arcsin`` output depends upon the input storage type:

  arcsin(default) = default
  arcsin(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsin" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_arcsin, unary_bwd<mshadow_op::arcsin_grad>);

// arccos
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(arccos, cpu, mshadow_op::arccos)
.describe(R"code(Returns element-wise inverse cosine of the input array.

The input should be in range `[-1, 1]`.
The output is in the closed interval :math:`[0, \pi]`

.. math::
   arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]

The storage type of ``arccos`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccos" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_arccos, unary_bwd<mshadow_op::arccos_grad>);

// arctan
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(arctan, cpu, mshadow_op::arctan)
.describe(R"code(Returns element-wise inverse tangent of the input array.

The output is in the closed interval :math:`[-\pi/2, \pi/2]`

.. math::
   arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]

The storage type of ``arctan`` output depends upon the input storage type:

  arctan(default) = default
  arctan(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctan" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_arctan, unary_bwd<mshadow_op::arctan_grad>);

// degrees
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(degrees, cpu, mshadow_op::degrees)
.describe(R"code(Converts each element of the input array from radians to degrees.

.. math::
   degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]

The storage type of ``degrees`` output depends upon the input storage type:

  degrees(default) = default
  degrees(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_degrees" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_degrees,
                                             unary_bwd<mshadow_op::degrees_grad>);

// radians
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(radians, cpu, mshadow_op::radians)
.describe(R"code(Converts each element of the input array from degrees to radians.

.. math::
   radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]

The storage type of ``radians`` output depends upon the input storage type:

  radians(default) = default
  radians(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_radians" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_radians,
                                             unary_bwd<mshadow_op::radians_grad>);

// sinh
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(sinh, cpu, mshadow_op::sinh)
.describe(R"code(Returns the hyperbolic sine of the input array, computed element-wise.

.. math::
   sinh(x) = 0.5\times(exp(x) - exp(-x))

The storage type of ``sinh`` output depends upon the input storage type:

  sinh(default) = default
  sinh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_sinh" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_sinh, unary_bwd<mshadow_op::sinh_grad>);

// cosh
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(cosh, cpu, mshadow_op::cosh)
.describe(R"code(Returns the hyperbolic cosine  of the input array, computed element-wise.

.. math::
   cosh(x) = 0.5\times(exp(x) + exp(-x))

The storage type of ``cosh`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_cosh" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(_backward_cosh, unary_bwd<mshadow_op::cosh_grad>);

// tanh
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(tanh, cpu, mshadow_op::tanh)
.describe(R"code(Returns the hyperbolic tangent of the input array, computed element-wise.

.. math::
   tanh(x) = sinh(x) / cosh(x)

The storage type of ``tanh`` output depends upon the input storage type:

  tanh(default) = default
  tanh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{ "_backward_tanh" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_tanh, unary_bwd<mshadow_op::tanh_grad>);

// arcsinh
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(arcsinh, cpu, mshadow_op::arcsinh)
.describe(R"code(Returns the element-wise inverse hyperbolic sine of the input array, \
computed element-wise.

The storage type of ``arcsinh`` output depends upon the input storage type:

  arcsinh(default) = default
  arcsinh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arcsinh" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_arcsinh,
                                             unary_bwd<mshadow_op::arcsinh_grad>);

// arccosh
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(arccosh, cpu, mshadow_op::arccosh)
.describe(R"code(Returns the element-wise inverse hyperbolic cosine of the input array, \
computed element-wise.

The storage type of ``arccosh`` output is always dense

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arccosh" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_arccosh,
                                             unary_bwd<mshadow_op::arccosh_grad>);

// arctanh
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(arctanh, cpu, mshadow_op::arctanh)
.describe(R"code(Returns the element-wise inverse hyperbolic tangent of the input array, \
computed element-wise.

The storage type of ``arctanh`` output depends upon the input storage type:

  arctanh(default) = default
  arctanh(row_sparse) = row_sparse

)code" ADD_FILELINE)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{ "_backward_arctanh" });

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_arctanh,
                                             unary_bwd<mshadow_op::arctanh_grad>);

// gamma
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(gamma, cpu, mshadow_op::gamma)
.MXNET_DESCRIBE(R"code(Returns the gamma function (extension of the factorial function
to the reals), computed element-wise on the input array.

The storage type of ``gamma`` output is always dense

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gamma"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_gamma, unary_bwd<mshadow_op::gamma_grad>);

// gammaln
MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(gammaln, cpu, mshadow_op::gammaln)
.MXNET_DESCRIBE(R"code(Returns element-wise log of the absolute value of the gamma function
of the input.

The storage type of ``gammaln`` output is always dense

)code")
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_gammaln"});

MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(_backward_gammaln,
                                             unary_bwd<mshadow_op::gammaln_grad>);

}  // namespace op
}  // namespace mxnet
