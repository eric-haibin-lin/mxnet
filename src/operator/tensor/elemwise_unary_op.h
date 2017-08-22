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
 * \file elementwise_unary_op-inl.h
 * \brief Function definition of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../special_functions-inl.h"
#include "./broadcast_reduce-inl.h"
#include "./init_op.h"

namespace mxnet {
namespace op {
template<typename xpu, typename op>
void UnaryLaunch(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Kernel<op, xpu>::Launch(s, outputs[0].Size(),
      outputs[0].dptr<DType>(), inputs[0].dptr<DType>());
  });
}

template<typename GRAD_OP>
struct unary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a*GRAD_OP::Map(b));
  }
};

template<typename xpu, typename OP>
void UnaryCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}

inline bool CopyStorageType(const nnvm::NodeAttrs& attrs,
                            const Context& ctx,
                            int* dispatch_type,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  auto& in_stype = in_attrs->at(0);
  auto& out_stype = out_attrs->at(0);
  bool fallback = true;
  CHECK_NE(in_stype, kUndefinedStorage);
  type_assign(&out_stype, in_stype);
  if (in_stype == kDefaultStorage && out_stype == kDefaultStorage) {
    type_assign(dispatch_type, kDispatchFCompute);
    fallback = false;
  } else if ((in_stype == kRowSparseStorage || in_stype == kCSRStorage) &&
             (in_stype == out_stype)) {
    type_assign(dispatch_type, kDispatchFComputeEx);
    fallback = false;
  }
  if (fallback) {
    type_assign(&out_stype, kDefaultStorage);
    type_assign(dispatch_type, kDispatchFComputeFallback);
    FALLBACK_WARNING(attrs, ctx, in_attrs, out_attrs);
  }
  return true;
}

template<typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;
  if (req[0] == kWriteInplace) {
    CHECK_EQ(inputs[0].dptr_, outputs[0].dptr_); return;
  }
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<mshadow_op::identity>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}

template<typename xpu>
void IdentityComputeRspRspImpl(const nnvm::NodeAttrs& attrs,
                               mshadow::Stream<xpu> *s,
                               const NDArray& input,
                               const OpReqType req,
                               NDArray* output) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace rowsparse;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "kWriteTo is expected for IdentityComputeRspRspImpl";
  if (!input.storage_initialized()) {
    FillZerosRspImpl(s, output);
    return;
  }
  TShape shape = input.aux_shape(kIdx);
  output->CheckAndAlloc({shape});
  MSHADOW_TYPE_SWITCH(output->dtype(), DType, {
    MSHADOW_TYPE_SWITCH(output->aux_type(kIdx), AuxType, {
      auto out_d = output->data().FlatTo1D<xpu, DType>(s);
      auto out_aux = output->aux_data(kIdx).FlatTo1D<xpu, AuxType>(s);
      auto in_aux = input.aux_data(kIdx).FlatTo1D<xpu, AuxType>(s);
      ASSIGN_DISPATCH(out_d, req,
                      F<mshadow_op::identity>(input.data().FlatTo1D<xpu, DType>(s)));
      ASSIGN_DISPATCH(out_aux, req, F<mshadow_op::identity>(in_aux));
    });
  });
}

template<typename xpu>
void IdentityComputeEx(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;
  if (in_stype == out_stype && (in_stype == kRowSparseStorage || in_stype == kCSRStorage)) {
    if (!inputs[0].storage_initialized()) {
      FillComputeZerosEx<xpu>(attrs, ctx, inputs, req, outputs);
      return;
    }
    CHECK_NE(req[0], kAddTo) << "kAddTo is not supported for IdentityComputeEx";
    const size_t n = mxnet::num_aux_data(out_stype);
    outputs[0].CheckAndAlloc(inputs[0].aux_shapes());
    IdentityCompute<xpu>(attrs, ctx, {inputs[0].data()}, req, {outputs[0].data()});
    for (size_t i = 0; i < n; ++i) {
      IdentityCompute<xpu>(attrs, ctx, {inputs[0].aux_data(i)}, req, {outputs[0].aux_data(i)});
    }
  } else {
    LOG(FATAL) << "Not implemented: " << OperatorInfoEx(attrs, ctx, inputs, req, outputs);
  }
}

inline bool IdentityAttrLikeRhsStorageType(const nnvm::NodeAttrs& attrs,
                                           const Context& ctx,
                                           int* dispatch_type,
                                           std::vector<int> *in_attrs,
                                           std::vector<int> *out_attrs) {
  // TODO(junwu): add ctx info into storage inference logic
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);
  auto& lhs_stype = in_attrs->at(0);
  auto& rhs_stype = in_attrs->at(1);
  auto& out_stype = out_attrs->at(0);
  bool fallback = true;
  CHECK_NE(rhs_stype, kUndefinedStorage);
  CHECK(type_assign(&out_stype, rhs_stype)) << "Failed to assign output stype like rhs";
  type_assign(&lhs_stype, rhs_stype);
  if (lhs_stype == kDefaultStorage && rhs_stype == kDefaultStorage &&
      out_stype == kDefaultStorage) {
    // dns, dns -> dns
    type_assign(dispatch_type, kDispatchFCompute);
    fallback = false;
  } else if ((lhs_stype == kRowSparseStorage || lhs_stype == kCSRStorage) &&
             (lhs_stype == out_stype)) {
    type_assign(dispatch_type, kDispatchFComputeEx);
    fallback = false;
  }
  if (fallback) {
    type_assign(&out_stype, kDefaultStorage);
    type_assign(dispatch_type, kDispatchFComputeFallback);
    FALLBACK_WARNING(attrs, ctx, in_attrs, out_attrs);
  }
  return true;
}

template<typename xpu>
void IdentityLikeRhsComputeEx(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (in_stype == out_stype) {
    std::vector<NDArray> in{inputs[0]};
    IdentityComputeEx<xpu>(attrs, ctx, in, req, outputs);
  } else {
    LOG(FATAL) << "Not implemented: " << OperatorInfoEx(attrs, ctx, inputs, req, outputs);
  }
}

struct CastParam : public dmlc::Parameter<CastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Output data type.");
  }
};

inline bool CastType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return (*in_attrs)[0] != -1;
}

template<typename xpu>
void CastCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      Assign(out, req[0], tcast<DstDType>(data));
    });
  });
}

namespace kernel_launch_op {
/*! \brief sigmoid unit */
struct sigmoid {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *in) {
    out[i] = DType(DType(1.0f) / (DType(1.0f) + expf(-in[i])));
  }
};
struct sigmoid_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *out_grad, const DType *in) {
    DType x = in[i];
    out[i] = out_grad[i] * DType(x * (DType(1.0f) - x));
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *in) {
    DType x = in[i];
    out[i] = DType(x > DType(0.0f) ? x : DType(0.0f));
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *out_grad, const DType *in) {
    out[i] = out_grad[i] * DType(in[i] > DType(0.0f) ? DType(1.0f) : DType(0.0f));
  }
};
}  // namespace kernel_launch_op

#define MXNET_OPERATOR_REGISTER_UNARY(name)                         \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input array.")

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
