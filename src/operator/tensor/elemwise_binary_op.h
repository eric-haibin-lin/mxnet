/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_op.h
 * \brief Function defintion of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {
template<typename xpu, typename OP>
void BinaryCompute(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rhs = inputs[1].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, rhs));
  });
}

// TODO(haibin) This is an inefficient temporary implementation
// Binary Compute between two row-sparse ndarray
template<typename xpu, typename OP>
void BinaryComputeExRsRs(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  auto &nd_l = inputs[0];
  auto &nd_r = inputs[1];
  auto &output = outputs[0];

  CHECK_EQ(nd_l.storage_type(), kRowSparseStorage) << "Sparse type not supported yet";
  // Memory Estimation
  auto num_rows_l = nd_l.aux_shape(rowsparse::kIdx)[0];
  auto num_rows_r = nd_r.aux_shape(rowsparse::kIdx)[0];
  // This is (roughly) the number of result rows
  output.CheckAndAlloc({TShape({num_rows_l + num_rows_r})});
  // LOG(INFO) << "BinaryComputeExRsRs" << output.aux_shape(rowsparse::kIdx)[0];
  // Indices
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
    MSHADOW_TYPE_SWITCH(nd_l.aux_type(rowsparse::kIdx), AuxType, {
      auto indices_l = nd_l.aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto indices_r = nd_r.aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto indices_out = output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      // Data
      auto data_l = nd_l.data().FlatTo2D<xpu, DType>(s);
      auto data_r = nd_r.data().FlatTo2D<xpu, DType>(s);
      auto out = output.data().FlatTo2D<xpu, DType>(s);

      // TODO(haibin) A more appropriate way: Copy to output, then apply ops
      size_t iter_l = 0;
      size_t iter_r = 0;
      size_t iter_out = 0;
      int32_t num_common_rows = 0;
      while (iter_l < num_rows_l && iter_r < num_rows_r) {
        auto idx_l = indices_l[iter_l];
        auto idx_r = indices_r[iter_r];
        if (idx_l == idx_r) {
          // Same row
          indices_out[iter_out] = idx_l;
          mshadow::Copy(out[iter_out], data_l[iter_l++], s);
          out[iter_out] += data_r[iter_r++];
          num_common_rows++;
        } else if (idx_l < idx_r) {
          // Left only
          indices_out[iter_out] = idx_l;
          mshadow::Copy(out[iter_out], data_l[iter_l++], s);
        } else {
          // Right only
          indices_out[iter_out] = idx_r;
          mshadow::Copy(out[iter_out], data_r[iter_r++], s);
        }
        iter_out++;
      }
      // Copying over the rest of the rows
      while (iter_l < num_rows_l) {
        indices_out[iter_out] = indices_l[iter_l];
        mshadow::Copy(out[iter_out++], data_l[iter_l++], s);
      }
      while (iter_r < num_rows_r) {
        indices_out[iter_out] = indices_r[iter_r];
        mshadow::Copy(out[iter_out++], data_r[iter_r++], s);
      }
      auto new_shape = output.aux_shape(rowsparse::kIdx);
      new_shape[0] -= num_common_rows;
      output.SetAuxShape(rowsparse::kIdx, new_shape);
    });
  });
}

template<typename xpu, typename OP>
void BinaryComputeEx(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // If any input is dense, fallback to FCompute
  if (common::HasDefaultStorage(inputs)) {
    FComputeExFallback<xpu>(attrs, ctx, inputs, req, outputs, BinaryCompute<xpu, OP>);
    return;
  }
  // Call RsRs function
  CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage) << "Sparse type not supported yet";
  BinaryComputeExRsRs<xpu, Op>(attrs, ctx, inputs, req, outputs);
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNone(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
    ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
  });
}

// Only implemented for _backward_add for now
template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseNoneEx(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (inputs[0].storage_type() == kDefaultStorage) {
    LOG(FATAL) << "BinaryBackwardUseNoneEx fallback not implemented yet";
  }
  // LOG(INFO) << "BinaryBackwardUseNoneEx";
  // The following code assumes LOP == mshadow_op::identity == ROP
  CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage);
  TShape shape = inputs[0].aux_shape(rowsparse::kIdx);
  outputs[0].CheckAndAlloc({shape});
  outputs[1].CheckAndAlloc({shape});
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), AuxType, {
      auto lgrad_idx = outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto rgrad_idx = outputs[1].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto ograd_idx = inputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto lgrad = outputs[0].data().FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> rgrad = outputs[1].data().FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> ograd = inputs[0].data().FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
      ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
      ASSIGN_DISPATCH(lgrad_idx, req[0], F<LOP>(ograd_idx));
      ASSIGN_DISPATCH(rgrad_idx, req[1], F<ROP>(ograd_idx));
    });
  });
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseOut(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> out = inputs[1].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(lgrad, req[0], ograd*F<LOP>(out));
    ASSIGN_DISPATCH(rgrad, req[1], ograd*F<ROP>(out));
  });
}

template<typename xpu, typename LOP, typename ROP>
void BinaryBackwardUseIn(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> lgrad = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rgrad = outputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[1].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> rhs = inputs[2].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(lgrad, req[0], ograd*F<LOP>(lhs, rhs));
    ASSIGN_DISPATCH(rgrad, req[1], ograd*F<ROP>(lhs, rhs));
  });
}

#define MXNET_OPERATOR_REGISTER_BINARY(name)                        \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(2)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FListInputNames>("FListInputNames",               \
    [](const NodeAttrs& attrs) {                                    \
      return std::vector<std::string>{"lhs", "rhs"};                \
    })                                                              \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<2, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}, {1, 0}};     \
    })                                                              \
  .add_argument("lhs", "NDArray-or-Symbol", "first input")          \
  .add_argument("rhs", "NDArray-or-Symbol", "second input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
