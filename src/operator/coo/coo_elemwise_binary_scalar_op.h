/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.h
 * \brief Function defintion of elementwise binary scalar operators
 */
#ifndef MXNET_OPERATOR_COO_ELEMWISE_BINARY_SCALAR_OP_H_
#define MXNET_OPERATOR_COO_ELEMWISE_BINARY_SCALAR_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

template<int n_in, int n_out>
inline bool ElemwiseChunkType2(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  auto num_in = in_attrs->size();
  auto num_out = out_attrs->size();
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  //TODO implement me
  //return ElemwiseAttr<int, type_is_none, type_assign, true>(
  //  attrs, in_attrs, out_attrs, -1);
  CHECK_EQ(n_in, 1);
  CHECK_EQ(n_out, 1);
  (*out_attrs)[0] = (*in_attrs)[0];
  // For elemwise ops, ChunkType doesn't change
  return true;
}


template<typename xpu, typename OP>
void COOBinaryScalarCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  double alpha = nnvm::get<double>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, scalar<DType>(DType(alpha))));
  });
}

template<typename xpu, typename OP>
void COOBinaryScalarComputeNDArray(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& inputs,
                         const std::vector<OpReqType>& req,
  // Temporarily mark output as non-const
                         const std::vector<NDArray>& outputs) {
                         //std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  double alpha = nnvm::get<double>(attrs.parsed);
  // The shape inferred for the output ndarray is okay, but the mem alloc is not necessary
  if (inputs[0].chunk_type() == DefaultChunk) {
    MSHADOW_TYPE_SWITCH(outputs[0].data().type_flag_, DType, {
      Tensor<xpu, 1, DType> out = outputs[0].data().FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> lhs = inputs[0].data().FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, scalar<DType>(DType(alpha))));
    });
  } else {
    
    // TODO add more chunk types
    CHECK(inputs[0].chunk_type() == RowSparseChunk);
    //outputs[0] = NDArray(RowSparseChunk, outputs[0].shape(), Context::Create(static_cast<Context::DeviceType>(0), 0));
    // context = cpu(0);
    auto output_ptr = const_cast<NDArray*>(&(outputs[0]));
    *output_ptr = NDArray(outputs[0].shape(), Context::Create(static_cast<Context::DeviceType>(1), 0));
    std::cout << "Compute input is sparse \n";
    auto &output = *output_ptr;
    {
        TShape aux_shape({1});
        output.CheckAndAlloc(aux_shape);
        mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
        mshadow::Copy(output.data().FlatTo2D<cpu, real_t>(s)[0], inputs[0].data().FlatTo2D<cpu, real_t>(s)[0], s);
     }
  } 
}

template<typename xpu, typename OP>
void COOBinaryScalarBackward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  double alpha = nnvm::get<double>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> igrad = outputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
    Tensor<xpu, 1, DType> lhs = inputs[1].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(igrad, req[0], ograd*F<OP>(lhs, scalar<DType>(DType(alpha))));
  });
}

#define MXNET_OPERATOR_REGISTER_COO_BINARY_SCALAR(name)             \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr_parser([](NodeAttrs* attrs) {                           \
      attrs->parsed = std::stod(attrs->dict["scalar"]);             \
    })                                                              \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInferChunkType>("FInferChunkType", ElemwiseChunkType2<1, 1>) \
  .add_argument("data", "ndarray-or-symbol", "source input")                   \
  .add_argument("scalar", "float", "scalar input")

// TODO Inplace Option?

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_COO_ELEMWISE_BINARY_SCALAR_OP_H_
