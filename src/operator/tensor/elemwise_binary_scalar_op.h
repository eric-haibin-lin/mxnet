/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_scalar_op.h
 * \brief Function definition of elementwise binary scalar operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "elemwise_unary_op.h"

namespace mxnet {
namespace op {

class BinaryScalarOp : public UnaryOp
{
  template<typename xpu, typename OP, typename DType, typename IType>
  static void LaunchExDenseResult(const nnvm::NodeAttrs &attrs,
                                  const OpContext &ctx,
                                  const NDArray &input,
                                  const OpReqType req,
                                  const NDArray output) {
//    test::print(&std::cout, "LaunchExDenseResult(): input", input);
//    test::print(&std::cout, "LaunchExDenseResult(): PRE OUTPUT", output);
    CHECK_EQ(output.storage_type(), kDefaultStorage);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const double alpha = nnvm::get<double>(attrs.parsed);
    switch(input.storage_type()) {
      case kRowSparseStorage: {
        CHECK_EQ(output.shape(), input.shape());
        const long row_count = output.shape()[0];
        const long items_per_row = output.shape().Size() / row_count;
        const DType result_for_zero = OP::Map(DType(0), DType(alpha));
        Tensor<xpu, 1, DType> input_data = input.data().FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> output_data = output.data().FlatTo1D<xpu, DType>(s);
        const long sparse_row_count = input.aux_shape(rowsparse::kIdx).Size();
        if(sparse_row_count != row_count) {
          Tensor<xpu, 1, IType> row_indexes = input.aux_data(
            rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
          long input_iter = 0;
          long output_row = 0;
          IType next_input_row = 0;
          while (output_row < row_count) {
            next_input_row = input_iter < sparse_row_count ? long(row_indexes[input_iter]) : row_count;
            // Split up into blocks of contiguous data and do those together

            // Do contiguous dense blocks
            const long dense_block_count = next_input_row - output_row;
            if (dense_block_count > 0) {
              MXNET_ASSIGN_REQ_SWITCH(req, Req, {
                mxnet_op::Kernel<MapIdentity<Req>, xpu>::Launch(
                  s,
                  items_per_row * dense_block_count,
                  output_data.dptr_ + items_per_row * output_row,
                  result_for_zero);
              });
              output_row += dense_block_count;
              continue;
            }

            // Do contiguous sparse blocks
            long next_non_contiguous_sparse = input_iter;
            while (next_non_contiguous_sparse < sparse_row_count - 1) {
              if (row_indexes[next_non_contiguous_sparse + 1]
                  != row_indexes[next_non_contiguous_sparse] + 1) {
                break;
              }
              ++next_non_contiguous_sparse;
            }
            const long sparse_block_count = next_non_contiguous_sparse - input_iter + 1;
            if (sparse_block_count > 0) {
              MXNET_ASSIGN_REQ_SWITCH(req, Req, {
                mxnet_op::Kernel<BMap<OP, Req>, xpu>::Launch(
                  s,
                  items_per_row * sparse_block_count,
                  &output_data.dptr_[items_per_row * output_row],
                  &input_data.dptr_[items_per_row * input_iter],
                  DType(alpha));
              });
              output_row += sparse_block_count;
              input_iter += sparse_block_count;
              continue;
            }
          }
        } else {
          // All rows exist (eventually we don't have to do complex
          // things to call GPU kernels because we don't need to access row indices)
          MXNET_ASSIGN_REQ_SWITCH(req, Req, {
            mxnet_op::Kernel<BMap<OP, Req>, xpu>::Launch(
              s,
              items_per_row * row_count,
              output_data.dptr_,
              input_data.dptr_,
              DType(alpha));
          });
        }
        break;
      }
      case kCSRStorage: {
        test::print(&std::cout, "LaunchExDenseResult(): input", input);
        CHECK_EQ(output.shape(), input.shape());
        const size_t  row_count = input.shape()[0];
        const size_t item_count = input.aux_shape(csr::kIdx).Size();
        const TBlob  row_starts = input.aux_data(csr::kIndPtr);
        const TBlob  column_pos =  input.aux_data(csr::kIdx);
        #pragma omp parallel for
        for(size_t i = 0; i < row_count; ++i)  {
          // Split up into blocks of contiguous data and do those together
          //const size_t start_col_iter =
          size_t start_col = 0;
          size_t end_col = 0;
        }
        test::print(&std::cout, "LaunchExDenseResult(): output", output);
        break;
      }
      default:
        CHECK(false) << "Unsupported sparse storage type";
        break;
    }
    //test::print(&std::cout, "LaunchExDenseResult(): output", output);
  }

 public:
  template<typename xpu, typename OP>
  static void BinaryScalarCompute(const nnvm::NodeAttrs &attrs,
                                  const OpContext &ctx,
                                  const std::vector<TBlob> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<TBlob> &outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const double alpha = nnvm::get<double>(attrs.parsed);
    // TODO: coolivie: Use Launch() instead
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> lhs = inputs[0].FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req[0], F<OP>(lhs, scalar < DType > (DType(alpha))));
    });
  }

  template<typename xpu, typename OP>
  static void BinaryScalarComputeEx(const nnvm::NodeAttrs &attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs) {
    DCHECK_EQ(inputs.size(), 1);
    DCHECK_EQ(outputs.size(), 1);
    CHECK_NE(inputs[0].storage_type(), kDefaultStorage);
    if(outputs[0].storage_type() != kDefaultStorage) {
      CHECK_EQ(outputs[0].storage_type(), inputs[0].storage_type());
      if (req[0] != kNullOp) {
        //test::print(&std::cout, "BinaryScalarComputeEx(): inputs[0]", inputs[0]);
        UnaryOp::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, BinaryScalarCompute<xpu, OP>);
        //test::print(&std::cout, "BinaryScalarComputeEx(): outputs[0]", outputs[0]);
      }
    } else {
      if(typeid(xpu) == typeid(gpu)) {
        mxnet::op::FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                                        BinaryScalarCompute<xpu, OP>,
                                        "BinaryScalarComputeEx");
      } else {
        MSHADOW_TYPE_SWITCH(outputs[0].data().type_flag_, DType, {
          MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            LaunchExDenseResult<xpu, OP, DType, IType>(attrs, ctx, inputs[0], req[0], outputs[0]);
          });
        });
      }
    };
  }

  template<typename xpu, typename OP>
  static void BinaryScalarBackward(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const double alpha = nnvm::get<double>(attrs.parsed);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Tensor<xpu, 1, DType> igrad = outputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> ograd = inputs[0].FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> lhs = inputs[1].FlatTo1D<xpu, DType>(s);
      ASSIGN_DISPATCH(igrad, req[0], ograd * F<OP>(lhs, scalar < DType > (DType(alpha))));
    });
  }

};

#define MXNET_OPERATOR_REGISTER_BINARY_SCALAR(name)                 \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr_parser([](NodeAttrs* attrs) {                           \
      attrs->parsed = std::stod(attrs->dict["scalar"]);             \
    })                                                              \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1>) \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "source input")                   \
  .add_argument("scalar", "float", "scalar input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_
