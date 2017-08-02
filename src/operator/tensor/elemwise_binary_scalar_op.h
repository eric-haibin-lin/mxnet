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

#ifndef NDEBUG
#include "../../../tests/cpp/include/test_ndarray_utils.h"
#endif

namespace mxnet {
namespace op {

class BinaryScalarOp : public UnaryOp
{

  template<typename xpu, typename DType, typename OP>
  static inline void FillDense(mshadow::Stream<xpu> *s,
                               const size_t size,
                               const DType val,
                               const OpReqType req,
                               DType *out) {
    using namespace mxnet_op;
    using namespace mshadow::expr;
    MXNET_ASSIGN_REQ_SWITCH(req, Req, {
      Kernel<MapSetToScalar<Req>, xpu>::Launch(s, size, out, val);
    });
  }

  /*! \brief Tensor operation against a scalar with a dense result */
  template<typename xpu, typename OP, typename DType, typename IType, typename CType>
  static void LaunchExDenseResultCSR(const nnvm::NodeAttrs &attrs,
                                     const OpContext &ctx,
                                     const NDArray& input,
                                     const OpReqType req,
                                     const NDArray& output) {
    test::print(&std::cout, "LaunchExDenseResult(): input", input) << std::endl;
    test::print_dense(&std::cout, "LaunchExDenseResult(): DENSE input", input) << std::endl;
    CHECK_EQ(output.shape(), input.shape());

    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    const double alpha = nnvm::get<double>(attrs.parsed);

    const DType dense_fill_val = OP::Map(DType(0), DType(alpha));
    //const DType dense_fill_val = 1;

    const auto row_count = static_cast<size_t>(input.shape()[0]);
    const TBlob  column_indexes = input.aux_data(csr::kIdx);
    const size_t item_count = column_indexes.Size();

    const DType *in = input.data().dptr<DType>();

//#ifndef NDEBUG
//    // Fill with recognizable garbage
//    FillDense<xpu, DType, OP>(s, output.shape().Size(), DType(99),
//                              req, output.data().dptr<DType>());
//#endif

    mshadow::Tensor<xpu, 2, DType> out = AsRowise2D<DType>(s, output.data());
    if(item_count) {
      const long last_real_col = out.shape_[1] - 1;
      const IType *column_indexes_ptr = column_indexes.dptr<IType>();

      const TBlob row_starts = input.aux_data(csr::kIndPtr);
      const CType *row_starts_ptr = row_starts.dptr<CType>();

      #pragma omp parallel for
      for (int i = 0; i < row_count; ++i) {
        std::cout << "** ROW " << i << "**" << std::endl << std::flush;
        const bool last_row = i == row_count - 1;
        // Split up into blocks of contiguous data and do those together
        const size_t row_item_start_iter = row_starts_ptr[i];
        const size_t input_items_this_row = !last_row
                                      ? static_cast<size_t>(row_starts_ptr[i + 1])
                                        - row_item_start_iter
                                      : item_count - row_item_start_iter;
        if(input_items_this_row) {
          const IType *this_row_column_indexes = column_indexes_ptr + row_item_start_iter;
          const DType *row_data_start = in + row_item_start_iter;
          const size_t this_row_first_col = this_row_column_indexes[0];
          //const size_t this_row_last_col = column_indexes_ptr[this_row_last_col_iter];
          const size_t this_row_last_col = this_row_column_indexes[input_items_this_row - 1];

          std::cout << "row: " << i
                    << ", first col: " << this_row_first_col
                    << ", last col: " << this_row_last_col
                    << std::endl << std::flush;

          size_t set_item_count = 0;

          // Fill dense up to first sparse column
          if(this_row_first_col) {
            std::cout << "PRE Dense: " << 0 << " -> "
                      << (this_row_first_col - 1)
                      << std::endl << std::flush;
            FillDense<xpu, DType, OP>(s, this_row_first_col, dense_fill_val,
                                      req, out[i].dptr_);
            set_item_count += this_row_first_col;
          }

          long last_filled_csr_col = -1;
          size_t col_iter = 0;
          while(col_iter < input_items_this_row) {
            // Fill dense between end of last pass and beginning of this pass
            const size_t start_input_col = this_row_column_indexes[col_iter];
            if(last_filled_csr_col >= 0) {
              const long output_from_col = last_filled_csr_col + 1;
              const long output_to_col = start_input_col - 1;
              DCHECK_GE(output_to_col, output_from_col);
              std::cout << "Backfill Dense: " << output_from_col << " -> " << output_to_col << std::endl << std::flush;
              const auto size = static_cast<size_t>(output_to_col - output_from_col + 1);
              FillDense<xpu, DType, OP>(s, size, dense_fill_val,
                                        req, out[i].dptr_ + output_from_col);
              set_item_count += size;
            }

            const size_t start_col_iter = col_iter;
            size_t next_col_iter = start_col_iter + 1;
            size_t csr_adjacent_count = 0;
            do {
              size_t tmp_col = start_input_col;
              for (; next_col_iter < input_items_this_row; ++next_col_iter) {
                const size_t next_input_col = this_row_column_indexes[next_col_iter];
                if (next_input_col != tmp_col + 1) {
                  break;
                }
                tmp_col = next_input_col;
                ++csr_adjacent_count;
              }
            } while (0);
            const size_t csr_col_end = start_input_col + csr_adjacent_count;
            std::cout << "CSR block: row: " << i
                      << ", left col: " << start_input_col
                      << ", right col: " << csr_col_end
                      //<< ", last col this row: " << prev_col
                      << std::endl << std::flush;
            last_filled_csr_col = csr_col_end;
            const long nr_csr_to_do = csr_adjacent_count + 1;
            CHECK_GT(nr_csr_to_do, 0);
            const size_t off = col_iter;
            std::cout << "CSR: " << start_input_col << " -> "
                      << (start_input_col + nr_csr_to_do - 1)
                      << std::endl << std::flush;
            MXNET_ASSIGN_REQ_SWITCH(req, Req, {
              mxnet_op::Kernel<BMap<OP, Req>, xpu>
              ::Launch(s, nr_csr_to_do, out[i].dptr_ + start_input_col,
                       (row_data_start + off), DType(alpha));
              //last_input_col += nr_csr_to_do;
            });
            set_item_count += nr_csr_to_do;

            //prev_col = last_input_col;
            col_iter = next_col_iter;
            test::print(&std::cout, "output row", out[i]) << std::endl << std::flush;
          }

          // Fill remaining columns
          if(last_filled_csr_col < last_real_col) {
            std::cout << "POST Dense: " << (last_filled_csr_col + 1) << " -> "
                      << ((last_filled_csr_col + 1) + (last_real_col - last_filled_csr_col) - 1)
                      << std::endl << std::flush;
            FillDense<xpu, DType, OP>(s, last_real_col - last_filled_csr_col,
                                      dense_fill_val,
                                      req, out[i].dptr_ + (last_filled_csr_col + 1));
            set_item_count += last_real_col - last_filled_csr_col;
          }
          test::print(&std::cout, "output row", out[i]) << std::endl << std::flush;
          // Make sure that we did the exact correct number of writes
          DCHECK_EQ(set_item_count, out[i].shape_.Size());
        } else {
          // Fill dense output row with value
          FillDense<xpu, DType, OP>(s, out[i].shape_.Size(), dense_fill_val,
                                    req, out[i].dptr_);
        }
      }
    } else {
      // Fill whole dense tensor with value
      FillDense<xpu, DType, OP>(s, output.shape().Size(), dense_fill_val,
                                req, output.data().dptr<DType>());
    }
    test::print(&std::cout, "LaunchExDenseResult(): output", output);
  }

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
                mxnet_op::Kernel<MapSetToScalar<Req>, xpu>::Launch(
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
        MSHADOW_TYPE_SWITCH(input.aux_data(csr::kIndPtr).type_flag_, CType, {
          LaunchExDenseResultCSR<xpu, OP, DType, IType, CType>(
            attrs, ctx, input, req, output);
        });
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
    //PRINT_OP_AND_ARRAYS(OP, inputs);
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
    //PRINT_OP_AND_ARRAYS(OP, outputs);
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
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1>) \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "source input")                   \
  .add_argument("scalar", "float", "scalar input")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_SCALAR_OP_H_
