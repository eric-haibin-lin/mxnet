/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_op.h
 * \brief Function definition of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <vector>
#include <string>
#include <utility>
#include <typeinfo>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "elemwise_unary_op.h"
#include "../../common/utils.h"

namespace mxnet {
namespace op
{

/*! Gather binary operator functions into BinaryOp class */
class ElemwiseBinaryOp : public OpBase
{
 public:
  template<typename OP, int Req>
  struct BinaryOpBackwardUseNone
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad, const DType *ograd) {
      KERNEL_ASSIGN(igrad[i], Req, OP::Map(ograd[i]));
    }
    template<typename DType>
    MSHADOW_XINLINE static DType Map(const DType ograd) {
      return OP::Map(ograd);
    }
  };

  template<typename OP, int Req>
  struct BinaryOpBackwardUseIn
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad,
                                    const DType *ograd, const DType *lhs, const DType *rhs) {
      KERNEL_ASSIGN(igrad[i], Req, ograd[i] * OP::Map(lhs[i], rhs[i]));
    }
  };

  /*! \brief For sparse, assume missing rvalue is 0 */
  template<typename OP, int Req>
  struct BinaryOpMissingRValue
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], DType(0)));
    }
  };

  /*! \brief For sparse, assume missing lvalue is 0 */
  template<typename OP, int Req>
  struct BinaryOpMissingLValue
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *rhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(DType(0), rhs[i]));
    }
  };

//  /*! \brief Return only lvalue */
//  template<typename OP>
//  struct BinaryOpLValue
//  {
//    template<typename DType>
//    MSHADOW_XINLINE static DType Map(DType lvalue, const DType rvalue) {
//      return OP::Map(lvalue);
//    }
//  };
//
//  /*! \brief Return only rvalue */
//  template<typename OP>
//  struct BinaryOpRValue
//  {
//    template<typename DType>
//    MSHADOW_XINLINE static DType Map(DType lvalue, const DType rvalue) {
//      return OP::Map(rvalue);
//    }
//  };

 private:

#if 0
  // TODO(cjolivier01) Precompute parallelizing strategy
  template<typename xpu, typename DType, typename IType, typename OP>
  static inline void RspRspElemwiseBinaryOp3(const nnvm::NodeAttrs &attrs,
                                            const OpContext &ctx,
                                            const std::vector<NDArray> &inputs,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<NDArray> &outputs) {
#if 1
    using namespace mshadow;
    using namespace mxnet_op;
    using namespace mshadow::expr;

    CHECK_EQ(inputs.size(), 3U);
    CHECK_EQ(outputs.size(), 1U);

    const NDArray &ograd  = inputs[0];
    const NDArray &lhs    = inputs[1];
    const NDArray &rhs    = inputs[2];
    const NDArray *output = &outputs[0];

    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(lhs.shape(), ograd.shape());

    std::unique_ptr<NDArray> tempSparse;
    if (output->storage_type() == kDefaultStorage) {
      // Make a temporary sparse tensor for the output
      NDArray *nd = new NDArray(lhs.storage_type(), lhs.shape(), lhs.ctx(), false,
                                output->dtype());
      tempSparse.reset(nd);
      output = tempSparse.get();
    }

    // Memory Estimation: This is (roughly) the number of result rows. We still
    // need to subtract the number of common rows
    const size_t num_rows_ograd = ograd.aux_shape(rowsparse::kIdx).Size();
    const size_t num_rows_l = lhs.aux_shape(rowsparse::kIdx).Size();
    const size_t num_rows_r = rhs.aux_shape(rowsparse::kIdx).Size();



    output->CheckAndAlloc({mshadow::Shape1(num_rows_l + num_rows_r + num_rows_ograd)});
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

    // Indices
    Tensor<xpu, 1, IType> indices_ograd = ograd.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_l = lhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_r = rhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_out = output->aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);

    // Data
    Tensor<xpu, 2, DType> data_ograd = ograd.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> data_l = lhs.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> data_r = rhs.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = output->data().FlatTo2D<xpu, DType>(s);

    size_t iter_ograd = 0;
    size_t iter_l = 0;
    size_t iter_r = 0;
    size_t iter_out = 0;
    int32_t num_common_rows = 0;

    // Possible contain-row permutations
    // none
    // ograd, lhs, rhs
    // ograd, lhs
    // ograd, rhs
    // lhs, rhs
    // ograd
    // lhs
    // rhs

    while (iter_ograd < num_rows_ograd && iter_l < num_rows_l && iter_r < num_rows_r) {
      const IType idx_ograd = indices_ograd[iter_ograd];
      const IType idx_l = indices_l[iter_l];
      const IType idx_r = indices_r[iter_r];
      if (idx_ograd == idx_l && idx_l == idx_r) {
        indices_out[iter_out] = iter_ograd;
        Tensor<xpu, 1, DType> ogradvalue = data_ograd[iter_ograd++];
        Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
        Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
        DCHECK_EQ(lvalue.shape_.Size(), rvalue.shape_.Size());
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          Kernel<BinaryOpBackwardUseIn<OP, Req>, xpu>::Launch(
            s, lvalue.shape_.Size(),
            out[iter_out].dptr_,
            ogradvalue.dptr_,
            lvalue.dptr_,
            rvalue.dptr_);
        });
        ++num_common_rows;
      } else if (idx_ograd == idx_l) {

        ++num_common_rows;
      } else if (idx_l == idx_r) {

        ++num_common_rows;
      } else if (idx_ograd < idx_l && idx_ograd < idx_r) {

      } else if (idx_l < idx_ograd && idx_l < idx_r) {

      } else if (idx_r < idx_ograd && idx_r < idx_l) {

      } else {
        CHECK(false);  // SHouldn't get here
      }
    }
/*
    while (iter_l < num_rows_l && iter_r < num_rows_r) {
      const IType idx_l = indices_l[iter_l];
      const IType idx_r = indices_r[iter_r];
      if (idx_l == idx_r) {
        // Same row
        indices_out[iter_out] = idx_l;
        Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
        Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
        DCHECK_EQ(lvalue.shape_.Size(), rvalue.shape_.Size());
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          Kernel<BMap<OP, Req>, xpu>::Launch(
            s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_, rvalue.dptr_);
        });
        num_common_rows++;
      } else if (idx_l < idx_r) {
        // Left only
        indices_out[iter_out] = idx_l;
        Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          Kernel<BinaryOpMissingRValue<OP, Req>, xpu>::Launch(
            s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
        });
      } else {
        // Right only
        indices_out[iter_out] = idx_r;
        Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          Kernel<BinaryOpMissingLValue<OP, Req>, xpu>::Launch(
            s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
        });
      }
      iter_out++;
    }
    // Evaluate the remaining rows beyond the l and r value row intersetion
    while (iter_l < num_rows_l) {
      indices_out[iter_out] = indices_l[iter_l];
      Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpMissingRValue<OP, Req>, xpu>::Launch(
          s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
      });
    }
    while (iter_r < num_rows_r) {
      indices_out[iter_out] = indices_r[iter_r];
      Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpMissingLValue<OP, Req>, xpu>::Launch(
          s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
      });
    }
    nnvm::TShape new_shape = output->aux_shape(rowsparse::kIdx);
    new_shape[0] -= num_common_rows;  // Reduce the first-dimension size by the number of common rows
    output->set_aux_shape(rowsparse::kIdx, new_shape);
*/
#endif
    if (tempSparse.get()) {
      // Required output is actually something other than RSP,
      // so cast out final RSP to the true output type
      CastStorageComputeImpl(s, *tempSparse, outputs[0]);
    }
  }
#endif

  template<typename xpu, typename DType, typename OP>
  static inline size_t FillDense(mshadow::Stream<xpu> *s,
                                 const size_t idx_l,
                                 const size_t idx_r,
                                 const OpReqType req,
                                 mshadow::Tensor<xpu, 2, DType>& out,
                                 const size_t iter_out) {
    using namespace mxnet_op;
    using namespace mshadow::expr;
    const int index_out_min = std::min(idx_l, idx_r);
    if (index_out_min > iter_out) {
      const size_t size = out[iter_out].shape_.Size();
      const DType zero_input_val = OP::Map(DType(0), DType(0));
//      std::cout << "FillDense( " << iter_out << " - " << (index_out_min-1) << " ) = "
//                << zero_input_val
//                << std::endl << std::flush;
      #pragma omp parallel for
      for(int i = static_cast<int>(iter_out), n = index_out_min; i < n; ++i) {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<MapIdentity<Req>, xpu>::Launch(s, size, out[i].dptr_, zero_input_val);
        });
      }
    }
    return static_cast<size_t>(index_out_min);
  }

  template<typename DType>
  static inline bool IsSameArray(const NDArray *a1, const NDArray *a2) {
    if (a1 && a2) {
      if (a1 == a2) {
        return true;
      }
      if (a1->ctx().dev_type == a2->ctx().dev_type) {
        const DType *pa1 = a1->data().dptr<DType>();
        if (pa1 && pa1 == a2->data().dptr<DType>()) {
          return true;
        }
      }
    }
    return false;
  }

  // TODO(cjolivier01) Precompute parallelizing strategy
  // TODO(cjolivier01) Optimize: change some bool parameters and internally-computed
  //                   bool variables (i.e. rhs_is_dense) to template parameters
  template<typename xpu, typename DType, typename IType, typename OP>
  static inline void RspRspElemwiseBinaryOp2(const nnvm::NodeAttrs &attrs,
                                             const OpContext &ctx,
                                             const NDArray& lhs,
                                             const NDArray& rhs,
                                             const OpReqType req,
                                             const NDArray& output,
                                             const bool rhs_may_be_dense,
                                             const bool allow_inplace) {
    using namespace mshadow;
    using namespace mxnet_op;
    using namespace mshadow::expr;

    //test::print(&std::cout, "lhs", lhs);
    //test::print(&std::cout, "rhs", rhs);

    const bool is_dense_result = output.storage_type() == kDefaultStorage;
    const bool rhs_is_dense = rhs.storage_type() == kDefaultStorage;
    CHECK(!rhs_is_dense || rhs_may_be_dense) << "rvalue cannot be dense";
    if (rhs_is_dense) {
      // For right-side dense, lhs input zero should always output zero
      CHECK(fabs(OP::Map(0, 99)) < 1e-4f);
      CHECK(!is_dense_result);  // Currently not handled
    }

    // Memory Estimation: This is (roughly) the number of result rows. We still
    // need to subtract the number of common rows
    bool lhs_in_place = false, rhs_in_place = false;
    const size_t num_rows_l = lhs.aux_shape(rowsparse::kIdx).Size();
    const size_t num_rows_r = rhs_is_dense ? rhs.shape()[0] : rhs.aux_shape(rowsparse::kIdx).Size();
    if (is_dense_result) {
      output.CheckAndAlloc();
    } else {
      if (rhs_is_dense) {
        output.CheckAndAlloc({mshadow::Shape1(num_rows_l)});
      } else {
        lhs_in_place = IsSameArray<DType>(&lhs, &output);
        rhs_in_place = IsSameArray<DType>(&rhs, &output);
        if (!lhs_in_place && !rhs_in_place) {
          output.CheckAndAlloc({mshadow::Shape1(num_rows_l + num_rows_r)});
        } else {
          CHECK_EQ(allow_inplace, true);
          CHECK_EQ(is_dense_result, false);
          if(lhs_in_place) {
            // For in-place, zero L-value must always be zero output
            //CHECK(fabs(float(OP::Map(DType(0), DType(99)))) < DType(1e-3));
          } else {
            // For in-place, zero R-value must always be zero output
            //CHECK(fabs(float(OP::Map(DType(99), DType(0)))) < DType(1e-3));
          }
        }
      }
    }
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

    // Indices
    Tensor<xpu, 1, IType> indices_l = lhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_r = rhs_is_dense
                                      ? Tensor<xpu, 1, IType>()
                                      : rhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_out = is_dense_result
                                        ? Tensor<xpu, 1, IType>()
                                        : output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);

    // Data
    // TODO(cjolivier01): Change to get_with_shape() calls
    Tensor<xpu, 2, DType> data_l = lhs.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> data_r = rhs.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = output.data().FlatTo2D<xpu, DType>(s);

    size_t iter_l = 0;
    size_t iter_r = 0;
    size_t iter_out = 0;
    int32_t num_common_rows = 0;

    if (is_dense_result) {
      if (!num_rows_l && !num_rows_r) {
        const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
        iter_out = FillDense<xpu, DType, OP>(s, all_rows, all_rows, req, out, iter_out);
      }
    }

    while (iter_l < num_rows_l && iter_r < num_rows_r) {
      IType idx_l = indices_l[iter_l];
      IType idx_r = rhs_is_dense ? idx_l : indices_r[iter_r];
      if(lhs_in_place) {
        while(idx_r < idx_l && ++iter_r < num_rows_r) {
          idx_r = indices_r[iter_r];
        }
        if(iter_r >= num_rows_r) {
          break;
        }
      } else if(rhs_in_place) {
        while(idx_l < idx_r && ++iter_l < num_rows_l) {
          idx_l = indices_l[iter_l];
        }
        if(iter_l >= num_rows_l) {
          break;
        }
      }
      if (is_dense_result) {
        iter_out = FillDense<xpu, DType, OP>(s, idx_l, idx_r, req, out, iter_out);
        DCHECK_EQ(iter_out, std::min(idx_l, idx_r));
      }
      if (idx_l == idx_r) {
        // Same row
        if (!is_dense_result) {
          indices_out[iter_out] = idx_l;
        }
        Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
        Tensor<xpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
        DCHECK_EQ(lvalue.shape_.Size(), rvalue.shape_.Size());
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<BMap<OP, Req>, xpu>::Launch(
            s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_, rvalue.dptr_);
        });
        num_common_rows++;
      } else if (idx_l < idx_r) {
        // Left only
        if (!is_dense_result) {
          indices_out[iter_out] = idx_l;
        }
        Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<BinaryOpMissingRValue<OP, Req>, xpu>::Launch(
            s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
        });
      } else {
        // Right only
        if (!is_dense_result) {
          indices_out[iter_out] = idx_r;
        }
        Tensor<xpu, 1, DType> rvalue = !rhs_is_dense ? data_r[iter_r++] : data_r[idx_r];
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<BinaryOpMissingLValue<OP, Req>, xpu>::Launch(
            s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
        });
      }
      iter_out++;
    }
    // Evaluate the remaining rows beyond the l and r value row intersetion
    while (iter_l < num_rows_l && !rhs_in_place) {
      if (!is_dense_result) {
        indices_out[iter_out] = indices_l[iter_l];
      } else {
        const IType idx_l = indices_l[iter_l];
        iter_out = FillDense<xpu, DType, OP>(s, lhs.shape()[0], idx_l, req, out, iter_out);
      }
      Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        Kernel<BinaryOpMissingRValue<OP, Req>, xpu>::Launch(
          s, lvalue.shape_.Size(), out[iter_out++].dptr_, lvalue.dptr_);
      });
    }
    while (iter_r < num_rows_r && !rhs_is_dense && !lhs_in_place) {
      if (!is_dense_result) {
        indices_out[iter_out] = indices_r[iter_r];
      } else {
        const IType idx_r = indices_r[iter_r];
        iter_out = FillDense<xpu, DType, OP>(s, lhs.shape()[0], idx_r, req, out, iter_out);
      }
      Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
      MXNET_ASSIGN_REQ_SWITCH(req, Req, {
        Kernel<BinaryOpMissingLValue<OP, Req>, xpu>::Launch(
          s, rvalue.shape_.Size(), out[iter_out++].dptr_, rvalue.dptr_);
      });
    }
    if (is_dense_result) {
      const size_t all_rows = static_cast<size_t>(lhs.shape()[0]);
      iter_out = FillDense<xpu, DType, OP>(s, all_rows, all_rows, req, out, iter_out);
      //test::print(&std::cout, "output", *output);
    } else {
      if(lhs_in_place) {
        CHECK_LE(iter_out, num_rows_l);
      }
      if(rhs_in_place) {
        CHECK_LE(iter_out, num_rows_r);
      }
      DCHECK_LE(iter_out, num_rows_l + num_rows_r);  // Make sure that we didn't overrun
      nnvm::TShape new_shape = output.aux_shape(rowsparse::kIdx);
      CHECK_LE(iter_out, new_shape.Size());
      if (!rhs_is_dense && !lhs_in_place && !rhs_in_place) {
        // Reduce the first-dimension size by the number of common rows
        new_shape[0] -= num_common_rows;
        output.set_aux_shape(rowsparse::kIdx, new_shape);
//        const size_t matrix_size = output.shape().Size() / output.shape()[0];
//        const_cast<NDArray &>(output).set_storage_shape(TShape({new_shape[0],
//                                                                index_t(matrix_size)}));
      }
      //test::print(&std::cout, "output", *output);
    }
  }

  /*! \brief Minimum of three */
  static MSHADOW_XINLINE size_t minthree(const size_t a, const size_t b, const size_t c) {
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
  }

  /*! \brief Maximum of three */
  static MSHADOW_XINLINE size_t maxthree(const size_t a, const size_t b, const size_t c) {
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
  }

  template<typename DType>
  static MSHADOW_XINLINE int LaunchSize(const size_t sz) {
    return static_cast<int>((sz + mxnet_op::DataType<DType>::kLanes - 1)
                     / mxnet_op::DataType<DType>::kLanes);
  }

//  // Binary Compute between two row-sparse ndarray
//  // This implementation only works on CPU
//  template<typename xpu, typename OP, WithHalf2 with_half2 = WithHalf2::WITHOUT_HALF2>
//  static void ComputeRspRsp(const nnvm::NodeAttrs &attrs,
//                            const OpContext &ctx,
//                            const NDArray &lhs,
//                            const NDArray &rhs,
//                            const OpReqType req,
//                            const NDArray &output,
//                            const bool rhs_may_be_dense = false,
//                            const bool allow_inplace = false) {
//    MSHADOW_TYPE_SWITCH(lhs.aux_type(rowsparse::kIdx), IType, {
//      if(with_half2 == WithHalf2::WITHOUT_HALF2) {
//        MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
//          RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
//            attrs, ctx, lhs, rhs, req, output,
//            rhs_may_be_dense, allow_inplace);
//        })
//      } else {
//        MSHADOW_TYPE_SWITCH_WITH_HALF2(output.dtype(), DType, {
//          RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
//            attrs, ctx, lhs, rhs, req, output,
//            rhs_may_be_dense, allow_inplace);
//        })
//      }
//    });
//  }

//  template<typename xpu, typename OP>
//  static void ComputeRspRsp3(const nnvm::NodeAttrs &attrs,
//                            const OpContext &ctx,
//                            const std::vector<NDArray> &inputs,
//                            const std::vector<OpReqType> &req,
//                            const std::vector<NDArray> &outputs) {
//    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
//      MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
//        RspRspElemwiseBinaryOp3<xpu, DType, IType, OP>(attrs, ctx, inputs, req, outputs);
//      })
//    });
//  }

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BinaryBackwardUseNone_(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
                                / DataType<DType>::kLanes);
    DCHECK_EQ(size, outputs[0].Size());
    DCHECK_EQ(size, outputs[1].Size());
    DCHECK_EQ(size, inputs[0].Size());
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    if (std::is_same<LOP, mshadow_op::identity>::value && req[0] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[0].dptr<DType>());
    } else if (req[0] != kNullOp) {
      DType *lgrad_dptr = outputs[0].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpBackwardUseNone<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr);
      });
    }
    if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, outputs[1].dptr<DType>());
    } else if (req[1] != kNullOp) {
      DType *rgrad_dptr = outputs[1].dptr<DType>();
      MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
        Kernel<BinaryOpBackwardUseNone<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr);
      });
    }
  }

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BinaryBackwardUseIn_(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
    DCHECK_EQ(outputs.size(), 2U);
    DCHECK_EQ(inputs.size(), 3U);
//    for(size_t x = 0, n = inputs.size(); x < n; ++x) {
//      std::stringstream ss;
//      ss << "BinaryBackwardUseIn_(): inputs[" << x << "]: ";
//      test::print_blob(&std::cout, ss.str(), inputs[x]);
//    }
    mxnet_op::Stream<xpu> *s = ctx.get_stream<xpu>();
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    const DType *lhs_dptr = inputs[1].dptr<DType>();
    const DType *rhs_dptr = inputs[2].dptr<DType>();
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      const int size = static_cast<int>(
        (outputs[0].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * lgrad_dptr = outputs[0].dptr<DType>();
      mxnet_op::Kernel<BinaryOpBackwardUseIn<LOP, Req>, xpu>::Launch(
        s, size, lgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
      const int size = static_cast<int>(
        (outputs[1].Size() + mxnet_op::DataType<DType>::kLanes - 1)
        / mxnet_op::DataType<DType>::kLanes);
      DType * rgrad_dptr = outputs[1].dptr<DType>();
      mxnet_op::Kernel<BinaryOpBackwardUseIn<ROP, Req>, xpu>::Launch(
        s, size, rgrad_dptr, ograd_dptr, lhs_dptr, rhs_dptr);});
//    test::print_blob(&std::cout, "output[0]", outputs[0]);
//    test::print_blob(&std::cout, "output[1]", outputs[1]);
  }

 public:
  template<typename xpu, typename OP>
  static void Launch(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] != kNullOp) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(inputs.size(), 2U);
      CHECK_EQ(outputs.size(), 1U);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
          // Why is 'size' necessary?
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          Kernel<BMap<OP, Req>, xpu>::Launch(s, size,
          outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void LaunchWithHalf2(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] != kNullOp) {
      Stream<xpu> *s = ctx.get_stream<xpu>();
      CHECK_EQ(inputs.size(), 2U);
      CHECK_EQ(outputs.size(), 1U);
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
          // Why is 'size' necessary?
          const size_t size = (minthree(outputs[0].Size(), inputs[0].Size(), inputs[1].Size())
          + DataType<DType>::kLanes - 1) / DataType<DType>::kLanes;
          Kernel<BMap<OP, Req>, xpu>::Launch(s, size,
          outputs[0].dptr<DType>(),
          inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
        });
      });
    }
  }

  template<typename xpu, typename OP>
  static void LaunchEx(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<NDArray> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] != kNullOp) {
//      for(size_t x = 0, n = inputs.size(); x < n; ++x) {
//        std::stringstream ss;
//        ss << "LaunchEx(): inputs[" << x << "]: ";
//        test::print(&std::cout, ss.str(), inputs[x]);
//      }
      // If any input or output is dense, fallback to FCompute
      // TODO(haibin) implement dns + rsp in a separate kernel
      if (!common::ContainsDefaultStorage(inputs)) {
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
              attrs, ctx, inputs[0], inputs[1],
              req[0], outputs[0],
              false, false);
          });
        });
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             Launch<xpu, OP>, "LaunchEx");
      }
//      test::print(&std::cout, "output[0]", outputs[0]);
    }
  }

//  template<typename xpu, typename OP>
//  static void LaunchExWithHalf2(const nnvm::NodeAttrs &attrs,
//                       const OpContext &ctx,
//                       const std::vector<NDArray> &inputs,
//                       const std::vector<OpReqType> &req,
//                       const std::vector<NDArray> &outputs) {
//    using namespace mshadow;
//    using namespace mshadow::expr;
//    CHECK_EQ(inputs.size(), 2);
//    CHECK_EQ(outputs.size(), 1);
//    if (req[0] != kNullOp) {
//      // If any input or output is dense, fallback to FCompute
//      // TODO(haibin) implement dns + rsp in a separate kernel
//      if (!common::ContainsDefaultStorage(inputs)) {
//        MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].dtype(), DType, {
//          RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
//            attrs, ctx, inputs[0], inputs[1],
//            req[0], outputs[0],
//            false, false);
//        });
//      } else {
//        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
//                             Launch<xpu, OP>, "LaunchEx");
//      }
//    }
//  }

//  template<typename xpu, typename OP>
//  static void LaunchExWithHalf2(const nnvm::NodeAttrs &attrs,
//                       const OpContext &ctx,
//                       const std::vector<NDArray> &inputs,
//                       const std::vector<OpReqType> &req,
//                       const std::vector<NDArray> &outputs) {
//    using namespace mshadow;
//    using namespace mshadow::expr;
//    CHECK_EQ(inputs.size(), 2);
//    CHECK_EQ(outputs.size(), 1);
//    if (req[0] != kNullOp) {
////      for(size_t x = 0, n = inputs.size(); x < n; ++x) {
////        std::stringstream ss;
////        ss << "LaunchEx(): inputs[" << x << "]: ";
////        test::print(&std::cout, ss.str(), inputs[x]);
////      }
//      // If any input or output is dense, fallback to FCompute
//      // TODO(haibin) implement dns + rsp in a separate kernel
//      if (!common::ContainsDefaultStorage(inputs)) {
//        ComputeRspRsp<xpu, OP, >(attrs, ctx, inputs[0], inputs[1],
//                               req[0], outputs[0]);
//      } else {
//        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
//                             Launch<xpu, OP>, "LaunchEx");
//      }
////      test::print(&std::cout, "output[0]", outputs[0]);
//    }
//  }

  /*! \brief LaunchEx allowing dense rvalue */
  template<typename xpu, typename OP>
  static void LaunchExDenseRValue(const nnvm::NodeAttrs &attrs,
                                  const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
//    for(size_t x = 0, n = inputs.size(); x < n; ++x) {
//      std::stringstream ss;
//      ss << "LaunchExDenseRValue(): inputs[" << x << "]: ";
//      test::print(&std::cout, ss.str(), inputs[x]);
//    }
    if (req[0] != kNullOp) {
      // If any input or output is dense, fallback to FCompute
      if (inputs[0].storage_type() != kDefaultStorage) {
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
              attrs, ctx, inputs[0], inputs[1],
              req[0], outputs[0],
              true, false);
          });
        });
      } else {
        // May be lhs=dense, rhs=sparse
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             Launch<xpu, OP>, "LaunchEx");
      }
    }
//    test::print(&std::cout, "outputs[0]", outputs[0]);
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNone(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneEx(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 1U);  // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    using namespace mshadow;
    using namespace mshadow::expr;
    auto stype = inputs[0].storage_type();
    CHECK_EQ(stype, kRowSparseStorage) << "Not implemented yet";
    if (req[0] != kNullOp) {
      // If any input is dense, fallback to FCompute
      if (!common::ContainsDefaultStorage(inputs)) {
        CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage);
        CHECK_EQ(inputs.size(), 1U);
        DCHECK_LT(fabs(LOP::Map(0)), 1e-5f);  // op requires 0-input returns 0-output (sparse<->sparse)
        DCHECK_LT(fabs(ROP::Map(0)), 1e-5f);  // op requires 0-input returns 0-output (sparse<->sparse)
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          UnaryOp::LaunchEx<xpu, BinaryOpBackwardUseNone<LOP, Req>>(attrs, ctx, inputs,
                                                                    req, {outputs[0]});
        });
        MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
          UnaryOp::LaunchEx<xpu, BinaryOpBackwardUseNone<ROP, Req>>(attrs, ctx, inputs,
                                                                    req, {outputs[1]});
        });
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             BinaryBackwardUseNone<xpu, LOP, ROP>,
                             "BinaryBackwardUseNoneEx");
      }
    }
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseIn(const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<TBlob> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseInWithHalf2(const nnvm::NodeAttrs &attrs,
                                                  const OpContext &ctx,
                                                  const std::vector<TBlob> &inputs,
                                                  const std::vector<OpReqType> &req,
                                                  const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename LOP, typename ROP, WithHalf2 with_half2 = WithHalf2::WITHOUT_HALF2>
  static inline void BinaryBackwardUseInEx(const nnvm::NodeAttrs &attrs,
                                           const OpContext &ctx,
                                           const std::vector<NDArray> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 3U);  // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
//    for(size_t x = 0, n = inputs.size(); x < n; ++x) {
//      std::stringstream ss;
//      ss << "BinaryBackwardUseInEx(): inputs[" << x << "]: ";
//      test::print(&std::cout, ss.str(), inputs[x]);
//    }
    if (req[0] != kNullOp) {
      // If any input is dense, fallback to FCompute
      // TODO(haibin) implement dns + rsp in a separate kernel
      if (!common::ContainsDefaultStorage(inputs)) {
        // ComputeRspRsp can handle dense outputs so long as OP(0, 0) == 0
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, LOP>(
              attrs, ctx, inputs[1], inputs[2], req[0], outputs[0],
              false, false
            );
          });
        });
        //test::print(&std::cout, "output[0]", outputs[0]);
        // LHS in-place
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, mshadow::op::mul>(
              attrs, ctx, outputs[0], inputs[0], req[0], outputs[0], false, true);
          });});
        //test::print(&std::cout, "output[0]", outputs[0]);
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH(outputs[1].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, ROP>(
              attrs, ctx, inputs[1], inputs[2], req[1], outputs[1],
              false, false);
          });});
        //test::print(&std::cout, "output[1]", outputs[1]);
        // RHS in-place
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH(outputs[1].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, mshadow::op::mul>(
              attrs, ctx, inputs[0], outputs[1], req[1], outputs[1],
              false, true);
          });});
        //test::print(&std::cout, "output[1]", outputs[1]);
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             BinaryBackwardUseIn<xpu, LOP, ROP>,
                             "BinaryBackwardUseInEx");
      }
    }
//    test::print(&std::cout, "output[0]", outputs[0]);
//    test::print(&std::cout, "output[1]", outputs[1]);
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseInExDense(const nnvm::NodeAttrs &attrs,
                                                const OpContext &ctx,
                                                const std::vector<NDArray> &inputs,
                                                const std::vector<OpReqType> &req,
                                                const std::vector<NDArray> &outputs) {
//    for(size_t x = 0, n = inputs.size(); x < n; ++x) {
//      std::stringstream ss;
//      ss << "BinaryBackwardUseInExDense(): inputs[" << x << "]: ";
//      test::print(&std::cout, ss.str(), inputs[x]);
//    }
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                         BinaryBackwardUseIn<xpu, LOP, ROP>, "BinaryBackwardUseInExDense");
  }
};  // class ElemwiseBinaryOp

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

/*! \brief Binary launch */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU(__name$, __kernel$)                \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                            \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>)       \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::LaunchEx<cpu, __kernel$>)

/*! \brief Binary launch, dense result */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DR(__name$, __kernel$)               \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                              \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageTypeDenseOutput<1>) \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::LaunchEx<cpu, __kernel$>)

/*! \brief Binary launch, dense rvalue */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DENSE_RVALUE(__name$, __kernel$)            \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                                     \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageTypeForce<2, 1, 0>)        \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<cpu>", ElemwiseBinaryOp::LaunchExDenseRValue<cpu, __kernel$>)

/*! \brief Binary CUDA launch */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA(__name$, __kernel$)               \
  NNVM_REGISTER_OP(__name$)                                                          \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<gpu, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchEx<gpu, __kernel$>)

/*! \brief Binary CUDA launch, dense result */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_DR(__name$, __kernel$)                       \
  NNVM_REGISTER_OP(__name$)                                                                     \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<gpu, __kernel$>)                \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchEx<gpu, __kernel$>)

/*! \brief Binary CUDA launch, dense rvalue */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_DENSE_RVALUE(__name$, __kernel$)           \
  NNVM_REGISTER_OP(__name$)                                                                   \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchExDenseRValue<gpu, __kernel$>)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
