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


 private:
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
      for(int i = static_cast<int>(iter_out); i < index_out_min; ++i) {
        MXNET_ASSIGN_REQ_SWITCH(req, Req, {
          Kernel<MapSetToScalar<Req>, xpu>::Launch(s, size, out[i].dptr_, zero_input_val);
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

  /*! \brief For some types or sparse, we can assume 100% density can be computer
   * more quickly with the standard dense procedure
   * @param arr the array to test
   * @return bool, whether the array is effectively dense
   */
  static inline bool IsEffectivelyDense(const NDArray& arr) {
    switch(arr.storage_type()) {
      case kDefaultStorage:
        return true;
      case kRowSparseStorage:
        return arr.shape().Size() == arr.aux_shape(rowsparse::kIdx).Size();
      default:
        return false;
    }
  }

  // TODO(cjolivier01) Precompute parallelizing strategy
  // TODO(cjolivier01) Optimize: change some bool parameters and internally-computed
  //                   bool variables (i.e. rhs_is_dense) to template parameters
  template<typename xpu, typename DType, typename IType, typename OP>
  static void RspRspElemwiseBinaryOp2(const nnvm::NodeAttrs &attrs,
                                             const OpContext &ctx,
                                             const NDArray& lhs,
                                             const NDArray& rhs,
                                             const OpReqType req,
                                             const NDArray& output,
                                             const bool lhs_may_be_dense,
                                             const bool rhs_may_be_dense,
                                             const bool allow_inplace) {
    using namespace mshadow;
    using namespace mxnet_op;
    using namespace mshadow::expr;

    const bool is_dense_result = output.storage_type() == kDefaultStorage;
    const bool lhs_is_dense = lhs.storage_type() == kDefaultStorage;
    const bool rhs_is_dense = rhs.storage_type() == kDefaultStorage;
    CHECK(!lhs_is_dense || lhs_may_be_dense) << "rvalue cannot be dense";
    CHECK(!rhs_is_dense || rhs_may_be_dense) << "rvalue cannot be dense";
    CHECK(!lhs_is_dense || !rhs_is_dense);
    if (rhs_is_dense) {
      // For right-side dense, lhs input zero should always output zero
      CHECK(fabs(OP::Map(0, 99)) < 1e-4f);
      CHECK(!is_dense_result);  // Currently not handled
    }
    if (lhs_is_dense) {
      // For right-side dense, lhs input zero should always output zero
      CHECK(fabs(OP::Map(99, 0)) < 1e-4f);
      CHECK(!is_dense_result);  // Currently not handled
    }

    // Memory Estimation: This is (roughly) the number of result rows. We still
    // need to subtract the number of common rows
    bool lhs_in_place = false, rhs_in_place = false;
    const size_t num_rows_l = lhs_is_dense ? lhs.shape()[0] : lhs.aux_shape(rowsparse::kIdx).Size();
    const size_t num_rows_r = rhs_is_dense ? rhs.shape()[0] : rhs.aux_shape(rowsparse::kIdx).Size();
    if (is_dense_result) {
      output.CheckAndAlloc();
    } else {
      if (rhs_is_dense) {
        output.CheckAndAlloc({mshadow::Shape1(num_rows_l)});
      } else if(lhs_is_dense) {
        output.CheckAndAlloc({mshadow::Shape1(num_rows_r)});
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
    Tensor<xpu, 1, IType> indices_l = lhs_is_dense
                                      ? Tensor<xpu, 1, IType>()
                                      : lhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_r = rhs_is_dense
                                      ? Tensor<xpu, 1, IType>()
                                      : rhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_out = is_dense_result
                                        ? Tensor<xpu, 1, IType>()
                                        : output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);

    // Data
    // TODO(cjolivier01): Change to get_with_shape() calls
    Tensor<xpu, 2, DType> data_l = AsRowise2D<DType>(s, lhs.data());
    Tensor<xpu, 2, DType> data_r = AsRowise2D<DType>(s, rhs.data());
    Tensor<xpu, 2, DType> out = AsRowise2D<DType>(s, output.data());

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
      IType idx_l = lhs_is_dense ? indices_r[iter_r] : indices_l[iter_l];
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
        Tensor<xpu, 1, DType> lvalue = !lhs_is_dense ? data_l[iter_l++] : data_l[idx_l];
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
        Tensor<xpu, 1, DType> lvalue = !lhs_is_dense ? data_l[iter_l++] : data_l[idx_l];
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
    while (iter_l < num_rows_l && !lhs_is_dense && !rhs_in_place) {
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
      if (!rhs_is_dense && !lhs_is_dense && !lhs_in_place && !rhs_in_place) {
        // Reduce the first-dimension size by the number of common rows
        new_shape[0] -= num_common_rows;
        output.set_aux_shape(rowsparse::kIdx, new_shape);
      }
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

  /*! \brief LaunchEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, typename DType,
    bool lhs_may_be_dense, bool rhs_may_be_dense, typename BackupCompute>
  static void LaunchExDenseLRValue_(const nnvm::NodeAttrs &attrs,
                                    const OpContext &ctx,
                                    const std::vector<NDArray> &inputs,
                                    const std::vector<OpReqType> &req,
                                    const std::vector<NDArray> &outputs,
                                    BackupCompute backup_compute) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    //PRINT_OP_AND_ARRAYS(OP, inputs);
    if (req[0] != kNullOp) {
      const NDArray *sparse = &inputs[0];
      if(sparse->storage_type() == kDefaultStorage) {
        sparse = &inputs[1];
        if(sparse->storage_type() == kDefaultStorage) {
          // Do we need to worry about sparse result here?
          CHECK_EQ(outputs[0].storage_type(), kDefaultStorage);
          MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Launch<xpu, OP>);
          return;
        }
      }
      bool allowed = false;
      if(lhs_may_be_dense && rhs_may_be_dense) {
        allowed = common::ContainsNonDefaultStorage(inputs);
      } else if (lhs_may_be_dense) {
        allowed = inputs[1].storage_type() != kDefaultStorage;
      } else if(rhs_may_be_dense) {
        allowed = inputs[0].storage_type() != kDefaultStorage;
      } else {
        allowed = !common::ContainsNonDefaultStorage(inputs);
      }
      // If any input or output is dense, fallback to FCompute
      if (allowed) {
        MSHADOW_TYPE_SWITCH(sparse->aux_type(rowsparse::kIdx), IType, {
          RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
            attrs, ctx, inputs[0], inputs[1],
            req[0], outputs[0],
            lhs_may_be_dense, rhs_may_be_dense, false);
        });
      } else {
        // May be lhs=dense, rhs=sparse
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             backup_compute,
                             "LaunchExDenseLRValue_");
      }
    }
    //PRINT_OP_AND_ARRAYS(OP, outputs);
  }

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
    //PRINT_NDARRAYS(inputs);
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
    //PRINT_NDARRAYS(outputs);
  }

  template<
    typename xpu,
    typename LOP,
    typename ROP,
    typename DType,
    bool in0_ok_dense = false,
    bool in1_ok_dense = false,
    bool in2_ok_dense = false,
    typename BackupCompute>
  static inline void BinaryBackwardUseInEx_(const nnvm::NodeAttrs &attrs,
                                           const OpContext &ctx,
                                           const std::vector<NDArray> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<NDArray> &outputs,
                                           BackupCompute backup_compute) {
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, inputs);
    CHECK_EQ(inputs.size(), 3U);  // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    if (req[0] != kNullOp) {
      // If any input is dense, fallback to FCompute
      // TODO(haibin) implement dns + rsp in a separate kernel
//      bool allow = true;
//      if(!in0_ok_dense && inputs[0].storage_type() == kDefaultStorage) {
//        allow = false;
//      }
//      if(allow && !in1_ok_dense && inputs[1].storage_type() == kDefaultStorage) {
//        allow = false;
//      }
//      if(allow && !in2_ok_dense && inputs[2].storage_type() == kDefaultStorage) {
//        allow = false;
//      }
      if (!common::ContainsDefaultStorage(inputs)) {
        // ComputeRspRsp can handle dense outputs so long as OP(0, 0) == 0
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, LOP>(
              attrs, ctx, inputs[1], inputs[2], req[0], outputs[0],
              false, false, false
            );
        });
        //test::print(&std::cout, "output[0]", outputs[0]);
        // LHS in-place
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, mshadow::op::mul>(
              attrs, ctx, outputs[0], inputs[0], req[0], outputs[0],
              false, false, true);
        });
        //test::print(&std::cout, "output[0]", outputs[0]);
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, ROP>(
              attrs, ctx, inputs[1], inputs[2], req[1], outputs[1],
              false, false, false);
        });
        //test::print(&std::cout, "output[1]", outputs[1]);
        // RHS in-place
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, mshadow::op::mul>(
              attrs, ctx, inputs[0], outputs[1], req[1], outputs[1],
              false, false, true);
        });
        //test::print(&std::cout, "output[1]", outputs[1]);
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             backup_compute,
                             "BinaryBackwardUseInEx_");
      }
    }
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, outputs);
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
      //PRINT_OP_AND_ARRAYS(OP, inputs);
      // If any input or output is dense, fallback to FCompute
      // TODO(haibin) implement dns + rsp in a separate kernel
      if (!common::ContainsDefaultStorage(inputs)) {
        switch(inputs[0].storage_type()) {
          case kRowSparseStorage:
            MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
              MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
                RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
                  attrs, ctx, inputs[0], inputs[1],
                  req[0], outputs[0],
                  false, false, false);
              });
            });
            break;
          case kCSRStorage:
            CHECK(false);
            break;
        }
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             Launch<xpu, OP>, "LaunchEx");
      }
      //PRINT_OP_AND_ARRAYS(OP, outputs);
    }
  }

  template<typename xpu, typename OP>
  static void LaunchWithHalf2Ex(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    if (req[0] != kNullOp) {
      if (!common::ContainsDefaultStorage(inputs)) {
        MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
          MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].dtype(), DType, {
            RspRspElemwiseBinaryOp2<xpu, DType, IType, OP>(
              attrs, ctx, inputs[0], inputs[1],
              req[0], outputs[0],
              false, false, false);
          });
        });
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             LaunchWithHalf2<xpu, OP>, "LaunchWithHalf2Ex");
      }
    }
  }

  /*! \brief LaunchEx allowing dense lvalue and/or rvalue */
  template<typename xpu, typename OP, bool lhs_may_be_dense, bool rhs_may_be_dense>
  static void LaunchExDenseLRValue(const nnvm::NodeAttrs &attrs,
                                  const OpContext &ctx,
                                  const std::vector<NDArray> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      LaunchExDenseLRValue_<xpu, OP, DType, lhs_may_be_dense, rhs_may_be_dense>(
        attrs, ctx, inputs, req, outputs, Launch<xpu, OP>);
    });
  }

  /*! \brief LaunchEx allowing dense rvalue */
  template<typename xpu, typename OP, bool lhs_may_be_dense, bool rhs_may_be_dense>
  static void LaunchExDenseWithHalf2LRValue(const nnvm::NodeAttrs &attrs,
                                            const OpContext &ctx,
                                            const std::vector<NDArray> &inputs,
                                            const std::vector<OpReqType> &req,
                                            const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].dtype(), DType, {
      LaunchExDenseLRValue_<xpu, OP, DType, lhs_may_be_dense, rhs_may_be_dense>(
        attrs, ctx, inputs, req, outputs, LaunchWithHalf2<xpu, OP>);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNone(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, inputs);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, outputs);
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneWithHalf2(const nnvm::NodeAttrs &attrs,
                                                    const OpContext &ctx,
                                                    const std::vector<TBlob> &inputs,
                                                    const std::vector<OpReqType> &req,
                                                    const std::vector<TBlob> &outputs) {
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, inputs);
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, outputs);
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneEx(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 1U);  // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, inputs);
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
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, outputs);
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneWithHalf2Ex(const nnvm::NodeAttrs &attrs,
                                                      const OpContext &ctx,
                                                      const std::vector<NDArray> &inputs,
                                                      const std::vector<OpReqType> &req,
                                                      const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 1U);  // output grad,
    CHECK_EQ(outputs.size(), 2U);  // lhs input grad, rhs input grad
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, inputs);
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
          UnaryOp::LaunchWithHalf2Ex<xpu, BinaryOpBackwardUseNone<LOP, Req>>(
            attrs, ctx, inputs, req, {outputs[0]});
        });
        MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
          UnaryOp::LaunchWithHalf2Ex<xpu, BinaryOpBackwardUseNone<ROP, Req>>(
            attrs, ctx, inputs, req, {outputs[1]});
        });
      } else {
        FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                             BinaryBackwardUseNoneWithHalf2<xpu, LOP, ROP>,
                             "BinaryBackwardUseNoneWithHalf2Ex");
      }
    }
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, outputs);
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseIn(const nnvm::NodeAttrs &attrs,
                                         const OpContext &ctx,
                                         const std::vector<TBlob> &inputs,
                                         const std::vector<OpReqType> &req,
                                         const std::vector<TBlob> &outputs) {
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, inputs);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      BinaryBackwardUseIn_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
    //PRINT_OP2_AND_ARRAYS(LOP, ROP, outputs);
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

  template<
    typename xpu, typename LOP, typename ROP,
    bool in0_ok_dense = false, bool in1_ok_dense = false, bool in2_ok_dense = false>
  static inline void BinaryBackwardUseInEx(const nnvm::NodeAttrs &attrs,
                                           const OpContext &ctx,
                                           const std::vector<NDArray> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      BinaryBackwardUseInEx_<xpu, LOP, ROP, DType, in0_ok_dense, in1_ok_dense, in2_ok_dense>(
        attrs, ctx, inputs, req, outputs, BinaryBackwardUseIn<xpu, LOP, ROP>);
    });
  }

  template<
    typename xpu, typename LOP, typename ROP,
    bool in0_ok_dense = false, bool in1_ok_dense = false, bool in2_ok_dense = false>
  static inline void BinaryBackwardUseInWithHalf2Ex(const nnvm::NodeAttrs &attrs,
                                                    const OpContext &ctx,
                                                    const std::vector<NDArray> &inputs,
                                                    const std::vector<OpReqType> &req,
                                                    const std::vector<NDArray> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].dtype(), DType, {
      BinaryBackwardUseInEx_<xpu, LOP, ROP, DType, in0_ok_dense, in1_ok_dense, in2_ok_dense>(
        attrs, ctx, inputs, req, outputs, BinaryBackwardUseInWithHalf2<xpu, LOP, ROP>);
    });
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseInExDense(const nnvm::NodeAttrs &attrs,
                                                const OpContext &ctx,
                                                const std::vector<NDArray> &inputs,
                                                const std::vector<OpReqType> &req,
                                                const std::vector<NDArray> &outputs) {
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                         BinaryBackwardUseIn<xpu, LOP, ROP>,
                         "BinaryBackwardUseInExDense");
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseInWithHalf2ExDense(const nnvm::NodeAttrs &attrs,
                                                const OpContext &ctx,
                                                const std::vector<NDArray> &inputs,
                                                const std::vector<OpReqType> &req,
                                                const std::vector<NDArray> &outputs) {
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                         BinaryBackwardUseInWithHalf2<xpu, LOP, ROP>,
                         "BinaryBackwardUseInWithHalf2ExDense");
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
  .set_attr<FComputeEx>("FComputeEx<cpu>",                                                    \
    ElemwiseBinaryOp::LaunchExDenseLRValue<cpu, __kernel$, false, true>)

/*! \brief Binary launch, dense rvalue */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CPU_DENSE_LRVALUE(__name$, __kernel$)           \
  MXNET_OPERATOR_REGISTER_BINARY(__name$)                                                     \
  .set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageTypeLeastDense<2, 1>)      \
  .set_attr<FCompute>("FCompute<cpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<cpu>",                                                    \
    ElemwiseBinaryOp::LaunchExDenseLRValue<cpu, __kernel$, true, true>)

/*! \brief Binary CUDA launch */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA(__name$, __kernel$)               \
  NNVM_REGISTER_OP(__name$)                                                          \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<gpu, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchEx<gpu, __kernel$>)

/*! \brief Binary CUDA launch */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_WITH_HALF2(__name$, __kernel$)             \
  NNVM_REGISTER_OP(__name$)                                                                   \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::LaunchWithHalf2<gpu, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchWithHalf2Ex<gpu, __kernel$>)

/*! \brief Binary CUDA launch, dense result */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_DR(__name$, __kernel$)                       \
  NNVM_REGISTER_OP(__name$)                                                                     \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<gpu, __kernel$>)                \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchEx<gpu, __kernel$>)

/*! \brief Binary CUDA launch, dense result */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_WITH_HALF2_CUDA_DR(__name$, __kernel$)            \
  NNVM_REGISTER_OP(__name$)                                                                     \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::LaunchWithHalf2<gpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<gpu>", ElemwiseBinaryOp::LaunchWithHalf2Ex<gpu, __kernel$>)

/*! \brief Binary CUDA launch, dense rvalue */
#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_DENSE_RVALUE(__name$, __kernel$)           \
  NNVM_REGISTER_OP(__name$)                                                                   \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<gpu>",                                                    \
    ElemwiseBinaryOp::LaunchExDenseLRValue<gpu, __kernel$, false, true>)

#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_DENSE_LRVALUE(__name$, __kernel$)           \
  NNVM_REGISTER_OP(__name$)                                                                   \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::Launch<cpu, __kernel$>)              \
  .set_attr<FComputeEx>("FComputeEx<gpu>",                                                    \
    ElemwiseBinaryOp::LaunchExDenseLRValue<gpu, __kernel$, true, true>)

#define MXNET_OPERATOR_REGISTER_BINARY_LAUNCH_CUDA_WITH_HALF2_DENSE_LRVALUE(__name$, __kernel$) \
  NNVM_REGISTER_OP(__name$)                                                                     \
  .set_attr<FCompute>("FCompute<gpu>", ElemwiseBinaryOp::LaunchWithHalf2<cpu, __kernel$>)       \
  .set_attr<FComputeEx>("FComputeEx<gpu>",                                                      \
    ElemwiseBinaryOp::LaunchExDenseWithHalf2LRValue<gpu, __kernel$, true, true>)

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
