/*!
 *  Copyright (c) 2016 by Contributors
 * \file elemwise_binary_op.h
 * \brief Function definition of elementwise binary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_BINARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <string>
#include <utility>
#include <typeinfo>
#include "../mxnet_op.h"
#include "../mshadow_op.h"
#include "elemwise_unary_op.h"

namespace mxnet {
namespace op {

/*! Gather binary operator functions into BinaryOp class */
class BinaryOp : public OpBase
{
  /*! \brief Launchable call to binary operator */
  template<typename OP, int Req>
  struct BinaryOpMapping
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *out, const DType *lhs,
                                    const DType *rhs) {
      KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], rhs[i]));
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

  template<typename xpu, typename DType, typename IType, typename OP>
  static inline void RspRspElemwiseBinaryOp(const nnvm::NodeAttrs &attrs,
                                     const OpContext &ctx,
                                     const std::vector<NDArray> &inputs,
                                     const std::vector<OpReqType> &req,
                                     const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    using namespace mshadow::expr;

    auto &lhs = inputs[0];
    auto &rhs = inputs[1];
    auto &output = outputs[0];

    bool init_l = lhs.storage_initialized();
    bool init_r = rhs.storage_initialized();
    // both inputs are zeros
    if (!init_l && !init_r) return;
    // one of the input is zeros
    if (!init_l || !init_r) {
      NDArray out(output);
      CopyFromToRspImpl<xpu, xpu>(!init_l ? rhs : lhs, &out, ctx.run_ctx);
      return;
    }
    // Memory Estimation: This is (roughly) the number of result rows. We still
    // need to subtract the number of common rows
    const size_t num_rows_l = lhs.aux_shape(rowsparse::kIdx).Size();
    const size_t num_rows_r = rhs.aux_shape(rowsparse::kIdx).Size();
    output.CheckAndAlloc({mshadow::Shape1(num_rows_l + num_rows_r)});
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

    // Indices
    Tensor<xpu, 1, IType> indices_l = lhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_r = rhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
    Tensor<xpu, 1, IType> indices_out = output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);

    // Data
    Tensor<xpu, 2, DType> data_l = lhs.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> data_r = rhs.data().FlatTo2D<xpu, DType>(s);
    Tensor<xpu, 2, DType> out = output.data().FlatTo2D<xpu, DType>(s);

    // TODO(coolivie) May need to build a list of operations and then do those in
    // parallel.  Problem with that, however, is that the memory allocation will be expensive

    size_t iter_l = 0;
    size_t iter_r = 0;
    size_t iter_out = 0;
    int32_t num_common_rows = 0;
    while (iter_l < num_rows_l && iter_r < num_rows_r) {
      const IType idx_l = indices_l[iter_l];
      const IType idx_r = indices_r[iter_r];
      if (idx_l == idx_r) {
        // Same row
        indices_out[iter_out] = idx_l;
        //mshadow::Copy(out[iter_out], data_l[iter_l++], s);
        Tensor<xpu, 1, DType> lvalue = data_l[iter_l++];
        Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
        DCHECK_EQ(lvalue.shape_.Size(), rvalue.shape_.Size());
        MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
          Kernel<BinaryOpMapping<OP, Req>, xpu>::Launch(
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
        //mshadow::Copy(out[iter_out], data_r[iter_r++], s);
        //ASSIGN_DISPATCH(out[iter_out], req[0], F<MissingLeftUnaryOP>(data_r[iter_r++]));
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
      //mshadow::Copy(out[iter_out++], data_l[iter_l++], s);
      //ASSIGN_DISPATCH(out[iter_out], req[0], F<MissingRightUnaryOP>(data_l[iter_l++]));
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpMissingRValue<OP, Req>, xpu>::Launch(
          s, lvalue.shape_.Size(), out[iter_out].dptr_, lvalue.dptr_);
      });
    }
    while (iter_r < num_rows_r) {
      indices_out[iter_out] = indices_r[iter_r];
      Tensor<xpu, 1, DType> rvalue = data_r[iter_r++];
      //mshadow::Copy(out[iter_out++], data_r[iter_r++], s);
      //ASSIGN_DISPATCH(out[iter_out], req[0], F<MissingLeftUnaryOP>(data_r[iter_r++]));
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpMissingLValue<OP, Req>, xpu>::Launch(
          s, rvalue.shape_.Size(), out[iter_out].dptr_, rvalue.dptr_);
      });
    }
    nnvm::TShape new_shape = output.aux_shape(rowsparse::kIdx);
    new_shape[0] -= num_common_rows;  // Reduce the first-dimension size by the number of common rows
    output.SetAuxShape(rowsparse::kIdx, new_shape);
  };

 public:
  // Binary Compute between two row-sparse ndarray
  // This implementation only works on CPU
  template<typename xpu, typename OP>
  static void ComputeRspRsp(const nnvm::NodeAttrs &attrs,
                            const OpContext &ctx,
                            const std::vector<NDArray> &inputs,
                            const std::vector<OpReqType> &req,
                            const std::vector<NDArray> &outputs) {
#if 1
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      MSHADOW_TYPE_SWITCH(inputs[0].aux_type(rowsparse::kIdx), IType, {
        RspRspElemwiseBinaryOp<xpu, DType, IType, OP>(attrs, ctx, inputs, req, outputs);
      })
    });
#else
    auto &lhs = inputs[0];
    auto &rhs = inputs[1];
    auto &output = outputs[0];

    bool init_l = lhs.storage_initialized();
    bool init_r = rhs.storage_initialized();
    // both inputs are zeros
    if (!init_l && !init_r) return;
    // one of the input is zeros
    if (!init_l || !init_r) {
      NDArray out(output);
      CopyFromToRspImpl<xpu, xpu>(!init_l ? rhs : lhs, &out, ctx.run_ctx);
      return;
    }
    // Memory Estimation: This is (roughly) the number of result rows. We still
    // need to subtract the number of common rows
    unsigned int num_rows_l = lhs.aux_shape(rowsparse::kIdx).Size();
    unsigned int num_rows_r = rhs.aux_shape(rowsparse::kIdx).Size();
    output.CheckAndAlloc({mshadow::Shape1(num_rows_l + num_rows_r)});
    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
      MSHADOW_TYPE_SWITCH(lhs.aux_type(rowsparse::kIdx), IType, {
        // Indices
        auto indices_l = lhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        auto indices_r = rhs.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        auto indices_out = output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        // Data
        auto data_l = lhs.data().FlatTo2D<xpu, DType>(s);
        auto data_r = rhs.data().FlatTo2D<xpu, DType>(s);
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
#endif
  }

//struct NDArrayIterator {
//  NDArrayIterator(const NDArray& array)
//    : array_(array) {}
//
// private:
//  const NDArray& array_;
//};
//
//template<typename Callback>
//void IterateMatrix(NDArray& array, Callback callback) {
//  switch(array.storage_type()) {
//    case kRowSparseStorage:
//
//      break;
//    case kCSRStorage:
//      break;
//    case kDefaultStorage:
//      break;
//    default:
//      LOG(ERROR) << "Unknown storage type: " << array.storage_type();
//  }
//}

  template<typename xpu, typename OP, typename DType>
  static inline void Compute_(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] == kNullOp) return;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
                                / DataType<DType>::kLanes);
    DType *out_dptr = outputs[0].dptr<DType>();
    const DType * lhs_dptr = inputs[0].dptr<DType>();
    const DType * rhs_dptr = inputs[1].dptr<DType>();
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      Kernel<BinaryOpMapping<OP, Req>, xpu>::Launch(s, size, out_dptr, lhs_dptr, rhs_dptr);
    });
  }

  template<typename xpu, typename OP>
  static inline void Compute(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const std::vector<TBlob> &inputs,
                             const std::vector<OpReqType> &req,
                             const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Compute_<xpu, OP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename OP>
  static inline void ComputeEx(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    // If any input is dense, fallback to FCompute
    // TODO(haibin) implement dns + rsp in a separate kernel
    if (common::ContainsDefaultStorage(inputs)) {
      FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                           Compute<xpu, OP>, "Compute");
      return;
    }
    CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage) << "Sparse type not supported yet";
    CHECK_EQ(inputs[1].storage_type(), kRowSparseStorage) << "Sparse type not supported yet";
    ComputeRspRsp<xpu, OP>(attrs, ctx, inputs, req, outputs);
  }

  template<typename xpu, typename OP>
  static inline void ComputeWithHalf2(const nnvm::NodeAttrs &attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      Compute_<xpu, OP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename xpu, typename OP>
  static void Launch(const nnvm::NodeAttrs &attrs,
                     const OpContext &ctx,
                     const std::vector<TBlob> &inputs,
                     const std::vector<OpReqType> &req,
                     const std::vector<TBlob> &outputs) {
    using namespace mshadow;
    using namespace mxnet_op;
    Stream<xpu> *s = ctx.get_stream<xpu>();

    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Kernel<OP, xpu>::Launch(s, outputs[0].Size(),
                              outputs[0].dptr<DType>(),
                              inputs[0].dptr<DType>(), inputs[1].dptr<DType>());
    });
  }

  template<typename xpu, typename OP>
  static void LaunchEx(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<NDArray> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<NDArray> &outputs) {
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 1U);
    //OpBase::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, UnaryOp::Compute<xpu, OP>);
    CHECK(false); // won't handle inconsistencies between input and output shapes.
                  // Need something similar to out RspRspElemwiseBinaryOp to just handle
                  // the intersections
  }

  template<typename OP, int Req>
  struct BinaryOpBackwardUseNone
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad, const DType *ograd) {
      KERNEL_ASSIGN(igrad[i], Req, OP::Map(ograd[i]));
    }
  };

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BinaryBackwardUseNone_(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<TBlob> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    std::cout << "BinaryBackwardUseNone_" << std::endl << std::flush;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
                                / DataType<DType>::kLanes);
    DCHECK_EQ(size, outputs[0].Size());
    DCHECK_EQ(size, outputs[1].Size());
    DCHECK_EQ(size, inputs[0].Size());
    DType *lgrad_dptr = outputs[0].dptr<DType>();
    DType *rgrad_dptr = outputs[1].dptr<DType>();
    const DType *ograd_dptr = inputs[0].dptr<DType>();
    if (std::is_same<LOP, mshadow_op::identity>::value && req[0] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, lgrad_dptr);
    } else if (req[0] != kNullOp) {
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<BinaryOpBackwardUseNone<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr);
      });
    }
    if (std::is_same<ROP, mshadow_op::identity>::value && req[1] == kWriteInplace) {
      CHECK_EQ(ograd_dptr, rgrad_dptr);
    } else if (req[1] != kNullOp) {
      MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
        Kernel<BinaryOpBackwardUseNone<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr);
      });
    }
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

// Only implemented for _backward_add for now
  template<typename xpu, typename LOP, typename ROP>
  static void BinaryBackwardUseNoneRsp(const nnvm::NodeAttrs &attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(inputs[0].storage_type(), kRowSparseStorage);
    CHECK_EQ(outputs[0].storage_type(), kRowSparseStorage);
    CHECK_EQ(outputs[1].storage_type(), kRowSparseStorage);
    CHECK(typeid(LOP) == typeid(mshadow_op::identity));
    CHECK(typeid(ROP) == typeid(mshadow_op::identity));
    const TShape &shape = inputs[0].aux_shape(rowsparse::kIdx);
    outputs[0].CheckAndAlloc({shape});
    outputs[1].CheckAndAlloc({shape});
    MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
      MSHADOW_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), IType, {
        Tensor<xpu, 1, IType> lgrad_idx = outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        Tensor<xpu, 1, IType> rgrad_idx = outputs[1].aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        Tensor<xpu, 1, IType> ograd_idx = inputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        Tensor<xpu, 1, DType> lgrad = outputs[0].data().FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> rgrad = outputs[1].data().FlatTo1D<xpu, DType>(s);
        Tensor<xpu, 1, DType> ograd = inputs[0].data().FlatTo1D<xpu, DType>(s);
        ASSIGN_DISPATCH(lgrad, req[0], F<LOP>(ograd));
        ASSIGN_DISPATCH(rgrad, req[1], F<ROP>(ograd));
        ASSIGN_DISPATCH(lgrad_idx, req[0], F<LOP>(ograd_idx));
        ASSIGN_DISPATCH(rgrad_idx, req[1], F<ROP>(ograd_idx));
      });
    });
  }

  // Only implemented for _backward_add for now
  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneEx(const nnvm::NodeAttrs &attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    auto stype = inputs[0].storage_type();
    CHECK_EQ(stype, kRowSparseStorage) << "Not implemented yet";
    //LaunchEx<xpu, LOP>(attrs, ctx, { inputs[0] }, req, outputs);
    //LaunchEx<xpu, ROP>(attrs, ctx, { inputs[1] }, req, outputs);
    //BinaryBackwardUseNoneRsp<xpu, LOP, ROP>(attrs, ctx, inputs, req, outputs);
    // TODO(haibin) fallback for kDefaultStorage
  }

  template<typename xpu, typename LOP, typename ROP>
  static inline void BinaryBackwardUseNoneWithHalf2(const nnvm::NodeAttrs &attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
    MSHADOW_TYPE_SWITCH_WITH_HALF2(outputs[0].type_flag_, DType, {
      BinaryBackwardUseNone_<xpu, LOP, ROP, DType>(attrs, ctx, inputs, req, outputs);
    });
  }

  template<typename OP, int Req>
  struct BinaryOpBackwardUseIn
  {
    template<typename DType>
    MSHADOW_XINLINE static void Map(int i, DType *igrad,
                                    const DType *ograd, const DType *lhs, const DType *rhs) {
      KERNEL_ASSIGN(igrad[i], Req, ograd[i] * OP::Map(lhs[i], rhs[i]));
    }
  };

  template<typename xpu, typename LOP, typename ROP, typename DType>
  static void BinaryBackwardUseIn_(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
    using namespace mxnet_op;
    if (req[0] == kNullOp && req[1] == kNullOp) return;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    int size = static_cast<int>((outputs[0].Size() + DataType<DType>::kLanes - 1)
                                / DataType<DType>::kLanes);
    DType * lgrad_dptr = outputs[0].dptr<DType>();
    DType * rgrad_dptr = outputs[1].dptr<DType>();
    DType * ograd_dptr = inputs[0].dptr<DType>();
    DType * lhs_dptr = inputs[1].dptr<DType>();
    DType * rhs_dptr = inputs[2].dptr<DType>();
    MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
      Kernel<BinaryOpBackwardUseIn<LOP, Req>, xpu>::Launch(s, size, lgrad_dptr, ograd_dptr,
        lhs_dptr, rhs_dptr);});
    MXNET_ASSIGN_REQ_SWITCH(req[1], Req, {
      Kernel<BinaryOpBackwardUseIn<ROP, Req>, xpu>::Launch(s, size, rgrad_dptr, ograd_dptr,
        lhs_dptr, rhs_dptr);});
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

};  // class BinaryOp

template<typename GRAD_OP>
struct binary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    std::cout << "binary_bwd::Map( a = " << a << ", b = " << b << " )" << std::endl << std::flush;
    //const DType res = a*GRAD_OP::Map(b);
    const DType res = GRAD_OP::Map(a, b);
    std::cout << "binary_bwd::Map( a = " << a << ", b = " << b << " ) -> " << res << std::endl << std::flush;
    return res;
  }
};


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
