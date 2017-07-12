/*!
 *  Copyright (c) 2015 by Contributors
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
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

class OpBase {
 protected:
  /*! \brief Copy blob data */
  template<typename xpu>
  static void CopyBlob(mshadow::Stream<xpu> *s,
                       const TBlob& dest_blob, const OpReqType reqi, const TBlob& src_blob) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(src_blob.type_flag_, dest_blob.type_flag_);
    CHECK_EQ(src_blob.shape_, dest_blob.shape_);
    MSHADOW_TYPE_SWITCH(src_blob.type_flag_, DType, {
      mshadow::Copy(dest_blob.FlatTo1D<xpu, DType>(s), src_blob.FlatTo1D<xpu, DType>(s));
    });
  }

  /*! \brief Allocate geometry-related blob data for sparse tensors */
  static void AllocateGeometry(const NDArray *dest, const NDArray* clone_from = nullptr) {
    if (clone_from) {
      const TShape ishape = clone_from->storage_shape();
      TShape sshape = dest->storage_shape();
      CHECK(shape_assign(&sshape, ishape));
      dest->CheckAndAllocData(sshape);
      CHECK_EQ(dest->storage_type(), clone_from->storage_type());
      for(size_t i = 0, n = clone_from->aux_shape_count(); i < n; ++i) {
        TShape ashape = dest->aux_shape(i);
        CHECK(shape_assign(&ashape, clone_from->aux_shape(i)));
        dest->CheckAndAllocAuxData(i, ashape);
      }
      DCHECK_EQ(dest->aux_shape_count(), clone_from->aux_shape_count());
    } else {
      for (size_t i = 0, n = dest->aux_shape_count(); i < n; ++i) {
        dest->CheckAndAllocAuxData(i, dest->aux_shape(i));
      }
      dest->CheckAndAllocData(dest->storage_shape());
    }
  }

  /*! \brief Copy the geometry-related blobs (row sparse indexes, etc.) */
  template<typename xpu>
  static inline void CopyGeometryBlobs(mshadow::Stream<xpu> *s,
                                       const NDArray *dest,
                                       const OpReqType reqi,
                                       const NDArray &src) {
    CHECK_EQ(src.aux_shape_count(), dest->aux_shape_count());
    // My assumption is that the geometry blobs are not large enough to justify an omp loop here,
    // since the thread synchronization calls for each fork will take longer
    // than copying a few floats
    for(size_t i = 0, n = src.aux_shape_count(); i < n; ++i) {
      const TBlob src_blob = src.aux_data(i);
      const TBlob dest_blob = dest->aux_data(i);
      CopyBlob<xpu>(s, dest_blob, reqi, src_blob);
    }
  }

  /*! \brief Generic copy NDArray */
  template<typename xpu>
  static inline void CopyNDArray(mshadow::Stream<xpu> *s,
                                 const NDArray& dest,
                                 const OpReqType reqi,
                                 const NDArray& src) {
    DCHECK_NE(dest.storage_type(), kDefaultStorage);
    DCHECK_EQ(dest.storage_type(), src.storage_type());
    AllocateGeometry(&dest, &src);
    CopyGeometryBlobs(s, &dest, reqi, src);
    CopyBlob(s, dest.data(), reqi, src.data());
  }

  /*! \brief Get NDArray's data blob, possibly reshaped if necessary to reflect actual
   *         number of stored items */
  static inline TBlob GetReshapedBlob(const NDArray& arr) {
    TBlob blob = arr.data();
    switch(arr.storage_type()) {
      case kDefaultStorage:  // most common first
        break;
      case kRowSparseStorage:
      case kCSRStorage:
        blob.shape_ = arr.storage_shape();
        break;
      default:
        LOG(FATAL) << "Unrecognized storage type: " << arr.storage_type();
        break;
    }
    return blob;
  }

  /*! \brief Map NDArray vectors to TBlob vectors and pass to compute function */
  template<typename xpu, typename FComputer>
  static inline void MapToFCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs,
                                   FComputer computer) {
    std::vector<TBlob> in_blobs, out_blobs;
    in_blobs.reserve(inputs.size());
    out_blobs.reserve(outputs.size());
    for(size_t i = 0, n = inputs.size(); i < n; ++i) {
      in_blobs.emplace_back(std::move(GetReshapedBlob(inputs[i])));
    }
    for(size_t i = 0, n = outputs.size(); i < n; ++i) {
      out_blobs.emplace_back(std::move(GetReshapedBlob(outputs[i])));
    }
    computer(attrs, ctx, in_blobs, req, out_blobs);
  }

};

/*! \brief Unary operator class */
class UnaryOp : public OpBase {
  /*! \brief Infer the output storage geometry */
  template<int n_in, int n_out>
  static bool InitStorageGeometry(const nnvm::NodeAttrs& attrs,
                                  const std::vector<NDArray>& inputs,
                                  const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), static_cast<size_t>(n_in))
      << " in operator " << attrs.name;
    CHECK_EQ(outputs.size(), static_cast<size_t>(n_out))
      << " in operator " << attrs.name;
    CHECK(n_in > 0 && n_out > 0);
    if(!shape_is_none(inputs[0].storage_shape())) {
      NDArray *output = nullptr;
      for(size_t i = 0, n = inputs.size(); i < n; ++i) {
        const NDArray& input = inputs[i];
        const TShape& ishape = input.storage_shape();
        if(i < n_out) {
          output = const_cast<NDArray *>(&outputs[i]);
        }
        TShape sshape = output->storage_shape();
        CHECK(shape_assign(&sshape, ishape));
        output->set_storage_shape(sshape);
        CHECK_EQ(output->storage_type(), input.storage_type());
        CHECK_EQ(output->aux_shape_count(), input.aux_shape_count());
        for(size_t j = 0, jn = input.aux_shape_count(); j < jn; ++j) {
          TShape ashape = output->aux_shape(j);
          CHECK(shape_assign(&ashape, input.aux_shape(j)));
          output->set_aux_shape(j, ashape);
        }
      }
      return true;
    } else {
      CHECK(false); // implement when necessary
    }
    return false;
  }

  /*! \brief Map NDArray vectors to TBlob vectors and pass to compute function */
  template<typename xpu, typename FComputer>
  static inline void MapToFCompute(const nnvm::NodeAttrs &attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &outputs,
                                   FComputer computer) {
    // Copy over geometry
    InitStorageGeometry<1, 1>(attrs, inputs, outputs);
    CHECK_EQ(inputs.size(), outputs.size()); // need to figure out what to do for binary type
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage);
    CHECK_EQ(inputs[0].storage_type(), outputs[0].storage_type());
    AllocateGeometry(&outputs[0], &inputs[0]);
    CopyGeometryBlobs<xpu>(ctx.get_stream<xpu>(), &outputs[0], req[0], inputs[0]);
    outputs[0].CheckAndAllocData(inputs[0].storage_shape());
    OpBase::MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, computer);
  }

 public:
  template<typename xpu, typename OP>
  static void Compute(const nnvm::NodeAttrs& attrs,
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

  template<typename xpu, typename OP>
  static void ComputeEx(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse output storage type";
    MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Compute<xpu, OP>);
  }

  /*! \brief Fall back to dense and compute */
  template<typename xpu, typename OP>
  static void ComputeAsDense(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a dense output storage type";
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                         Compute<xpu, OP>, "ComputeAsDense");
  }

  template<typename xpu, typename op>
  static void Launch(const nnvm::NodeAttrs& attrs,
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

  template<typename xpu, typename OP>
  static void LaunchEx(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_NE(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a sparse output storage type";
    MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, Launch<xpu, OP>);
  }

  template<typename xpu, typename OP>
  static void LaunchAsDense(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    CHECK_EQ(outputs[0].storage_type(), kDefaultStorage)
      << "Operation requires a dense output storage type";
    FCompExFallback<xpu>(attrs, ctx, inputs, req, outputs,
                         Launch<xpu, OP>, "LaunchAsDense");
  }

  template<typename xpu>
  static void IdentityCompute(const nnvm::NodeAttrs& attrs,
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
  static void IdentityComputeEx(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 1U);
    CHECK_EQ(outputs.size(), 1U);
    MapToFCompute<xpu>(attrs, ctx, inputs, req, outputs, IdentityCompute<xpu>);
  }

  template<typename xpu>
  static void IdentityComputeFirstItemsEx(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
#if 0
  size_t rhs_idx = 1;
  NDArrayStorageType stype = inputs[rhs_idx].storage_type();
  if (stype == kRowSparseStorage) {
    IdentityComputeRsp<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    LOG(FATAL) << "Not implemented yet";
  }
#else
    OpBase::CopyNDArray(ctx.get_stream<xpu>(), outputs[0], req[0], inputs[0]);
#endif
  }

//  template<typename xpu>
//  static void IdentityComputeRsp(const nnvm::NodeAttrs& attrs,
//                                 const OpContext& ctx,
//                                 const std::vector<NDArray>& inputs,
//                                 const std::vector<OpReqType>& req,
//                                 const std::vector<NDArray>& outputs) {
//    using namespace mshadow;
//    using namespace mshadow::expr;
//    Stream<xpu> *s = ctx.get_stream<xpu>();
//    auto &input = inputs[0];
//    auto &output = outputs[0];
//    CHECK_NE(req[0], kNullOp) << "kNullOp in IdentityComputeEx not supported yet";
//    CHECK_NE(req[0], kWriteInplace) << "kWriteInplace in IdentityComputeEx not supported yet";
//    if (!input.storage_initialized()) return;
//    TShape shape = input.aux_shape(rowsparse::kIdx);
//    output.CheckAndAlloc({shape});
//    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
//      MSHADOW_TYPE_SWITCH(output.aux_type(rowsparse::kIdx), AuxType, {
//        auto out_d = output.data().FlatTo1D<xpu, DType>(s);
//        auto out_aux = output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
//        auto in_aux = input.aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
//        ASSIGN_DISPATCH(out_d, req[0],
//                        F<mshadow_op::identity>(input.data().FlatTo1D<xpu, DType>(s)));
//        ASSIGN_DISPATCH(out_aux, req[0], F<mshadow_op::identity>(in_aux));
//      });
//    });
//  }
};

template<typename GRAD_OP>
struct unary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a * GRAD_OP::Map(b);
  }
};

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
  MSHADOW_XINLINE static DType Map(DType out_grad, DType in) {
    return out_grad * DType(in * (DType(1.0f) - in));
  }
};
/*! \brief Rectified Linear Operation */
struct relu {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *in) {
    DType x = in[i];
    out[i] = x > DType(0.0f) ? x : DType(0.0f);
  }
};
struct relu_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType out_grad, DType in) {
    return out_grad * DType(in > DType(0.0f) ? DType(1.0f) : DType(0.0f));
  }
};
}  // namespace kernel_launch_op

struct CastStorageParam : public dmlc::Parameter<CastStorageParam> {
  int storage_type;
  DMLC_DECLARE_PARAMETER(CastStorageParam) {
    DMLC_DECLARE_FIELD(storage_type)
      .add_enum("default", kDefaultStorage)
      .add_enum("row_sparse", kRowSparseStorage)
      .add_enum("csr", kCSRStorage)
      .describe("Output storage type.");
  }
};

inline bool CastStorageInferStorageType(const nnvm::NodeAttrs& attrs,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE(in_attrs->at(0), kUndefinedStorage)
    << "src ndarray's storage type must be specified";
  const CastStorageParam& param = nnvm::get<CastStorageParam>(attrs.parsed);
  CHECK_NE(param.storage_type, kUndefinedStorage)
    << "dst ndarray's storage type must be specified";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.storage_type);
  return true;
}

template<typename xpu>
void CastStorageComputeEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  CastStorageComputeImpl<xpu>(s, inputs[0], outputs[0]);
}

#define MXNET_OPERATOR_REGISTER_UNARY(name)                         \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1>) \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input array.")

#define MXNET_OPERATOR_REGISTER_UNARY_DR(name)                      \
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

/*! \brief Unary launch */
#define MXNET_OPERATOR_REGISTER_UNARY_LAUNCH(name, __xpu$, __kernel$)                   \
  MXNET_OPERATOR_REGISTER_UNARY(name)                                                   \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Launch<__xpu$, __kernel$>)      \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::LaunchEx<__xpu$, __kernel$>)

/*! \brief Unary launch, dense result */
#define MXNET_OPERATOR_REGISTER_UNARY_LAUNCH_DR(name, __xpu$, __kernel$)                \
  MXNET_OPERATOR_REGISTER_UNARY_DR(name)                                                \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Launch<__xpu$, __kernel$>)      \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::LaunchAsDense<__xpu$, __kernel$>)

/*! \brief Unary compute */
#define MXNET_OPERATOR_REGISTER_UNARY_COMPUTE(name, __xpu$, __kernel$)                  \
  MXNET_OPERATOR_REGISTER_UNARY(name)                                                   \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::ComputeEx<__xpu$, __kernel$>)

/*! \brief Unary compute, dense result */
#define MXNET_OPERATOR_REGISTER_UNARY_COMPUTE_DR(name, __xpu$, __kernel$)               \
  MXNET_OPERATOR_REGISTER_UNARY_DR(name)                                                \
  .set_attr<FCompute>("FCompute<" #__xpu$ ">", UnaryOp::Compute<__xpu$, __kernel$>)     \
  .set_attr<FComputeEx>("FComputeEx<" #__xpu$ ">", UnaryOp::ComputeAsDense<__xpu$, __kernel$>)


}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
