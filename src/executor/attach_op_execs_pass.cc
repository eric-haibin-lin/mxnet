/*!
 * Copyright (c) 2016 by Contributors
 * \file attach_op_execs_pass.cc
 * \brief Operator executor to execute each operator.
 */
#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/graph_attr_types.h>
#include "./exec_pass.h"
#include "../common/utils.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif

#define EXEC_ATTACH_OP_DEBUG 0

namespace mxnet {

namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {

// forward executor
class ForwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    using namespace common;
    op_ctx.run_ctx = rctx;

    // If any input ndarray contains non-default storage,
    // we need to cast it to default storage and setup the tblobs again. For example,
    // if any of the input ndarray changes, the updated value won't be reflected in the temporary
    // ndarray with default storage.
    in_data_.clear(); out_data_.clear(); aux_data_.clear();
    temp_in_.clear(); temp_out_.clear(); temp_aux_.clear();
    if (is_gpu) {
#if MXNET_USE_CUDA
#if __CUDACC__
      GetDefaultBlobs<gpu>(in_array_, &in_data_, &temp_in_, op_ctx);
      GetDefaultBlobs<gpu>(aux_array_, &aux_data_, &temp_aux_, op_ctx);
      GetDefaultBlobs<gpu>(out_array, &out_data_, &temp_out_, op_ctx);
      op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
      CastNonDefaultStorage<gpu>(out_array, temp_out_, op_ctx);
#endif  // __CUDACC__
#elif NDEBUG == 0
      LOG(INFO) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
    } else {
      GetDefaultBlobs<cpu>(in_array_, &in_data_, &temp_in_, op_ctx);
      GetDefaultBlobs<cpu>(aux_array_, &aux_data_, &temp_aux_, op_ctx);
      GetDefaultBlobs<cpu>(out_array, &out_data_, &temp_out_, op_ctx);
      op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
      CastNonDefaultStorage<cpu>(out_array, temp_out_, op_ctx);
    }
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }

  void Setup() override {
    // We need to tell whether in NDArray is input or aux
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        in_array_.emplace_back(in_array[i]);
      } else {
        aux_array_.emplace_back(in_array[i]);
      }
    }
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit ForwardOpExecutor(std::shared_ptr<Operator> op,
      std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  friend Graph AttachOpExecs(Graph g);
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> in_data_, out_data_, aux_data_;
  std::vector<NDArray> in_array_, aux_array_, temp_in_, temp_aux_, temp_out_;
};

// backward executor
class BackwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    // TODO(haibin) support storage fallback for BackwardOpExecutor
    op_ctx.run_ctx = rctx;
    op_->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(out_grad_);
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
    mkl_tblobs_prv_to_cpu(in_grad_);
    mkl_tblobs_prv_to_cpu(aux_data_);
#endif
  }
  void Setup() override {
    size_t arg_top = 0, aux_top = 0;
    aux_data_.resize(aux_index_.size());
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        CHECK_GT(arg_data_ptr_.size(), arg_top);
        *arg_data_ptr_[arg_top++] = in_array[i].data();
      } else {
        aux_data_.at(aux_top++) = in_array[i].data();
      }
    }
    CHECK_EQ(out_array.size(), in_grad_.size());
    std::transform(out_array.begin(), out_array.end(),
                   in_grad_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit BackwardOpExecutor(std::shared_ptr<Operator> op,
                              const OperatorProperty* prop,
                              std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
    out_grad_.resize(prop->NumVisibleOutputs());
    in_data_.resize(prop->ListArguments().size());
    in_grad_.resize(in_data_.size());
    out_data_.resize(prop->NumOutputs());

    std::vector<TBlob*> out_grad_ptr(out_grad_.size());
    for (size_t i = 0; i < out_grad_.size(); ++i) {
      out_grad_ptr[i] = &out_grad_[i];
    }
    std::vector<TBlob*> in_data_ptr(in_data_.size());
    for (size_t i = 0; i < in_data_.size(); ++i) {
      in_data_ptr[i] = &in_data_[i];
    }
    std::vector<TBlob*> out_data_ptr(out_data_.size());
    for (size_t i = 0; i < out_data_.size(); ++i) {
      out_data_ptr[i] = &out_data_[i];
    }
    arg_data_ptr_ = prop->BackwardInputs(
        out_grad_ptr, in_data_ptr, out_data_ptr);
  }

 private:
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> out_grad_, in_grad_, in_data_, out_data_, aux_data_;
  std::vector<TBlob*> arg_data_ptr_;
};

// fcompute executor executor
class FComputeExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    using namespace common;
    op_ctx.run_ctx = rctx;
    // setup blobs
    // TODO(haibin) avoid repeating this if all inputs are already in default-storage.
    {
      in_data_.clear(); out_data_.clear();
      temp_in_.clear(); temp_out_.clear();
      if (is_gpu) {
#if MXNET_USE_CUDA
#if __CUDACC__
        GetDefaultBlobs<gpu>(in_array, &in_data_, &temp_in_, op_ctx);
        GetDefaultBlobs<gpu>(out_array, &out_data_, &temp_out_, op_ctx);
        fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
        CastNonDefaultStorage<gpu>(out_array, temp_out_, op_ctx);
#endif  // __CUDACC__
#else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
      } else {
        GetDefaultBlobs<cpu>(in_array, &in_data_, &temp_in_, op_ctx);
        GetDefaultBlobs<cpu>(out_array, &out_data_, &temp_out_, op_ctx);
        fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
        CastNonDefaultStorage<cpu>(out_array, temp_out_, op_ctx);
      }
    }
#if MKL_EXPERIMENTAL == 1
    mkl_tblobs_prv_to_cpu(in_data_);
    mkl_tblobs_prv_to_cpu(out_data_);
#endif
  }
  void Setup() override {}
  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }
  explicit FComputeExecutor(FCompute fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {
  }

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  std::vector<TBlob> in_data_, out_data_;
  std::vector<NDArray> temp_in_, temp_out_;
};

// fcomputend executor
class FComputeExExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx, bool is_gpu) override {
    op_ctx.run_ctx = rctx;
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
  }
  void Setup() override {
    in_data_ = in_array;
    out_data_ = out_array;
  }
  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }
  explicit FComputeExExecutor(FComputeEx fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {
  }

 private:
  FComputeEx fcompute_;
  NodeAttrs attrs_;
  std::vector<NDArray> in_data_, out_data_;
};

// pass to attach operator executors
Graph AttachOpExecs(Graph g) {
  using nnvm::DTypeVector;
  using nnvm::StorageTypeVector;
  using nnvm::ShapeVector;
  using nnvm::FMutateInputs;

  auto& fcreate_layer_op = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  auto& fmutate_inputs = nnvm::Op::GetAttr<FMutateInputs>("FMutateInputs");
  auto& is_layer_backward = nnvm::Op::GetAttr<bool>("TIsLayerOpBackward");

  const auto& vdtype = g.GetAttr<DTypeVector>("dtype");
  const auto& vshape = g.GetAttr<ShapeVector>("shape");
  const auto& vctx = g.GetAttr<ContextVector>("context");
  const auto& saved_opr = g.GetAttr<
    std::unordered_map<const nnvm::Node*, std::shared_ptr<Operator>>>("saved_opr");
  const auto& dispatch_stypes = g.GetAttr<StorageTypeVector>("dispatch_stypes");

  // get the graph
  const auto& idx = g.indexed_graph();
  std::vector<std::shared_ptr<OpExecutor> > ret(idx.num_nodes());

  // initialize the nodes
  for (size_t i = 0; i < idx.num_nodes(); ++i) {
    const auto& inode = idx[i];
    if (inode.source->is_variable()) continue;
    std::vector<uint32_t> mutate_index;
    if (fmutate_inputs.count(inode.source->op())) {
      mutate_index = fmutate_inputs[inode.source->op()](inode.source->attrs);
    }
    FCompute fcompute = common::GetFCompute(inode.source->op(), vctx[i]);
    FComputeEx fcompute_ex =
      common::GetFComputeEx(inode.source->op(), vctx[i], dispatch_stypes[i]);
#if EXEC_ATTACH_OP_DEBUG
    LOG(INFO) << "dispatch storage type = " << dispatch_stypes[i];
#endif
    if (fcreate_layer_op.count(inode.source->op())) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      std::shared_ptr<Operator> opr;
      if (saved_opr.count(inode.source)) {
        opr = saved_opr.at(inode.source);
      } else {
        opr.reset(fcreate_layer_op[inode.source->op()](
              inode.source->attrs, vctx[i], ishape, itype));
      }
      ret[i] = std::make_shared<ForwardOpExecutor>(opr, mutate_index);
#if EXEC_ATTACH_OP_DEBUG
      LOG(INFO) << "ForwardOp for op " << inode.source->op()->name;
#endif
    } else if (is_layer_backward.get(inode.source->op(), false)) {
      CHECK_GE(inode.control_deps.size(), 1);
      uint32_t fwd_id = inode.control_deps[0];
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      CHECK_EQ(dispatch_stypes[i], kDefaultStorage)
               << "BackwardOp doesn't handle non-default storage yet";
      ret[i] = std::make_shared<BackwardOpExecutor>(
          dynamic_cast<ForwardOpExecutor*>(ret[fwd_id].get())->op_,
          mxnet::op::OpPropGetOpProperty(inode.source->attrs),
          mutate_index);
#if EXEC_ATTACH_OP_DEBUG
      LOG(INFO) << "BackwardOp for op " << inode.source->op()->name;
#endif
    } else if (fcompute_ex != nullptr) {
#if EXEC_ATTACH_OP_DEBUG
      LOG(INFO) << "FComputeEx for op " << inode.source->op()->name;
#endif
      ret[i] = std::make_shared<FComputeExExecutor>(fcompute_ex, inode.source->attrs);
    } else if (fcompute != nullptr) {
#if EXEC_ATTACH_OP_DEBUG
      LOG(INFO) << "FCompute for op " << inode.source->op()->name;
#endif
      ret[i] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
