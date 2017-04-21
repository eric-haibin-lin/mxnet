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

namespace mxnet {

namespace op {
const OperatorProperty* OpPropGetOpProperty(const NodeAttrs& attrs);
}  // namespace op

namespace exec {

// forward executor
class ForwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Forward(op_ctx, in_data_, req, out_data_, aux_data_);
  }

  void Setup() override {
    in_data_.clear(); aux_data_.clear();
    for (size_t i = 0; i < in_array.size(); ++i) {
      if (!std::binary_search(aux_index_.begin(), aux_index_.end(), i)) {
        in_data_.push_back(in_array[i].data());
      } else {
        aux_data_.push_back(in_array[i].data());
      }
    }
    out_data_.resize(out_array.size());
    std::transform(out_array.begin(), out_array.end(), out_data_.begin(), [](const NDArray& nd) {
        return nd.data();
      });
  }
  Operator::ExecType exec_type() const override {
    return op_->exec_type();
  }
  explicit ForwardOpExecutor(Operator* op, std::vector<uint32_t> aux_index)
      : op_(op), aux_index_(aux_index) {
    std::sort(aux_index_.begin(), aux_index_.end());
  }

 private:
  friend Graph AttachOpExecs(Graph g);
  std::shared_ptr<Operator> op_;
  std::vector<uint32_t> aux_index_;
  std::vector<TBlob> in_data_, out_data_, aux_data_;
};

// backward executor
class BackwardOpExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    op_ctx.run_ctx = rctx;
    op_->Backward(op_ctx, out_grad_, in_data_, out_data_,
                  req, in_grad_, aux_data_);
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
  void Run(RunContext rctx) override {
    // std::cout << "FCompute::Run" << std::endl;
    op_ctx.run_ctx = rctx;
    // TODO(haibin) Get stream?
    // mshadow::Stream<cpu> *s = rctx.get_stream<cpu>();
    if (!initialized) {
      common::PrepDefaultBlobs<cpu>(in_array, out_array, &in_data_, &out_data_,
                                    &tmp_nds_, true, nullptr);
      initialized = true;
    }
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
    // std::cout << "FCompute::Done" << std::endl;
  }
  void Setup() override {
    in_array_ = in_array;
    out_array_ = out_array;
  }
  Operator::ExecType exec_type() const override {
    return Operator::kSync;
  }
  explicit FComputeExecutor(FCompute fcompute, const NodeAttrs& attrs)
      : fcompute_(fcompute), attrs_(attrs) {
  }

  static FCompute GetFCompute(const Op* op, Context ctx) {
    static auto& fcompute_cpu = nnvm::Op::GetAttr<FCompute>("FCompute<cpu>");
    static auto& fcompute_gpu = nnvm::Op::GetAttr<FCompute>("FCompute<gpu>");
    if (ctx.dev_mask() == cpu::kDevMask) {
      // if (fcompute_cpu.get(op, nullptr) != nullptr) std::cout << "FCompute for op " << op->name << std::endl;
      return fcompute_cpu.get(op, nullptr);
    } else if (ctx.dev_mask() == gpu::kDevMask) {
      return fcompute_gpu.get(op, nullptr);
    } else {
      LOG(FATAL) << "Unknown device mask";
      return nullptr;
    }
  }

 private:
  FCompute fcompute_;
  NodeAttrs attrs_;
  std::vector<TBlob> in_data_, out_data_;
  std::vector<NDArray> in_array_, out_array_, tmp_nds_;
  bool initialized = false;
};

// fcomputend executor
class FComputeExExecutor : public OpExecutor {
 public:
  void Run(RunContext rctx) override {
    // std::cout << "FComputeExExecutor::Run" << std::endl;
    op_ctx.run_ctx = rctx;
    fcompute_(attrs_, op_ctx, in_data_, req, out_data_);
    // std::cout << "FComputeExExecutor::Done" << std::endl;
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

  static FComputeEx GetFComputeEx(const Op* op, Context ctx,
                                  NDArrayStorageType dispatch_storage_type) {
    static auto& fcompute_cpu = nnvm::Op::GetAttr<FComputeEx>("FComputeEx<cpu, row_sparse>");
    static auto& fcompute_gpu = nnvm::Op::GetAttr<FComputeEx>("FComputeEx<gpu, row_sparse>");
    if (dispatch_storage_type != kRowSparseStorage) {
      return nullptr;
    }
    if (ctx.dev_mask() == cpu::kDevMask) {
#if EXECUTOR_DEBUG
      if (fcompute_cpu.get(op, nullptr) != nullptr)
        LOG(INFO) << "FComputeEx for op " << op->name;
#endif
      return fcompute_cpu.get(op, nullptr);
    } else if (ctx.dev_mask() == gpu::kDevMask) {
      return fcompute_gpu.get(op, nullptr);
    } else {
      LOG(FATAL) << "Unknown device mask";
      return nullptr;
    }
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
  const auto& dispatch_stypes = g.GetAttr<StorageTypeVector>("dispatch_storage_types");

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
    NDArrayStorageType dispatch_stype = static_cast<NDArrayStorageType>(dispatch_stypes[i]);
    FCompute fcompute = FComputeExecutor::GetFCompute(inode.source->op(), vctx[i]);
    FComputeEx fcompute_ndarray =
      FComputeExExecutor::GetFComputeEx(inode.source->op(), vctx[i], dispatch_stype);
    if (fcreate_layer_op.count(inode.source->op())) {
      std::vector<TShape> ishape;
      std::vector<int> itype;
      for (const auto& e : inode.inputs) {
        ishape.emplace_back(vshape[idx.entry_id(e)]);
        itype.emplace_back(vdtype[idx.entry_id(e)]);
      }
      ret[i] = std::make_shared<ForwardOpExecutor>(
          fcreate_layer_op[inode.source->op()](
              inode.source->attrs, vctx[i], ishape, itype), mutate_index);
    } else if (is_layer_backward.get(inode.source->op(), false)) {
      uint32_t fwd_id = inode.control_deps[0];
      CHECK_GE(inode.control_deps.size(), 1U);
      CHECK(vctx[fwd_id] == vctx[i]);
      CHECK(ret[fwd_id] != nullptr);
      ret[i] = std::make_shared<BackwardOpExecutor>(
          dynamic_cast<ForwardOpExecutor*>(ret[fwd_id].get())->op_,
          mxnet::op::OpPropGetOpProperty(inode.source->attrs),
          mutate_index);
    } else if (fcompute_ndarray != nullptr) {
      // Also check the storage type
      // std::cout << "S - fcompute_ndarray" << std::endl;
      ret[i] = std::make_shared<FComputeExExecutor>(fcompute_ndarray, inode.source->attrs);
    } else if (fcompute != nullptr) {
      // std::cout << "S - fcompute" << std::endl;
      ret[i] = std::make_shared<FComputeExecutor>(fcompute, inode.source->attrs);
    } else {
      // LOG(INFO) << "FCompute not registered " << inode.source->op()->name;
    }
  }
  g.attrs["op_execs"] = std::make_shared<nnvm::any>(ret);
  return g;
}

}  // namespace exec
}  // namespace mxnet
