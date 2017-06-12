/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_ndarray.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <string>
#include "./c_api_common.h"
#include "../common/utils.h"
#include "../ndarray/autograd.h"

#define IMPERATIVE_EXEC_DEBUG 0

using namespace mxnet;
using mxnet::autograd::AutogradRuntime;

void SetOpAttrs(const nnvm::Op *op,
                nnvm::NodeAttrs *p_attrs,
                const int& num_inputs,
                const int& num_params,
                const char **param_keys,
                const char **param_vals) {
  static auto& num_args = nnvm::Op::GetAttr<std::string>("key_var_num_args");
  nnvm::NodeAttrs& attrs = *p_attrs;
  attrs.op = op;
  for (int i = 0; i < num_params; ++i) {
    attrs.dict.emplace(param_keys[i], param_vals[i]);
  }

  if (num_args.count(op)) {
    attrs.dict.emplace(num_args[op], std::to_string(num_inputs));
  }
  if (op->attr_parser != nullptr) {
    op->attr_parser(&attrs);
  }
}

void SetNumOutputs(const nnvm::Op *op,
                   const nnvm::NodeAttrs& attrs,
                   const int& num_inputs,
                   int* infered_num_outputs,
                   int* num_visible_outputs) {
  static auto& visible_out = nnvm::Op::GetAttr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs");
  int infered_num_inputs;
  if (op->get_num_inputs != nullptr) {
    infered_num_inputs = op->get_num_inputs(attrs);
  } else {
    infered_num_inputs = op->num_inputs;
  }
  CHECK_EQ(num_inputs, infered_num_inputs)
    << "Expecting " << infered_num_inputs << " inputs, got "
    << num_inputs << " in operator " << op->name;
  if (op->get_num_outputs != nullptr) {
    *infered_num_outputs = op->get_num_outputs(attrs);
  } else {
    *infered_num_outputs = op->num_outputs;
  }
  *num_visible_outputs = *infered_num_outputs;
  if (visible_out.count(op)) {
    *num_visible_outputs = visible_out[op](attrs);
    CHECK_LE(*num_visible_outputs, *infered_num_outputs);
  }
}

void SetNDInputsOutputs(const nnvm::Op* op,
                        std::vector<NDArray>* p_ndinputs,
                        std::vector<NDArray>* p_ndoutputs,
                        const int& num_inputs,
                        const NDArrayHandle *inputs,
                        int *num_outputs,
                        const int& infered_num_outputs,
                        const int& num_visible_outputs,
                        NDArray** outarray) {
  std::vector<NDArray>& ndinputs  = *p_ndinputs;
  std::vector<NDArray>& ndoutputs = *p_ndoutputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.emplace_back(*reinterpret_cast<NDArray*>(inputs[i]));
  }
  if (outarray == nullptr) {
    *num_outputs = num_visible_outputs;
    ndoutputs.resize(infered_num_outputs);
  } else {
    CHECK(!AutogradRuntime::Get()->IsTraining())
      << "Inplace operations (+=, -=, op(..., out=x) etc.) and assignment are "
      << "not supported when you are inside a train_section using autograd.";
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Expecting " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, got "
      << *num_outputs << " in operator " << op->name;
    ndoutputs.reserve(infered_num_outputs);
    for (int i = 0; i < num_visible_outputs; ++i) {
      ndoutputs.emplace_back(std::move(*outarray[i]));
    }
    ndoutputs.resize(infered_num_outputs);
  }
}

void SetContext(Context* p_ctx,
                const nnvm::NodeAttrs& attrs,
                const int& num_inputs,
                const std::vector<NDArray>& ndinputs,
                const int& infered_num_outputs,
                const std::vector<NDArray>& ndoutputs) {
  Context& ctx = *p_ctx;
  if (num_inputs) {
    ctx = ndinputs[0].ctx();
  } else if (infered_num_outputs && !ndoutputs[0].is_none()) {
    ctx = ndoutputs[0].ctx();
  } else if (attrs.dict.find("ctx") != attrs.dict.end()) {
    ctx = Context::FromString(attrs.dict.at("ctx"));
  } else {
    ctx = Context::CPU();
  }
  // Pinned context doesn't propagate
  if (ctx.dev_type == Context::kCPUPinned) {
    ctx = Context::CPU();
  }
}

/*! \brief Retuen reference to indexed item in vector if the vector is large enough.
 * Otherwise, return an object to a blank (default) static object
 * This allows us to not waste time and resources allocating vectors whose items won't be used
 * @tparam Object
 * @param vec Potential vector of objects
 * @param i Potential index into the vector
 * @return reference to the appropriate object
 */
template<typename Object>
inline const Object& IndexedOrBlank(const std::vector<Object>& vec, const int i) {
  static Object blank;
  return i < vec.size() ? vec[i] : blank;
}

//struct StorageGeometry {
//  TShape              data_shape_;
//  struct Aux {
//    TShape  aux_shape_;
//    int     type_;
//  };
//  std::vector<Aux> aux_;
//};

// Set the shape, dtype and storage type
void SetShapeType(const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<NDArray>& ndinputs,
                  const int& infered_num_outputs,
                  std::vector<NDArray>* p_ndoutputs,
                  int* dispatch_stype) {
  std::vector<NDArray>& ndoutputs = *p_ndoutputs;
  static auto& infershape = nnvm::Op::GetAttr<nnvm::FInferShape>("FInferShape");
  static auto& infertype = nnvm::Op::GetAttr<nnvm::FInferType>("FInferType");
  static auto& inferstorage = nnvm::Op::GetAttr<nnvm::FInferStorageType>("FInferStorageType");
  //static auto& inferstoragegeometry = nnvm::Op::GetAttr<nnvm::FInferStorageGeometry>("FInferStorageGeometry");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  // infer shape
  std::vector<TShape>& in_shapes  = ret->arg_shapes;
  std::vector<TShape>& out_shapes = ret->out_shapes;
  in_shapes.clear();
  out_shapes.clear();

  for (auto& i : ndinputs) {
    in_shapes.emplace_back(i.shape());
  }

  for (auto& i : ndoutputs) {
    out_shapes.emplace_back(i.shape());
  }

  CHECK(infershape.count(op))
    << "Operator " << op->name << " is missing FInferShape attribute";
  CHECK(infershape[op](attrs, &in_shapes, &out_shapes));
  CHECK_EQ(out_shapes.size(), static_cast<size_t>(infered_num_outputs));

  // infer type
  std::vector<int>& in_types = ret->arg_types;
  std::vector<int>& out_types = ret->out_types;
  in_types.clear();
  in_types.reserve(ndinputs.size());
  out_types.clear();
  out_types.reserve(ndoutputs.size());

  for (auto& i : ndinputs) {
    in_types.push_back(i.dtype());
  }
  for (auto& i : ndoutputs) {
    out_types.push_back(i.dtype());
  }
  CHECK(infertype.count(op))
    << "Operator " << op->name << " is missing FInferType attribute";
  CHECK(infertype[op](attrs, &in_types, &out_types));
  CHECK_EQ(out_types.size(), static_cast<size_t>(infered_num_outputs));

  // infer storage type
  auto& in_storage_types = ret->arg_storage_types;
  auto& out_storage_types = ret->out_storage_types;
  in_storage_types.clear();
  in_storage_types.reserve(ndinputs.size());
  out_storage_types.clear();
  out_storage_types.reserve(ndoutputs.size());

  for (auto& i : ndinputs) {
    in_storage_types.push_back(i.storage_type());
  }
  for (auto& i : ndoutputs) {
    out_storage_types.push_back(i.storage_type());
  }
  if (inferstorage.count(op)) {
    CHECK(inferstorage[op](attrs, &in_storage_types, &out_storage_types));
    CHECK_EQ(out_storage_types.size(), static_cast<size_t>(infered_num_outputs));
  } else {
#if IMPERATIVE_EXEC_DEBUG
    LOG(INFO) << "FInferStorageType not present.";
#endif
  }

//  std::vector<TShape> in_ss, out_ss;
//  std::vector<std::vector<TShape>> in_aux_storage_shapes, out_aux_storage_shapes;
//  std::vector<std::vector<int>> in_aux_storage_types, out_aux_storage_types;

//  // See if the output storage and/or aux shapes are already known (ie unary, some binary, etc.)
//  if (inferstoragegeometry.count(op)) {
//
//    const size_t inputSize = ndinputs.size();
//    in_ss.resize(inputSize);
//    in_aux_storage_shapes.resize(inputSize);
//    in_aux_storage_types.resize(inputSize);
//
//    const size_t outputSize = ndoutputs.size();
//    out_ss.resize(outputSize);
//    out_aux_storage_shapes.resize(outputSize);
//    out_aux_storage_types.resize(outputSize);
//
//    auto ndarray_reader = [](const std::vector<NDArray>& puts,
//                             std::vector<TShape>& ss,
//                             std::vector<std::vector<TShape>>& aux_ss) {
//      for (size_t o = 0, no = puts.size(); o < no; ++o) {
//        const NDArray& i = puts[o];
//        if (i.storage_type() != kDefaultStorage) {
//          if(!i.is_none()) {
//            ss[o] = i.storage_shape();
//            std::vector<TShape> &shapeVect = aux_ss[o];
//            shapeVect.resize(i.aux_shape_count());
//            for (size_t x = 0, n = i.aux_shape_count(); x < n; ++x) {
//              shapeVect[x] = i.aux_shape(x);
//            }
//          }
//        }
//      }
//    };
//
//    ndarray_reader(ndinputs,  in_ss,  in_aux_storage_shapes);
//    ndarray_reader(ndoutputs, out_ss, out_aux_storage_shapes);
//
//    CHECK(inferstoragegeometry[op](attrs,
//                                 &in_ss, &in_aux_storage_shapes, &in_aux_storage_types,
//                                 &out_ss, &out_aux_storage_shapes, &out_aux_storage_types));
//    CHECK_LE(out_ss.size(), ndoutputs.size());
//    CHECK_LE(out_aux_storage_shapes.size(), ndoutputs.size());
//    CHECK_EQ(out_aux_storage_types.size(), out_aux_storage_shapes.size());
//  }

  const bool contains_non_default = common::ContainsNonDefaultStorage(in_storage_types)
                              || common::ContainsNonDefaultStorage(out_storage_types);
  const int kNonDefaultStorage = kUndefinedStorage - 1;
  *dispatch_stype = contains_non_default ? kNonDefaultStorage : kDefaultStorage;

  for (int i = 0; i < infered_num_outputs; ++i) {
    const NDArrayStorageType storage_type = static_cast<NDArrayStorageType>(out_storage_types[i]);
    if (ndoutputs[i].is_none()) {
      // If failed to infer the storage type, assume the output storage is dense
      if (storage_type == kDefaultStorage || out_storage_types[i] == kUndefinedStorage) {
        ndoutputs[i] = NDArray(out_shapes[i], ctx, true, out_types[i]);
      } else {
        ndoutputs[i] = NDArray(storage_type, out_shapes[i], ctx, true, out_types[i] /*,
                               std::vector<int>(),
                               IndexedOrBlank(out_aux_storage_shapes, i),
                               IndexedOrBlank(out_ss, i)*/);
      }
    } else {
      CHECK_EQ(ndoutputs[i].shape(), out_shapes[i])
        << i << "th output has invalid shape. "
        << "Expecting " << out_shapes[i] << " got "
        << ndoutputs[i].shape() << " in operator " << op->name;
      CHECK_EQ(ndoutputs[i].dtype(), out_types[i])
        << i << "th output has invalid shape. "
        << "Expecting " << out_types[i] << " got "
        << ndoutputs[i].dtype()  << " in operator " << op->name;
    }
  }
}

void SetDependency(std::vector<engine::VarHandle> *p_read_vars,
                   std::vector<engine::VarHandle> *p_write_vars,
                   std::vector<Resource> *p_requested,
                   std::vector<uint32_t> *p_auxidx,
                   const nnvm::Op* op,
                   const nnvm::NodeAttrs& attrs,
                   const Context& ctx,
                   const std::vector<NDArray>& ndinputs,
                   const std::vector<NDArray>& ndoutputs) {
  static auto& mutate = nnvm::Op::GetAttr<nnvm::FMutateInputs>("FMutateInputs");
  static auto& tmp_resource = nnvm::Op::GetAttr<FResourceRequest>("FResourceRequest");

  std::vector<engine::VarHandle>& read_vars  = *p_read_vars;
  std::vector<engine::VarHandle>& write_vars = *p_write_vars;
  std::vector<Resource>& requested = *p_requested;
  std::vector<uint32_t>& auxidx = *p_auxidx;

  if (tmp_resource.count(op)) {
    int ntmp = 0;
    for (const auto& req : tmp_resource[op](attrs)) {
      switch (req.type) {
       case ResourceRequest::kTempSpace:
        ++ntmp;
       case ResourceRequest::kRandom:
        requested.push_back(ResourceManager::Get()->Request(ctx, req));
        write_vars.push_back(requested.back().var);
        break;
       default:
        LOG(FATAL) << "resource type not yet supported";
      }
    }
    CHECK_LE(ntmp, 1) << "Only support 1 temp space request";
  }
  for (auto& i : ndinputs) read_vars.emplace_back(i.var());
  for (auto& i : ndoutputs) write_vars.emplace_back(i.var());
  if (mutate.count(op)) {
    auxidx = mutate[op](attrs);
    std::sort(auxidx.begin(), auxidx.end());
    for (auto& i : auxidx) {
      auto var = ndinputs[i].var();
      write_vars.push_back(var);
    }
  }
  Engine::Get()->DeduplicateVarHandle(&read_vars, &write_vars);
}


void PushFCompute(const FCompute& fn,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<NDArray>& ndinputs,
                  const std::vector<NDArray>& ndoutputs) {
  using namespace common;
  bool is_train = AutogradRuntime::Get()->IsTraining();
  Engine::Get()->PushAsync(
    [ctx, attrs, fn, ndinputs, ndoutputs, requested, is_train](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, output_blobs;
      std::vector<NDArray> temp_in;
      std::vector<NDArray> temp_out;
      OpContext opctx{is_train, rctx,
                      engine::CallbackOnComplete(),
                      requested};
      if (ctx.dev_mask() == gpu::kDevMask) {
#if MXNET_USE_CUDA
        GetDefaultBlobs<gpu>(ndinputs, &input_blobs, &temp_in, opctx);
        GetDefaultBlobs<gpu>(ndoutputs, &output_blobs, &temp_out, opctx);
        std::vector<OpReqType> req(output_blobs.size(), kWriteTo);
        fn(attrs, opctx, input_blobs, req, output_blobs);
        // cast to original storage type, if necessary
        CastNonDefaultStorage<gpu>(ndoutputs, temp_out, opctx);
        rctx.get_stream<gpu>()->Wait();
#else
        LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
#endif
      } else {
        GetDefaultBlobs<cpu>(ndinputs, &input_blobs, &temp_in, opctx);
        GetDefaultBlobs<cpu>(ndoutputs, &output_blobs, &temp_out, opctx);
        std::vector<OpReqType> req(output_blobs.size(), kWriteTo);
        fn(attrs, opctx, input_blobs, req, output_blobs);
        CastNonDefaultStorage<cpu>(ndoutputs, temp_out, opctx);
      }
      on_complete();
    }, ctx, read_vars, write_vars, FnProperty::kNormal,
    0, PROFILER_MESSAGE(op->name.c_str()));
}

void PushFComputeEx(const FComputeEx& fn,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<NDArray>& ndinputs,
                  const std::vector<NDArray>& ndoutputs) {
  Engine::Get()->PushAsync(
    [ctx, attrs, fn, ndinputs, ndoutputs, requested](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, output_blobs;
      OpContext opctx{false, rctx,
                      engine::CallbackOnComplete(),
                      requested};
      std::vector<OpReqType> req(ndoutputs.size(), kWriteTo);
      fn(attrs, opctx, ndinputs, req, ndoutputs);
      if (ctx.dev_mask() == gpu::kDevMask) {
        rctx.get_stream<gpu>()->Wait();
      }
      on_complete();
    }, ctx, read_vars, write_vars, FnProperty::kNormal,
    0, PROFILER_MESSAGE(op->name.c_str()));
}

void PushOperator(std::shared_ptr<Operator> opr,
                  const nnvm::Op* op,
                  const nnvm::NodeAttrs& attrs,
                  const Context& ctx,
                  const std::vector<engine::VarHandle>& read_vars,
                  const std::vector<engine::VarHandle>& write_vars,
                  const std::vector<Resource>& requested,
                  const std::vector<uint32_t>& auxidx,
                  const std::vector<NDArray>& ndinputs,
                  const std::vector<NDArray>& ndoutputs) {
  struct Capture {
    engine::CallbackOnComplete on_complete;
    std::shared_ptr<Operator> opr;
  };

  bool is_train = AutogradRuntime::Get()->IsTraining();
  Engine::Get()->PushAsync(
    [ctx, opr, auxidx, ndinputs, ndoutputs, requested, is_train](
        RunContext rctx,
        engine::CallbackOnComplete on_complete) {
      std::vector<TBlob> input_blobs, aux_blobs, output_blobs;
      auto atop = auxidx.begin();
      for (size_t i = 0; i < ndinputs.size(); ++i) {
        if (atop != auxidx.end() && i == *atop) {
          aux_blobs.push_back(ndinputs[i].data());
          ++atop;
        } else {
          input_blobs.push_back(ndinputs[i].data());
        }
      }
      for (auto& i : ndoutputs) {
        output_blobs.push_back(i.data());
      }
      Capture* capture = new Capture({on_complete, opr});
      OpContext opctx{is_train, rctx,
                      Engine::Get()->CreateCallback(
                        [](Engine* engine, void *cpt_handle) {
                            Capture* cpt = static_cast<Capture*>(cpt_handle);
                            cpt->on_complete();
                            delete cpt;
                          }, static_cast<void*>(capture)),
                      requested};
      std::vector<OpReqType> req(output_blobs.size(), kWriteTo);
      opr->Forward(opctx, input_blobs, req, output_blobs, aux_blobs);
      if (opr->exec_type() != Operator::kAsync) {
        if (ctx.dev_mask() == gpu::kDevMask) {
          rctx.get_stream<gpu>()->Wait();
        }
        delete capture;
        on_complete();
      }
    }, ctx, read_vars, write_vars, FnProperty::kNormal,
    0, PROFILER_MESSAGE(op->name.c_str()));
}

void ImperativeInvokeImpl(const nnvm::NodeAttrs& attrs,
                          int num_inputs,
                          NDArrayHandle *inputs,
                          int *num_outputs,
                          NDArrayHandle **outputs) {
  static auto& ndfunc = nnvm::Op::GetAttr<FNDArrayFunction>("FNDArrayFunction");
  static auto& createop = nnvm::Op::GetAttr<FCreateLayerOp>("FCreateLayerOp");
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  NDArray** outarray = *reinterpret_cast<NDArray***>(outputs);
  const nnvm::Op *op = attrs.op;

  int infered_num_outputs;
  int num_visible_outputs;
  SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray> ndinputs, ndoutputs;
  SetNDInputsOutputs(op, &ndinputs, &ndoutputs, num_inputs, inputs,
                     num_outputs, infered_num_outputs, num_visible_outputs, outarray);

  if (ndfunc.count(op)) {
    ndfunc[op](attrs, ndinputs, &ndoutputs);
#if IMPERATIVE_EXEC_DEBUG
    LOG(INFO) << "NDArray function executed.";
#endif
  } else {
    // TODO(piiswrong): infer ctx
    Context ctx;
    int storage_type;
    SetContext(&ctx, attrs, num_inputs, ndinputs, infered_num_outputs, ndoutputs);
    SetShapeType(op, attrs, ctx, ndinputs, infered_num_outputs, &ndoutputs, &storage_type);

    std::vector<engine::VarHandle> read_vars, write_vars;
    std::vector<Resource> requested;
    std::vector<uint32_t> auxidx;
    SetDependency(&read_vars, &write_vars, &requested, &auxidx,
        op, attrs, ctx, ndinputs, ndoutputs);

    FCompute fn = common::GetFCompute(op, ctx);
    FComputeEx fcomp_ex = common::GetFComputeEx(op, ctx, storage_type);
    if (fcomp_ex) {
      PushFComputeEx(fcomp_ex, op, attrs, ctx, read_vars, write_vars, requested,
                     ndinputs, ndoutputs);
#if IMPERATIVE_EXEC_DEBUG
      LOG(INFO) << "FComputeEx executed.";
#endif
    } else if (fn) {
      if (AutogradRuntime::Get()->IsTraining()) {
        AutogradRuntime::Get()->RecordImperativeFCompute(op,
            attrs, &ndinputs, &ndoutputs);
      }
      PushFCompute(fn, op, attrs, ctx, read_vars, write_vars,
          requested, ndinputs, ndoutputs);
#if IMPERATIVE_EXEC_DEBUG
      LOG(INFO) << "FCompute executed.";
#endif
    } else if (createop.count(op)) {
      std::shared_ptr<Operator> opr(
          createop[op](attrs, ctx, ret->arg_shapes, ret->arg_types));
      if (AutogradRuntime::Get()->IsTraining()) {
        AutogradRuntime::Get()->RecordImperativeOperator(opr, op,
            attrs, &ndinputs, &ndoutputs);
      }
      PushOperator(opr, op, attrs, ctx, read_vars, write_vars,
          requested, auxidx, ndinputs, ndoutputs);
#if IMPERATIVE_EXEC_DEBUG
      LOG(INFO) << "CreateOp executed.";
#endif
    } else {
      LOG(FATAL)
        << "Operator " << op->name
        << " cannot be run; requires at least one of"
        << " FCompute<xpu>, FComputeEx<xpu> NDArrayFunction, FCreateOperator be registered";
    }
  }

  if (outarray == nullptr) {
    ret->ret_handles.clear();
    for (int i = 0; i < num_visible_outputs; ++i) {
      ret->ret_handles.push_back(
        reinterpret_cast<NDArrayHandle>(new NDArray(std::move(ndoutputs[i]))));
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  } else {
    for (int i = 0; i < *num_outputs; ++i) {
      *outarray[i] = std::move(ndoutputs[i]);
    }
  }
}

int MXImperativeInvoke(AtomicSymbolCreator creator,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       int num_params,
                       const char **param_keys,
                       const char **param_vals) {
  const nnvm::Op* op = static_cast<nnvm::Op*>(creator);

  API_BEGIN();
  nnvm::NodeAttrs attrs;
  SetOpAttrs(op, &attrs, num_inputs, num_params, param_keys, param_vals);
  ImperativeInvokeImpl(attrs, num_inputs, inputs, num_outputs, outputs);
  API_END();
}

int MXCachedCreateOp(AtomicSymbolCreator creator,
                     int num_inputs,
                     int num_params,
                     const char **param_keys,
                     const char **param_vals,
                     CachedOpHandle *out) {
  const nnvm::Op* op = static_cast<nnvm::Op*>(creator);

  API_BEGIN();
  nnvm::NodeAttrs *attrs = new nnvm::NodeAttrs;
  SetOpAttrs(op, attrs, num_inputs, num_params, param_keys, param_vals);
  *out = attrs;
  API_END();
}

int MXCachedFree(CachedOpHandle handle) {
  nnvm::NodeAttrs *attrs = static_cast<nnvm::NodeAttrs*>(handle);

  API_BEGIN();
  delete attrs;
  API_END();
}

int MXCachedInvoke(CachedOpHandle handle,
                   int num_inputs,
                   NDArrayHandle *inputs,
                   int *num_outputs,
                   NDArrayHandle **outputs) {
  nnvm::NodeAttrs *attrs = static_cast<nnvm::NodeAttrs*>(handle);

  API_BEGIN();
  ImperativeInvokeImpl(*attrs, num_inputs, inputs, num_outputs, outputs);
  API_END();
}

int MXAutogradSetIsTraining(int is_training, int* prev) {
  API_BEGIN();
  *prev = AutogradRuntime::Get()->SetIsTraining(static_cast<bool>(is_training));
  API_END();
}

int MXAutogradMarkVariables(mx_uint num_var,
                            NDArrayHandle *var_handles,
                            mx_uint *reqs_array,
                            NDArrayHandle *grad_handles) {
  API_BEGIN();
  std::vector<NDArray*> variables, gradients;
  std::vector<mx_uint> grad_reqs;
  variables.reserve(num_var);
  gradients.reserve(num_var);
  grad_reqs.reserve(num_var);
  for (mx_uint i = 0; i < num_var; ++i) {
    variables.emplace_back(static_cast<NDArray*>(var_handles[i]));
    gradients.emplace_back(static_cast<NDArray*>(grad_handles[i]));
    grad_reqs.emplace_back(reqs_array[i]);
  }
  AutogradRuntime::Get()->MarkVariables(variables, grad_reqs, gradients);
  API_END();
}

int MXAutogradComputeGradient(mx_uint num_output,
                              NDArrayHandle *output_handles) {
  return MXAutogradBackward(num_output, output_handles, nullptr, 0);
}

int MXAutogradBackward(mx_uint num_output,
                       NDArrayHandle *output_handles,
                       NDArrayHandle *ograd_handles,
                       int retain_graph) {
  API_BEGIN();

  std::vector<NDArray> outputs, ograds;
  outputs.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    outputs.emplace_back(*static_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr && ograd_handles[i] != nullptr) {
      ograds.emplace_back(*static_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back();
    }
  }

  AutogradRuntime::Get()->ComputeGradient(outputs, ograds, retain_graph);
  API_END();
}
