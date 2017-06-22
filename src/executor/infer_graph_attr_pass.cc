/*!
 *  Copyright (c) 2017 by Contributors
 * \file infer_graph_attr_pass.cc
 * \brief infer graph shape, dtype, and storage type
 */

#include <mxnet/op_attr_types.h>
#include "./exec_pass.h"

namespace mxnet {
namespace exec {

template<typename AttrType>
nnvm::Graph InitAttrVector(nnvm::Graph&& g,
                           const std::string& attr_input_name,
                           std::vector<AttrType>* attr_vec) {
  const AttrVector& attr_args= g.GetAttr<AttrVector>(attr_input_name);
  CHECK_LE(attr_args.size(), idx.input_nodes().size())
      << "More provided shapes than number of arguments.";
  for (size_t i = 0; i < attr_args.size(); ++i) {
    (*attr_vec)[idx.entry_id(idx.input_nodes()[i], 0)] = attr_args[i];
  }

  // specialize for attr_input_name = "storage_type_inputs"
  // populate ctx for 
  if (attr_input_name == "storage_type_inputs") {
  }

  // erase the provided arguments
  g.attrs.erase(attr_input_name);
  return g;
}

template<typename AttrType, typename IsNone, typename FDefault>
nnvm::Graph InferAttr(nnvm::Graph &&ret,
                      const AttrType empty_val,
                      const char* infer_name,
                      const char* input_name,
                      const char* attr_key_name,
                      const char* attr_name,
                      const char* unknown_name,
                      IsNone fis_none,
                      FDefault fdefault,
                      bool backward_identity_assign) {
  using nnvm::IndexedGraph;
  using nnvm::Op;
  using AttrVector = std::vector<AttrType>;
  using dmlc::any;

  const IndexedGraph& idx = ret.indexed_graph();
  static auto& finfer_shape =
      Op::GetAttr<nnvm::FInferNodeEntryAttr<AttrType> >(infer_name);
  static auto& is_backward =
      Op::GetAttr<nnvm::TIsBackward>("TIsBackward");
  // gradient function, used to get node correspondence.
  static auto& fgrad =
      Op::GetAttr<nnvm::FGradient>("FGradient");
  // reshape shape vector
  AttrVector rshape;
  if (ret.attrs.count(attr_name) != 0) {
    rshape = ret.MoveCopyAttr<AttrVector>(attr_name);
  } else {
    rshape.resize(idx.num_node_entries(), empty_val);
  }

  if (ret.attrs.count(input_name) != 0) {
    const AttrVector& shape_args = ret.GetAttr<AttrVector>(input_name);
    CHECK_LE(shape_args.size(), idx.input_nodes().size())
        << "More provided shapes than number of arguments.";
    for (size_t i = 0; i < shape_args.size(); ++i) {
      rshape[idx.entry_id(idx.input_nodes()[i], 0)] = shape_args[i];
    }
    // erase the provided arguments
    ret.attrs.erase(input_name);
  }

  // get the shape hints
  std::string shape_hints_key = std::string(attr_name) + "_hints";
  if (ret.attrs.count(shape_hints_key)) {
    nnvm::NodeEntryMap<AttrType> shape_hints =
      ret.GetAttr<nnvm::NodeEntryMap<AttrType>>(shape_hints_key);
    for (const auto& kv : shape_hints) {
      nnvm::NodeEntry e = kv.first;
      if (idx.exist(e.node.get())) {
        rshape[idx.entry_id(kv.first)] = kv.second;
      }
    }
  }

  std::string shape_attr_key;
  if (ret.attrs.count(attr_key_name) != 0) {
    shape_attr_key = ret.GetAttr<std::string>(attr_key_name);
    // erase the provided arguments
    ret.attrs.erase(attr_key_name);
  }
  // Temp space for shape inference.
  std::vector<AttrType> ishape, oshape;

  // inference step function for nid
  auto infer_step = [&](uint32_t nid, bool last_iter) {
    const auto& inode = idx[nid];
    const uint32_t num_inputs = inode.inputs.size();
    const uint32_t num_outputs = inode.source->num_outputs();
    if (inode.source->is_variable()) {
      // Variable node. No operator. Only one output entry.
      CHECK(inode.source->op() == nullptr);
      CHECK_EQ(num_outputs, 1U);
      const uint32_t out_ent_id = idx.entry_id(nid, 0);
      if (shape_attr_key.length() != 0 && fis_none(rshape[out_ent_id])) {
        auto it = inode.source->attrs.dict.find(shape_attr_key);
        if (it != inode.source->attrs.dict.end()) {
          std::istringstream is(it->second);
          CHECK(is >> rshape[out_ent_id]) << "Invalid attribute";
        }
      }
    } else if (is_backward.get(inode.source->op(), false) &&
               inode.control_deps.size() && backward_identity_assign) {
      CHECK_GE(inode.control_deps.size(), 1U)
        << "BackwardOp need to have control_deps to its forward op";
      const IndexedGraph::Node& fnode = idx[inode.control_deps[0]];
      nnvm::NodePtr fwd_ptr = inode.source->control_deps[0];
      CHECK(fwd_ptr->op() != nullptr) << "Forward op cannot be a variable";
      // use gradient function to find out the correspondence.
      std::vector<nnvm::NodeEntry> ograd(fwd_ptr->num_outputs());
      for (size_t i = 0; i < ograd.size(); ++i) {
        ograd[i].index = static_cast<uint32_t>(i);
      }
      // input gradient list
      auto igrad = fgrad[fwd_ptr->op()](fwd_ptr, ograd);
      const nnvm::Node* igrad_node = nullptr;
      // Input gradient assignement
      for (size_t i = 0; i < igrad.size(); ++i) {
        if (igrad[i].node->op() == inode.source->op()) {
          uint32_t eid = idx.entry_id(nid, igrad[i].index);
          if (fis_none(rshape[eid])) {
            rshape[eid] = rshape[idx.entry_id(fnode.inputs[i])];
          } else {
            CHECK_EQ(rshape[eid], rshape[idx.entry_id(fnode.inputs[i])])
                << "Backward shape inconsistent with the forward shape";
          }
          if (igrad_node == nullptr) {
            igrad_node = igrad[i].node.get();
          } else {
            CHECK(igrad_node == igrad[i].node.get());
          }
        }
      }
      // out grad entries
      CHECK(igrad_node != nullptr)
        << "Cannot find matching backward op for " << inode.source->attrs.name;
      for (size_t i = 0; i < igrad_node->inputs.size(); ++i) {
        const nnvm::NodeEntry& e = igrad_node->inputs[i];
        if (e.node == nullptr) {
          uint32_t eid = idx.entry_id(inode.inputs[i]);
          if (fis_none(rshape[eid])) {
            rshape[eid] = rshape[idx.entry_id(inode.control_deps[0], e.index)];
          }
        }
      }
    } else {
      bool forward_known = true;
      // Forward operator inference.
      ishape.resize(num_inputs, empty_val);
      for (uint32_t i = 0; i < ishape.size(); ++i) {
        ishape[i] = rshape[idx.entry_id(inode.inputs[i])];
        if (fis_none(ishape[i])) forward_known = false;
      }
      oshape.resize(num_outputs, empty_val);
      for (uint32_t i = 0; i < oshape.size(); ++i) {
        oshape[i] = rshape[idx.entry_id(nid, i)];
        if (fis_none(oshape[i])) forward_known = false;
      }
      auto finfer = finfer_shape.get(inode.source->op(), fdefault);
      if (!forward_known) {
        if (finfer != nullptr) {
          // Call inference function of the operator.
          try {
            forward_known = finfer(inode.source->attrs, &ishape, &oshape);
          } catch (const std::exception& e) {
            throw dmlc::Error("Error in operator " + inode.source->attrs.name + ": " + e.what());
          }
        } else {
          CHECK(!last_iter)
              << "Attribute " << infer_name
              << " is not registed by op " << inode.source->op()->name
              << " we are not able to complete the inference because of this";
        }
      }
      // Save to the result map.
      for (uint32_t i = 0; i < num_inputs; ++i) {
        rshape[idx.entry_id(inode.inputs[i])] = ishape[i];
      }
      for (uint32_t i = 0; i < num_outputs; ++i) {
        rshape[idx.entry_id(nid, i)] = oshape[i];
      }
    }
  };

  size_t last_num_unknown;
  size_t num_unknown = rshape.size();
  int i = 0;
  do {
    if (i % 2 == 0) {
      for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
        infer_step(nid, false);
      }
    } else {
      // backward inference
      for (uint32_t i = idx.num_nodes(); i != 0; --i) {
        infer_step(i - 1, false);
      }
    }
    last_num_unknown = num_unknown;
    num_unknown = 0;
    for (size_t j = 0; j < idx.num_node_entries(); ++j) {
      if (fis_none(rshape[j])) {
        ++num_unknown;
      }
    }
    ++i;
  } while (num_unknown > 0 && last_num_unknown > num_unknown);
  // set the shapes
  ret.attrs[attr_name] = std::make_shared<any>(std::move(rshape));
  // number of nodes who knows the shape.
  ret.attrs[unknown_name] = std::make_shared<any>(num_unknown);
  return ret;
}

// inference fucntion for same type
inline bool SameType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *iattr,
                     std::vector<int> *oattr) {
  int def_v = -1;
  for (int v : *oattr) {
    if (v != -1) {
      def_v = v; break;
    }
  }
  if (def_v == -1) {
    for (int v : *iattr) {
      if (v != -1) {
        def_v = v; break;
      }
    }
  }
  if (def_v == -1) return false;
  for (int& v : *oattr) {
    v = def_v;
  }
  for (int& v : *iattr) {
    v = def_v;
  }
  return true;
}

// assigning default type N to both input and output attrs with value -1
template <int default_val, int none>
inline bool DefaultType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *iattr,
                        std::vector<int> *oattr) {
  for (int& v : *oattr) {
    if (v == none) v = default_val;
  }
  for (int& v : *iattr) {
    if (v == none) v = default_val;
  }
  return true;
}

nnvm::Graph InferShape(nnvm::Graph graph,
                       nnvm::ShapeVector shape_inputs,
                       const std::string& shape_attr_key) {
  using dmlc::any;
  if (shape_inputs.size() != 0) {
    graph.attrs["shape_inputs"] = std::make_shared<any>(std::move(shape_inputs));
  }
  if (shape_attr_key.length() != 0) {
    graph.attrs["shape_attr_key"] = std::make_shared<any>(std::move(shape_attr_key));
  }
  return InferAttr<nnvm::TShape>(
      std::move(graph), nnvm::TShape(),
      "FInferShape", "shape_inputs", "shape_attr_key",
      "shape", "shape_num_unknown_nodes",
      [](const nnvm::TShape& s) { return s.ndim() == 0 || s.Size() == 0; },
      nullptr, true);
}

nnvm::Graph InferType(nnvm::Graph graph,
                      nnvm::DTypeVector dtype_inputs,
                      const std::string& dtype_attr_key) {
  using dmlc::any;
  if (dtype_inputs.size() != 0) {
    graph.attrs["dtype_inputs"] = std::make_shared<any>(std::move(dtype_inputs));
  }
  if (dtype_attr_key.length() != 0) {
    graph.attrs["dtype_attr_key"] = std::make_shared<any>(std::move(dtype_attr_key));
  }
  return InferAttr<int>(
      std::move(graph), -1,
      "FInferType", "dtype_inputs", "dtype_attr_key",
      "dtype", "dtype_num_unknown_nodes",
      [](const int t) { return t == -1; },
      SameType, true);
}

nnvm::Graph InferStorageType(nnvm::Graph graph,
                             nnvm::StorageTypeVector storage_type_inputs,
                             const std::string& storage_type_attr_key) {
  using dmlc::any;
  if (storage_type_inputs.size() != 0) {
    graph.attrs["storage_type_inputs"] = std::make_shared<any>(std::move(storage_type_inputs));
  }
  if (storage_type_attr_key.length() != 0) {
    graph.attrs["storage_type_attr_key"] = std::make_shared<any>(std::move(storage_type_attr_key));
  }
  // for storage type, the backward attr is not necessarily the same as it's correspondence
  const int kDefaultStorage = 0;
  return InferAttr<int>(
      std::move(graph), -1,
      "FInferStorageType", "storage_type_inputs", "storage_type_attr_key",
      "storage_type", "storage_type_num_unknown_nodes",
      [](const int t) { return t == -1; },
      DefaultType<kDefaultStorage, -1>, false);
}

}  // namespace exec
}  // namespace mxnet
