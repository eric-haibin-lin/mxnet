/*!
 *  Copyright (c) 2015 by Contributors
 * \file c_api_error.h
 * \brief Error handling for C API.
 */
#ifndef MXNET_C_API_C_API_COMMON_H_
#define MXNET_C_API_C_API_COMMON_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <mxnet/c_api.h>
#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <vector>
#include <string>

/*! \brief  macro to guard beginning and end section of all functions */
#define API_BEGIN() try {
/*! \brief every function starts with API_BEGIN();
     and finishes with API_END() or API_END_HANDLE_ERROR */
#define API_END() } catch(dmlc::Error &_except_) { return MXAPIHandleException(_except_); } return 0;  // NOLINT(*)
/*!
 * \brief every function starts with API_BEGIN();
 *   and finishes with API_END() or API_END_HANDLE_ERROR
 *   The finally clause contains procedure to cleanup states when an error happens.
 */
#define API_END_HANDLE_ERROR(Finalize) } catch(dmlc::Error &_except_) { Finalize; return MXAPIHandleException(_except_); } return 0; // NOLINT(*)

/*!
 * \brief Set the last error message needed by C API
 * \param msg The error message to set.
 */
void MXAPISetLastError(const char* msg);
/*!
 * \brief handle exception throwed out
 * \param e the exception
 * \return the return value of API after exception is handled
 */
inline int MXAPIHandleException(const dmlc::Error &e) {
  MXAPISetLastError(e.what());
  return -1;
}

using namespace mxnet;

/*! \brief entry to to easily hold returning information */
struct MXAPIThreadLocalEntry {
  /*! \brief result holder for returning string */
  std::string ret_str;
  /*! \brief result holder for returning strings */
  std::vector<std::string> ret_vec_str;
  /*! \brief result holder for returning string pointers */
  std::vector<const char *> ret_vec_charp;
  /*! \brief result holder for returning handles */
  std::vector<void *> ret_handles;
  /*! \brief result holder for returning shapes */
  std::vector<TShape> arg_shapes, out_shapes, aux_shapes;
  /*! \brief result holder for returning type flags */
  std::vector<int> arg_types, out_types, aux_types;
  /*! \brief result holder for returning shape dimensions */
  std::vector<mx_uint> arg_shape_ndim, out_shape_ndim, aux_shape_ndim;
  /*! \brief result holder for returning shape pointer */
  std::vector<const mx_uint*> arg_shape_data, out_shape_data, aux_shape_data;
  // helper function to setup return value of shape array
  inline static void SetupShapeArrayReturn(
      const std::vector<TShape> &shapes,
      std::vector<mx_uint> *ndim,
      std::vector<const mx_uint*> *data) {
    ndim->resize(shapes.size());
    data->resize(shapes.size());
    for (size_t i = 0; i < shapes.size(); ++i) {
      ndim->at(i) = shapes[i].ndim();
      data->at(i) = shapes[i].data();
    }
  }
};

// define the threadlocal store.
typedef dmlc::ThreadLocalStore<MXAPIThreadLocalEntry> MXAPIThreadLocalStore;

namespace mxnet {
// copy attributes from inferred vector back to the vector of each type.
template<typename AttrType>
inline void CopyAttr(const nnvm::IndexedGraph& idx,
                     const std::vector<AttrType>& attr_vec,
                     std::vector<AttrType>* in_attr,
                     std::vector<AttrType>* out_attr,
                     std::vector<AttrType>* aux_attr) {
  in_attr->clear();
  out_attr->clear();
  aux_attr->clear();
  for (uint32_t nid : idx.input_nodes()) {
    if (idx.mutable_input_nodes().count(nid) == 0) {
      in_attr->push_back(attr_vec[idx.entry_id(nid, 0)]);
    } else {
      aux_attr->push_back(attr_vec[idx.entry_id(nid, 0)]);
    }
  }
  for (auto& e : idx.outputs()) {
    out_attr->push_back(attr_vec[idx.entry_id(e)]);
  }
}

// convert nnvm symbol to a nnvm graph.
inline nnvm::Graph Symbol2Graph(const nnvm::Symbol &s) {
  nnvm::Graph g;
  g.outputs = s.outputs;
  g.attrs["mxnet_version"] = std::make_shared<nnvm::any>(static_cast<int>(MXNET_VERSION));
  return g;
}

template<typename AttrType>
inline void MatchArguments(
    const nnvm::IndexedGraph& idx,
    const std::unordered_map<std::string, AttrType>& known_arg_attrs,
    std::vector<AttrType>* arg_attrs,
    const char* source) {
  auto& arg_nodes = idx.input_nodes();
  CHECK_EQ(arg_attrs->size(), arg_nodes.size());
  size_t nmatched = 0;
  for (size_t i = 0; i < arg_nodes.size(); ++i) {
    const std::string& name = idx[arg_nodes[i]].source->attrs.name;
    auto it = known_arg_attrs.find(name);
    if (it != known_arg_attrs.end()) {
      arg_attrs->at(i) = it->second;
      ++nmatched;
    }
  }
  if (nmatched != known_arg_attrs.size()) {
    std::unordered_set<std::string> keys;
    std::ostringstream head, msg;
    msg << "\nCandidate arguments:\n";
    for (size_t i = 0; i < arg_nodes.size(); ++i) {
      std::string arg_name = idx[arg_nodes[i]].source->attrs.name;
      keys.insert(arg_name);
      msg << "\t[" << i << ']' << arg_name << '\n';
    }
    for (const auto& kv : known_arg_attrs) {
      const std::string& key = kv.first;
      if (keys.count(key) == 0) {
        LOG(FATAL) << source
                   << "Keyword argument name " << key << " not found."
                   << msg.str();
      }
    }
  }
}

// stores keys that will be converted to __key__
extern const std::vector<std::string> kHiddenKeys;
}  // namespace mxnet

#endif  // MXNET_C_API_C_API_COMMON_H_
