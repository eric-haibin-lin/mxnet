/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
* \file elemwise_op_common.h
* \brief common function used for broadcasting and reducing
* \author Xingjian Shi
*/
#ifndef MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#define MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
#include <dmlc/logging.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <nnvm/op.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mxnet_op.h"

namespace mxnet {
namespace op {
template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         std::string (*attr_string)(const AttrType&),
         int n_in = -1, int n_out = -1>
inline bool ElemwiseAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs,
                         const AttrType& none) {
  AttrType dattr = none;
  size_t in_size = in_attrs->size();
  size_t out_size = out_attrs->size();
  if (n_in != -1)
    in_size = static_cast<size_t>(n_in);
  if (n_out != -1)
    out_size = static_cast<size_t>(n_out);

  auto deduce = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&dattr, (*vec)[i]))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  deduce(in_attrs, in_size, "input");
  if (reverse_infer) deduce(out_attrs, out_size, "output");

  auto write = [&](std::vector<AttrType> *vec, size_t size, const char *name) {
      for (size_t i = 0; i < size; ++i) {
        CHECK(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << attr_string(dattr)
          << ", got " << attr_string((*vec)[i]);
      }
    };
  write(in_attrs, in_size, "input");
  write(out_attrs, out_size, "output");

  if (is_none(dattr)) return false;
  return true;
}

// Only inferring output storage types from input for now
template<typename AttrType, bool (*is_none)(const AttrType&),
         bool (*assign)(AttrType*, const AttrType&), bool reverse_infer,
         bool enable_fallback>
inline bool ElemwiseStorageAttr(const nnvm::NodeAttrs& attrs,
                         std::vector<AttrType> *in_attrs,
                         std::vector<AttrType> *out_attrs) {
  auto deduce = [&](std::vector<AttrType> *vec, const char *name, AttrType& result,
                    bool fallback) {
      auto &v = *vec;
      for (size_t i = 0; i < vec->size(); ++i) {
        if (v[i] == kUndefinedStorage) {
          // if input type is unknown, assume it's default storage
          CHECK(assign(&v[i], kDefaultStorage));
        } else if (assign(&result, v[i]) == false && fallback) {
          result = kDefaultStorage;
        }
      }
    };
  AttrType dattr = kUndefinedStorage;
  deduce(in_attrs, "input", dattr, enable_fallback);
  if (reverse_infer) {
    LOG(FATAL) << "not implemented yet";
  }
  auto write = [&](std::vector<AttrType> *vec, const char *name) {
      for (size_t i = 0; i < vec->size(); ++i) {
        CHECK(assign(&(*vec)[i], dattr))
          << "Incompatible attr in node " << attrs.name << " at " << i << "-th "
          << name << ": " << "expected " << dattr << ", got " << (*vec)[i];
      }
    };
  if (is_none(dattr)) dattr = kDefaultStorage;
  write(out_attrs, "output");
  return true;
}

template<int n_in, int n_out>
inline bool ElemwiseShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<TShape, shape_is_none, shape_assign, true, shape_string>(
    attrs, in_attrs, out_attrs, TShape());
}

template<int n_in, int n_out>
inline bool ElemwiseType(const nnvm::NodeAttrs& attrs,
                         std::vector<int> *in_attrs,
                         std::vector<int> *out_attrs) {
  if (n_in != -1) {
    CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  }
  if (n_out != -1) {
    CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  }
  return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
    attrs, in_attrs, out_attrs, -1);
}

template<int n_in, int n_out>
inline bool ElemwiseStorageType(const nnvm::NodeAttrs& attrs,
                                const Context& ctx,
                                std::vector<int> *in_attrs,
                                std::vector<int> *out_attrs) {
  // TODO(junwu): add ctx info into storage inference logic
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  return ElemwiseStorageAttr<int, type_is_none, type_assign, false, true>(
    attrs, in_attrs, out_attrs);
}

template<int n_out>
inline bool ElemwiseStorageTypeDenseOutput(const nnvm::NodeAttrs& attrs,
                                           const Context& ctx,
                                           std::vector<int> */*in_attrs*/,
                                           std::vector<int> *out_attrs) {
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  for (size_t i = 0; i < n_out; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, kDefaultStorage);
  }
  return true;
}

template<int n_in, int n_out, int n_force_out_as_input_index>
inline bool ElemwiseStorageTypeForce(const nnvm::NodeAttrs& attrs,
                                     const Context& ctx,
                                     std::vector<int> *in_attrs,
                                     std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  CHECK_LT(n_force_out_as_input_index, n_in) << "";
  for (size_t i = 0; i < n_out; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, (*in_attrs)[n_force_out_as_input_index]);
  }
  return true;
}

/*! \brief Set outputs as storage type of first non-dense input
 *  Optionally specify a "preferred" output storage type in case they vary
 */
template<int n_in, int n_out, NDArrayStorageType prefer = kUndefinedStorage>
inline bool ElemwiseStorageTypeLeastDense(const nnvm::NodeAttrs& attrs,
                                          const Context& ctx,
                                          std::vector<int> *in_attrs,
                                          std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(n_in)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(n_out)) << " in operator " << attrs.name;
  NDArrayStorageType stype = kDefaultStorage;
  for (size_t i = 0; i < n_in; ++i) {
    const NDArrayStorageType in_stype = static_cast<NDArrayStorageType>((*in_attrs)[i]);
    if (in_stype != kDefaultStorage) {
      if (stype == kDefaultStorage || in_stype == prefer) {
        stype = in_stype;
      }
      if (prefer == kUndefinedStorage || stype == prefer) {
        break;
      }
    }
  }
  for (size_t i = 0; i < n_out; ++i) {
    STORAGE_TYPE_ASSIGN_CHECK(*out_attrs, i, stype);
  }
  return true;
}

inline bool IdentityAttrLikeRhsStorageType(const nnvm::NodeAttrs& attrs,
                                           const Context& ctx,
                                           std::vector<int> *in_attrs,
                                           std::vector<int> *out_attrs) {
  // TODO(junwu): add ctx info into storage inference logic
  CHECK_EQ(in_attrs->size(), static_cast<size_t>(2)) << " in operator " << attrs.name;
  CHECK_EQ(out_attrs->size(), static_cast<size_t>(1)) << " in operator " << attrs.name;
  auto &in = *in_attrs;
  auto &out = *out_attrs;
  CHECK_NE(in[1], kUndefinedStorage) << "rhs storage type must be known";
  if (in[0] == kUndefinedStorage) in[0] = in[1];
  if (out[0] == kUndefinedStorage) out[0] = in[1];
  return true;
}

// Transfer gradient and input to FGradient function
struct ElemwiseGradUseIn {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode(op_name, n, ograds, n->inputs, n->attrs.dict);
  }
};

// Transfer gradient and output to FGradient function
struct ElemwiseGradUseOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads;
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
    return MakeNonlossGradNode(op_name, n, ograds, heads, n->attrs.dict);
  }
};

// Transfer gradient and input and output to FGradient function
struct ElemwiseGradUseInOut {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    for (auto& h : n->inputs) {
      heads.push_back(h);
    }
    index_t n_out = n->num_outputs();
    for (index_t i = 0; i < n_out; ++i) {
      heads.emplace_back(nnvm::NodeEntry{n, i, 0});
    }
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
  }
};

// Transfer only gradient to FGradient function
struct ElemwiseGradUseNone {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    return MakeNonlossGradNode(op_name, n, ograds, {}, n->attrs.dict);
  }
};

/*! \brief Generic conversion of F<OP> kernel mapping to Kernel::Launch mapping */
template<typename OP, int Req>
struct BMap {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out,
                                  const DType *lhs,
                                  const DType *rhs) {
    KERNEL_ASSIGN(out[i], Req, OP::Map(lhs[i], rhs[i]));
  }

  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in, const DType value) {
    KERNEL_ASSIGN(out[i], Req, OP::Map(in[i], value));
  }
};

// TODO(haibin) this is a temporary function for debugging purpose. Remove later.
#ifndef NDEBUG
template <int dim, typename DType>
void print_info(const mshadow::Tensor<cpu, dim, DType>& tensor, const std::string& name) {
  std::cout << "Tensor " << name << " with shape (";
  int len = 1;
  for (int i = 0; i < dim; i++) {
    len *= tensor.shape_[i];
    std::cout << tensor.shape_[i] << ",";
    if (i == dim - 1) std::cout << ")";
  }
  std::cout << std::endl;
  for (int j = 0; j < len; j ++) std::cout << tensor.dptr_[j] << " ";
  std::cout << std::endl;
}
#endif  // NDEBUG

struct CloneGradient {
  const char *op_name;
  std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                          const std::vector<nnvm::NodeEntry>& ograds) {
    std::vector<nnvm::NodeEntry> ret;
    for (size_t i = 0; i < n->inputs.size(); ++i)
      ret.emplace_back(ograds[0]);
    return ret;
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_ELEMWISE_OP_COMMON_H_
