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
 *  Copyright (c) 2016 by Contributors
 * \file ordering_op-inl.h
 * \brief Function definition of ordering operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_
#define MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <algorithm>
#include <vector>
#include <type_traits>
#include "./indexing_op.h"

namespace mshadow {
template<typename xpu, int src_dim, typename DType, int dst_dim>
inline Tensor<xpu, dst_dim, DType> inplace_reshape(const Tensor<xpu, src_dim, DType> &src,
                                                   const Shape<dst_dim> &target_shape) {
  CHECK_EQ(src.CheckContiguous(), true);
  return Tensor<xpu, dst_dim, DType>(src.dptr_, target_shape, src.stream_);
}
};


namespace mxnet {
namespace op {
// These enums are only visible within this header
namespace topk_enum {
enum TopKReturnType {kReturnValue, kReturnIndices, kReturnMask, kReturnBoth};
}  // topk_enum

struct TopKParam : public dmlc::Parameter<TopKParam> {
  dmlc::optional<int> axis;
  int k;
  int ret_typ;
  bool is_ascend;
  int dtype;
  DMLC_DECLARE_PARAMETER(TopKParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to choose the top k indices."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(k).set_default(1)
    .describe("Number of top elements to select,"
              " should be always smaller than or equal to the element number in the given axis."
              " A global sort is performed if set k < 1.");
    DMLC_DECLARE_FIELD(ret_typ).set_default(topk_enum::kReturnIndices)
    .add_enum("value", topk_enum::kReturnValue)
    .add_enum("indices", topk_enum::kReturnIndices)
    .add_enum("mask", topk_enum::kReturnMask)
    .add_enum("both", topk_enum::kReturnBoth)
    .describe("The return type.\n"
        " \"value\" means to return the top k values,"
        " \"indices\" means to return the indices of the top k values,"
        " \"mask\" means to return a mask array containing 0 and 1. 1 means the top k values."
        " \"both\" means to return a list of both values and indices of top k elements.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(false)
      .describe("Whether to choose k largest or k smallest elements."
                " Top K largest elements will be chosen if set to false.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("int64", mshadow::kInt64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output indices. It is only valid when ret_typ is \"indices\" or"
                " \"both\". An error will be raised if the selected data type cannot precisely "
                "represent the indices.");
  }
};

struct SortParam : public dmlc::Parameter<SortParam> {
  dmlc::optional<int> axis;
  bool is_ascend;
  DMLC_DECLARE_PARAMETER(SortParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to choose sort the input tensor."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order.");
  }
};

struct ArgSortParam : public dmlc::Parameter<ArgSortParam> {
  dmlc::optional<int> axis;
  bool is_ascend;
  int dtype;
  DMLC_DECLARE_PARAMETER(ArgSortParam) {
    DMLC_DECLARE_FIELD(axis).set_default(dmlc::optional<int>(-1))
    .describe("Axis along which to sort the input tensor."
              " If not given, the flattened array is used. Default is -1.");
    DMLC_DECLARE_FIELD(is_ascend).set_default(true)
      .describe("Whether to sort in ascending or descending order.");
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .add_enum("int64", mshadow::kInt64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .set_default(mshadow::kFloat32)
    .describe("DType of the output indices. It is only valid when ret_typ is \"indices\" or"
              " \"both\". An error will be raised if the selected data type cannot precisely "
              "represent the indices.");
  }
};

inline void ParseTopKParam(const mxnet::TShape& src_shape, const TopKParam& param,
                           mxnet::TShape *target_shape, int *batch_size, int *element_num,
                           int *axis, int *k, bool *do_transpose, bool *is_ascend) {
  *do_transpose = false;
  *k = param.k;
  *is_ascend = param.is_ascend;
  // get batch_size, axis and element_num
  if (!static_cast<bool>(param.axis)) {  // No axis given
    *axis = 0;
    *batch_size = 1;
    *element_num = src_shape.Size();
  } else {
    *axis = param.axis.value();
    if (*axis < 0) {
      *axis += src_shape.ndim();
    }
    CHECK(*axis >= 0 && *axis < static_cast<int>(src_shape.ndim()))
                                                  << "Invalid axis! axis should be between 0 and "
                                                  << src_shape.ndim() << ", found axis=" << *axis;
    *batch_size = src_shape.Size() / src_shape[*axis];
    *element_num = src_shape[*axis];
    if (*axis != src_shape.ndim() - 1) {
      *do_transpose = true;
    }
  }
  // get k
  if (param.k <= 0) {
    *k = *element_num;
  }
  // get target_shape
  if (!static_cast<bool>(param.axis)) {
    if (param.ret_typ != topk_enum::kReturnMask) {
      *target_shape = mshadow::Shape1(*k);
    } else {
      *target_shape = src_shape;
    }
  } else {
    *target_shape = src_shape;
    if (param.ret_typ != topk_enum::kReturnMask) {
      (*target_shape)[*axis] = *k;
    }
  }
  CHECK(*k >= 1 && *k <= *element_num) << "k must be smaller than "
                                      << *element_num << ", get k = " << *k;
}

using namespace mshadow;


struct fill_ind_to_one {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const int* indices, DType* out) {
    out[indices[i]] = static_cast<DType>(1);
  }
};

struct fill_ind {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const int* indices, const DType* val,
                                  int req, DType* out) {
    KERNEL_ASSIGN(out[indices[i]], req, val[i]);
  }
};

template<typename DType>
MSHADOW_FORCE_INLINE void TopKSort(const Tensor<cpu, 1, DType>& dat,
                                   const Tensor<cpu, 1, int>& ind,
                                   const Tensor<cpu, 1, char>& work,
                                   int K, int N, bool is_ascend,
                                   Stream<cpu> *s) {
  // Use full sort when K is relatively large.
  const bool full_sort(K*8 > N);
  // Batch size.
  const int M(work.size(0)/(sizeof(DType)*N));
  const int omp_threads(engine::OpenMP::Get()->GetRecommendedOMPThreadCount());
  #pragma omp parallel for num_threads(omp_threads)
  for (int i = 0; i < M; ++i) {
    // Tensor `work` stores the flattened source data, while `dat` stores the sorted result.
    DType *vals = reinterpret_cast<DType*>(work.dptr_);
    DType *sorted_vals = dat.dptr_+i*N;
    int *indices = ind.dptr_+i*N;
    if (is_ascend) {
      if (full_sort) {
        std::sort(indices, indices+N,
                  [&](const int& i1, const int& i2){ return vals[i1] < vals[i2]; });
      } else {
        std::partial_sort(indices, indices+K, indices+N,
                          [&](const int& i1, const int& i2){ return vals[i1] < vals[i2]; });
      }
    } else {
      if (full_sort) {
        std::sort(indices, indices+N,
                  [&](const int& i1, const int& i2){ return vals[i1] > vals[i2]; });
      } else {
        std::partial_sort(indices, indices+K, indices+N,
                          [&](const int& i1, const int& i2){ return vals[i1] > vals[i2]; });
      }
    }
    for (int j = 0; j < K; ++j) {
      sorted_vals[j] = vals[indices[j]];
    }
  }
}

#ifdef __CUDACC__

template<typename DType>
MSHADOW_XINLINE bool TopKCompare(DType val1, int ind1, DType val2, int ind2, bool is_ascend) {
  // Negative indices denote undefined values which are considered arbitrary small resp. large.
  return (ind2 < 0) || (ind1 >= 0 && ((is_ascend && val1 < val2) || (!is_ascend && val1 > val2)));
}

template<typename DType>
MSHADOW_XINLINE void MergeTopK(int K, DType *val1, int *ind1, DType *val2, int *ind2,
                               bool is_ascend) {
  // In-place merge of two sorted top-K lists into val1/ind1. First determine the intervals
  // [0,..,i1], [0,..i2] of the two lists that will be part of the merged list.
  int i1(K-1), i2(K-1);
  for (int i = 0; i < K; ++i) {
    if (TopKCompare(val1[i1], ind1[i1], val2[i2], ind2[i2], is_ascend)) {
      --i2;
    } else {
      --i1;
    }
  }
  // Now merge the lists from back to front.
  for (int i = K; i--;) {
    if (i2 < 0 || i1 >= 0 && TopKCompare(val2[i2], ind2[i2], val1[i1], ind1[i1], is_ascend)) {
      val1[i] = val1[i1];
      ind1[i] = ind1[i1];
      --i1;
    } else {
      val1[i] = val2[i2];
      ind1[i] = ind2[i2];
      --i2;
    }
  }
}

template<typename DType>
__global__ void PartialSortSmallK(int K, int N, DType *val, int *ind, bool is_ascend) {
  // Buffer for pairwise reduction.
  extern __shared__ int buff[];
  // Start of buffer sections associated with this thread.
  const int offset(threadIdx.x*K);
  int *ind_buff = &buff[offset];
  DType *val_buff = reinterpret_cast<DType*>(&buff[blockDim.x*K])+offset;
  // Initialize top-K values for this thread.
  for (int i = 0; i < K; ++i) {
    ind_buff[i] = -1;
  }
  // Range of values this thread cares about. Each thread block processes
  // a different batch item (i.e. a different set of ind/val where we
  // have to select the top-K elements). All threads within the same
  // block work on the same batch item.
  const int first(blockIdx.x*N+threadIdx.x), last((blockIdx.x+1)*N);
  // Select top-K from this range and store it sorted in the buffer.
  // We assume a small K, so linear insertion is o.k.
  for (int i = first; i < last; i += blockDim.x) {
    DType cur_val(val[i]);
    int cur_ind(ind[i]);
    for (int j = K; j-- && TopKCompare(cur_val, cur_ind, val_buff[j], ind_buff[j], is_ascend); ) {
      if (j+1 < K) {
        val_buff[j+1] = val_buff[j];
        ind_buff[j+1] = ind_buff[j];
      }
      val_buff[j] = cur_val;
      ind_buff[j] = cur_ind;
    }
  }
  // Recursive merge of sorted lists for this thread block. Note that blockDim.x is not
  // necessary a power of two, therefore the additional checks for last_s.
  for (unsigned int s = (blockDim.x+1)/2, last_s = blockDim.x;
       last_s > 1; last_s = s, s = (s+1)/2) {
    __syncthreads();
    if (threadIdx.x < s && threadIdx.x+s < last_s) {
      MergeTopK(K, val_buff, ind_buff, val_buff+s*K, ind_buff+s*K, is_ascend);
    }
  }
  // Final updates on master thread.
  if (threadIdx.x == 0) {
    for (int i = 0; i < K; ++i) {
      ind[blockIdx.x*N+i] = ind_buff[i];
      val[blockIdx.x*N+i] = val_buff[i];
    }
  }
}

template<typename DType>
MSHADOW_FORCE_INLINE void TopKSort(const Tensor<gpu, 1, DType>& dat,
                                   const Tensor<gpu, 1, int>& ind,
                                   const Tensor<gpu, 1, char>& work,
                                   int K, int N, bool is_ascend,
                                   Stream<gpu> *s) {
  // Use full sort for all but very small K for which we
  // can do a partial sort entirely within shared memory.
  const bool full_sort(K > 5);
  // Batch size.
  const int M(dat.size(0)/N);
  if (full_sort) {
    // Divide workspace into two parts. The first one is needed to store batch ids.
    const size_t alignment = std::max(sizeof(DType), sizeof(int));
    const size_t id_size = PadBytes(sizeof(int) * ind.size(0), alignment);
    Tensor<gpu, 1, int> batch_id(reinterpret_cast<int*>(work.dptr_), Shape1(ind.size(0)), s);
    Tensor<gpu, 1, char> sort_work(work.dptr_+id_size, Shape1(work.size(0)-id_size), s);
    mxnet::op::SortByKey(dat, ind, is_ascend, &sort_work);
    if (M > 1) {
      // Back to back sorting. Note that mxnet::op::SortByKey is a stable sort.
      batch_id = ind / N;
      mxnet::op::SortByKey(batch_id, dat, true, &sort_work);
      batch_id = ind / N;
      mxnet::op::SortByKey(batch_id, ind, true, &sort_work);
    }
  } else {
    const int nthreads(mshadow::cuda::kBaseThreadNum);
    PartialSortSmallK<<<M, nthreads, nthreads*K*(sizeof(int)+sizeof(DType)),
                        mshadow::Stream<gpu>::GetStream(s)>>>
                        (K, N, dat.dptr_, ind.dptr_, is_ascend);
  }
}

#endif

template<typename xpu, typename DType>
size_t GetMemorySize(const TBlob& src) {
  const auto srcSize = src.Size();
  const auto alignment = std::max(sizeof(DType), sizeof(int));
  size_t temp_size = 0;
  // Temp space needed by the gpu-based full sorts.
  temp_size = std::max(temp_size,
                     mxnet::op::SortByKeyWorkspaceSize<int, int, xpu>(srcSize));
  temp_size = std::max(temp_size,
                       mxnet::op::SortByKeyWorkspaceSize<int, DType, xpu>(srcSize));
  temp_size = std::max(temp_size,
                       mxnet::op::SortByKeyWorkspaceSize<DType, int, xpu>(srcSize));
  // Additional temp space for gpu full sorts for batch ids.
  temp_size += PadBytes(sizeof(int) * srcSize, alignment);
  // Temp space for cpu sorts.
  return std::max(temp_size, static_cast<size_t>(sizeof(DType) * srcSize));
}

typedef void (*topK_func)(const RunContext &ctx,
                          const Resource &resource,
                          const std::vector<OpReqType>& req,
                          const TBlob& src,
                          const size_t temp_size,
                          const std::vector<TBlob>& ret,
                          const TopKParam& param);
/*!
   * \brief Implementation of the TopK operation
   *
   *
   * \param ctx the running context
   * \param resource temporary resource handler
   * \param src the Source blob
   * \param ret the destination blobs
   * \param k the K elements to keep
   * \param param the topk parameters
   * \tparam xpu the device type.
   * \tparam DType type of the output value/mask.
   * \tparam IDType type of the output indices.
   */
template<typename xpu, typename DType, typename IDType>
void TopKImpl(const RunContext &ctx,
              const Resource &resource,
              const std::vector<OpReqType>& req,
              const TBlob& src,
              const size_t temp_size,
              const std::vector<TBlob>& ret,
              const TopKParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  // 1. Parse and initialize information
  Stream<xpu> *s = ctx.get_stream<xpu>();
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  mxnet::TShape target_shape;
  ParseTopKParam(src.shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  CHECK_LE(element_num, mxnet::common::MaxIntegerValue<IDType>())
    << "'IDType' does not have a sufficient precision to represent the indices of the input array. "
    << "The total element_num is " << element_num << ", but the selected IDType can only represent "
    << mxnet::common::MaxIntegerValue<IDType>() << " elements";
  Tensor<xpu, 3, DType> dat = src.FlatTo3D<xpu, DType>(axis, axis, s);
  const auto srcSize = src.Size();
  const auto total_size = batch_size * k;
  const auto retMask = param.ret_typ == topk_enum::kReturnMask;
  const size_t alignment = std::max(sizeof(DType), sizeof(int));
  const auto size_1 = PadBytes(sizeof(DType) * srcSize, alignment);
  const auto size_2 = PadBytes(sizeof(int) * srcSize, alignment);
  const auto size_3 = retMask? PadBytes(sizeof(int) * total_size, alignment) : 0;
  const size_t workspace_size = temp_size + size_1 + size_2 + size_3;

  const Tensor<xpu, 1, char> workspace = resource.get_space_typed<xpu, 1, char>
                                                  (Shape1(workspace_size), s);
  char* workspace_curr_ptr = workspace.dptr_;
  Tensor<xpu, 1, DType>sorted_dat(reinterpret_cast<DType*>(workspace_curr_ptr),
                                      Shape1(srcSize), s);  // contain sorted dat
  workspace_curr_ptr += size_1;

  Tensor<xpu, 1, int> indices(reinterpret_cast<int*>(workspace_curr_ptr),
                                Shape1(srcSize), s);  // indices in the original matrix
  workspace_curr_ptr += size_2;

  Tensor<xpu, 1, int> sel_indices;
  if (retMask) {
    sel_indices = Tensor<xpu, 1, int>(reinterpret_cast<int*>(workspace_curr_ptr),
                                      Shape1(total_size), s);
    workspace_curr_ptr += size_3;
    CHECK_EQ(sel_indices.CheckContiguous(), true);
  }

  Tensor<xpu, 1, char> temp_workspace;
  if (std::is_same<xpu, cpu>::value) {
    Tensor<xpu, 1, DType> flattened_data;
    if (do_transpose) {
      flattened_data = Tensor<xpu, 1, DType>(reinterpret_cast<DType*>(workspace_curr_ptr),
                                              Shape1(srcSize), s);
      flattened_data = reshape(transpose(dat, Shape3(0, 2, 1)), Shape1(srcSize));
      CHECK_EQ(flattened_data.CheckContiguous(), true);
    } else {
      flattened_data = src.FlatTo1D<xpu, DType>(s);
    }
    // `temp_workspace` stores the flattened data
    temp_workspace = Tensor<xpu, 1, char>(reinterpret_cast<char*>(flattened_data.dptr_),
                                          Shape1(sizeof(DType)*srcSize), s);
    CHECK_EQ(temp_workspace.CheckContiguous(), true);
  } else {
    if (do_transpose) {
      sorted_dat = reshape(transpose(dat, Shape3(0, 2, 1)), Shape1(srcSize));
    } else {
      sorted_dat = reshape(dat, Shape1(srcSize));
    }
    CHECK_EQ(sorted_dat.CheckContiguous(), true);
    temp_workspace = Tensor<xpu, 1, char>(workspace_curr_ptr, Shape1(temp_size), s);  // temp space
  }

  mxnet_op::Kernel<range_fwd, xpu>::Launch(s, batch_size * element_num, 1, 0, 1,
    kWriteTo, indices.dptr_);
  CHECK_EQ(indices.CheckContiguous(), true);

  // 2. Perform inplace batch sort.
  // After sorting, each batch in `sorted_dat` will be sorted in the corresponding order
  // up to the k-th element and the `indices` will contain the corresponding index in `sorted_dat`
  // `temp_workspace` is used to store the flattened source data for CPU device, and it's used as
  // a temporal buffer for GPU device.
  TopKSort(sorted_dat, indices, temp_workspace, k, element_num, is_ascend, s);

  // 3. Assign results to the ret blob
  // When returning indices, only update(modulo) required elements instead of full elements
  // to avoid redundant calculation.
  // Cast `ret_indices` from int to real_t could introduce conversion error when the element_num
  // is large enough.
  if (retMask) {
    Tensor<xpu, 1, DType> ret_mask = ret[0].FlatTo1D<xpu, DType>(s);
    ret_mask = scalar<DType>(0);
    sel_indices = reshape(slice<1>(
                              inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k),
                              Shape1(total_size));
    if (do_transpose) {
      mxnet::TShape src_shape = src.shape_.FlatTo3D(axis);
      CHECK_EQ(sel_indices.CheckContiguous(), true);
      sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                      Shape3(0, 2, 1));
    }
    if (req[0] == kNullOp) {
      return;
    } else if (req[0] == kWriteTo) {
      mxnet_op::Kernel<fill_ind_to_one, xpu>::Launch(s, total_size,
                                                     sel_indices.dptr_, ret_mask.dptr_);
    } else {
      LOG(FATAL) << "req=" << req[0] << " is not supported yet.";
    }
  } else if (param.ret_typ == topk_enum::kReturnIndices) {
    if (do_transpose) {
      Tensor<xpu, 3, IDType> ret_indices = ret[0].FlatTo3D<xpu, IDType>(axis, axis, s);
      ASSIGN_DISPATCH(ret_indices, req[0], tcast<IDType>(F<mshadow_op::mod>(transpose(
          slice<2>(inplace_reshape(indices,
                            Shape3(ret_indices.shape_[0], ret_indices.shape_[2], element_num)),
                   0, k), Shape3(0, 2, 1)), element_num)));
    } else {
      Tensor<xpu, 2, IDType> ret_indices =
        ret[0].get_with_shape<xpu, 2, IDType>(Shape2(batch_size, k), s);
      ASSIGN_DISPATCH(ret_indices, req[0], tcast<IDType>(F<mshadow_op::mod>(slice<1>(
                      inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k),
                      element_num)));
    }
  } else {
    if (do_transpose) {
      Tensor<xpu, 3, DType> ret_value = ret[0].FlatTo3D<xpu, DType>(axis, axis, s);
      Tensor<xpu, 3, IDType> ret_indices = ret[1].FlatTo3D<xpu, IDType>(axis, axis, s);
      ASSIGN_DISPATCH(ret_value, req[0], transpose(
          slice<2>(inplace_reshape(sorted_dat,
                            Shape3(ret_value.shape_[0], ret_value.shape_[2], element_num)),
                   0, k), Shape3(0, 2, 1)));
      ASSIGN_DISPATCH(ret_indices, req[1], tcast<IDType>(F<mshadow_op::mod>(transpose(
          slice<2>(inplace_reshape(indices,
                            Shape3(ret_indices.shape_[0], ret_indices.shape_[2], element_num)),
                   0, k), Shape3(0, 2, 1)), element_num)));
    } else {
      Tensor<xpu, 2, DType> ret_value =
        ret[0].get_with_shape<xpu, 2, DType>(Shape2(batch_size, k), s);
      Tensor<xpu, 2, IDType> ret_indices =
        ret[1].get_with_shape<xpu, 2, IDType>(Shape2(batch_size, k), s);
      ASSIGN_DISPATCH(ret_value, req[0],
             slice<1>(inplace_reshape(sorted_dat, Shape2(batch_size, element_num)), 0, k));
      ASSIGN_DISPATCH(ret_indices, req[1], tcast<IDType>(F<mshadow_op::mod>(slice<1>(
                 inplace_reshape(indices, Shape2(batch_size, element_num)), 0, k), element_num)));
    }
  }
}

template<typename xpu>
void TopK_Operation(const TopKParam& param,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const bool flag = param.ret_typ == topk_enum::kReturnIndices
                 || param.ret_typ == topk_enum::kReturnBoth;

  topK_func F;
  size_t size;
  MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    size = GetMemorySize<xpu, DType>(inputs[0]);
    if (flag) {
      MSHADOW_TYPE_SWITCH(param.dtype, IDType, F = TopKImpl<xpu, DType, IDType>;)
    } else {
      F = TopKImpl<xpu, DType, int>;
    }
  });

  (*F)(ctx.run_ctx, ctx.requested[0], req, inputs[0], size, outputs, param);
}

template<typename xpu>
void TopK(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  TopK_Operation<xpu>(nnvm::get<TopKParam>(attrs.parsed), ctx, inputs, req, outputs);
}

template<typename xpu>
void Sort(const nnvm::NodeAttrs& attrs,
          const OpContext& ctx,
          const std::vector<TBlob>& inputs,
          const std::vector<OpReqType>& req,
          const std::vector<TBlob>& outputs) {
  const SortParam& param = nnvm::get<SortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;
  TopK_Operation<xpu>(topk_param, ctx, inputs, req, outputs);
}

template<typename xpu>
void ArgSort(const nnvm::NodeAttrs& attrs,
             const OpContext& ctx,
             const std::vector<TBlob>& inputs,
             const std::vector<OpReqType>& req,
             const std::vector<TBlob>& outputs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.dtype = param.dtype;
  topk_param.ret_typ = topk_enum::kReturnIndices;
  TopK_Operation<xpu>(topk_param, ctx, inputs, req, outputs);
}

template<typename xpu, typename DType, typename IDType>
void TopKBackwardImpl(const OpContext &ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs,
                      const TopKParam& param) {
  CHECK_NE(req[0], kWriteInplace);
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.run_ctx.get_stream<xpu>();
  CHECK(param.ret_typ == topk_enum::kReturnValue || param.ret_typ == topk_enum::kReturnBoth);
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;

  mxnet::TShape target_shape;
  ParseTopKParam(outputs[0].shape_, param,
                 &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  CHECK_LE(element_num, mxnet::common::MaxIntegerValue<IDType>())
    << "'IDType' does not have a sufficient precision to represent the indices of the input array. "
    << "The total element_num is " << element_num << ", but the selected IDType can only represent "
    << mxnet::common::MaxIntegerValue<IDType>() << " elements";
  const auto total_size = batch_size * k;
  Tensor<xpu, 1, int> workspace =
    ctx.requested[0].get_space_typed<xpu, 1, int>(Shape1(total_size + batch_size), s);
  Tensor<xpu, 1, int> sel_indices(workspace.dptr_, Shape1(total_size), s);
  Tensor<xpu, 1, int> batch_shift(workspace.dptr_ + total_size, Shape1(batch_size), s);

  Tensor<xpu, 2, DType> out_grad =
    inputs[0].get_with_shape<xpu, 2, DType>(Shape2(inputs[0].shape_.Size(), 1), s);
  Tensor<xpu, 2, DType> in_grad =
    outputs[0].get_with_shape<xpu, 2, DType>(Shape2(outputs[0].shape_.Size(), 1), s);
  mxnet_op::Kernel<range_fwd, xpu>::Launch(s, batch_size, 1, 0, element_num, kWriteTo,
                                           batch_shift.dptr_);
  if (do_transpose) {
    Tensor<xpu, 1, IDType> indices = inputs[2].FlatTo1D<xpu, IDType>(s);
    mxnet::TShape src_shape = outputs[0].shape_.FlatTo3D(axis);
    sel_indices = reshape(transpose(
                            broadcast_to(inplace_reshape(batch_shift,
                                                         Shape3(src_shape[0], src_shape[2], 1)),
                                         mxnet::TShape(Shape3(src_shape[0], src_shape[2], k))),
                            Shape3(0, 2, 1)),
                          Shape1(total_size));
    sel_indices += tcast<int>(indices);
    sel_indices = transpose_indices(sel_indices, Shape3(src_shape[0], src_shape[2], src_shape[1]),
                                    Shape3(0, 2, 1));
  } else {
    Tensor<xpu, 2, IDType> indices =
      inputs[2].get_with_shape<xpu, 2, IDType>(Shape2(batch_size, k), s);
    sel_indices = reshape(tcast<int>(indices) +
                          broadcast_to(inplace_reshape(batch_shift, Shape2(batch_size, 1)),
                                       mxnet::TShape(Shape2(batch_size, k))),
                          Shape1(total_size));
  }
  CHECK_EQ(sel_indices.CheckContiguous(), true);
  if (kWriteTo == req[0] || kAddTo == req[0]) {
    if (kWriteTo == req[0]) {
      in_grad = scalar<DType>(0);
    }
    mxnet_op::Kernel<fill_ind, xpu>::Launch(s, total_size,
                                            sel_indices.dptr_,
                                            out_grad.dptr_,
                                            req[0],
                                            in_grad.dptr_);
  } else {
    LOG(FATAL) << "Not Implemented!";
  }
}

typedef void (*topKBackward_func)(const OpContext &ctx,
                                  const std::vector<TBlob>& inputs,
                                  const std::vector<OpReqType>& req,
                                  const std::vector<TBlob>& outputs,
                                  const TopKParam& param);
template<typename xpu>
void TopKBackward_(const nnvm::NodeAttrs& attrs,
                   const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  const bool flag = param.ret_typ == topk_enum::kReturnBoth;
  if (flag || param.ret_typ == topk_enum::kReturnValue) {
    topKBackward_func F;
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      if (flag) {
        MSHADOW_TYPE_SWITCH(param.dtype, IDType, F = TopKBackwardImpl<xpu, DType, IDType>;)
      } else {
        F = TopKBackwardImpl<xpu, DType, int>;
      }
    });

    (*F)(ctx, inputs, req, outputs, param);
  } else {
    LOG(FATAL) << "Not Implemented";
  }
}

inline uint32_t TopKNumOutputs(const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnIndices ||
    param.ret_typ == topk_enum::kReturnMask) {
    return static_cast<uint32_t>(1);
  } else {
    return static_cast<uint32_t>(2);
  }
}

inline uint32_t TopKNumVisibleOutputs(const NodeAttrs& attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  if (param.ret_typ == topk_enum::kReturnBoth) {
    return static_cast<uint32_t>(2);
  } else {
    return static_cast<uint32_t>(1);
  }
}

inline bool AssignTypes(std::vector<int> *in_attrs, std::vector<int> *out_attrs) {
  int data_type = -1;
  CHECK(type_assign(&data_type, (*in_attrs)[0])) << "Incompatible dtype of input, in_attrs[0]="
        << (*in_attrs)[0];
  CHECK(type_assign(&data_type, (*out_attrs)[0])) << "Incompatible dtype of output, out_attrs[0]="
        << (*out_attrs)[0];
  CHECK(type_assign(&(*in_attrs)[0], data_type)) << "Incompatible dtype of input, in_attrs[0]="
        << (*in_attrs)[0];
  CHECK(type_assign(&(*out_attrs)[0], data_type)) << "Incompatible dtype of output, out_attrs[0]="
        << (*out_attrs)[0];
  return data_type != -1;
}

inline bool TopKType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);

  const size_t in_size = in_attrs->size();
  const size_t out_size = out_attrs->size();
  CHECK_EQ(in_size, 1);
  CHECK(out_size == 1 || out_size == 2);
  if (out_size > 1) {
    const int retType = (param.ret_typ == topk_enum::kReturnValue? mshadow::kInt32 : param.dtype);
    CHECK(type_assign(&(*out_attrs)[1], retType)) << "Failed to set the type of ret_indices.";
  }
  if (param.ret_typ == topk_enum::kReturnIndices) {
    CHECK(type_assign(&(*out_attrs)[0], param.dtype)) << "Failed to set the type of ret_indices.";
    return true;
  }

  return AssignTypes(in_attrs, out_attrs);
}

inline bool TopKShapeImpl(const TopKParam& param,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  const auto flag = param.ret_typ == topk_enum::kReturnIndices ||
                    param.ret_typ == topk_enum::kReturnMask;
  CHECK_EQ(out_attrs->size(), (flag? 1U : 2U));

  mxnet::TShape& in_shape = (*in_attrs)[0];
  int batch_size, element_num;  // number of batches + the size of each batch
  int axis = 0;
  bool do_transpose = false;
  bool is_ascend = false;
  int k = 0;
  mxnet::TShape target_shape;
  ParseTopKParam(in_shape, param,
    &target_shape, &batch_size, &element_num, &axis, &k, &do_transpose, &is_ascend);
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, target_shape);
  if (!flag) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, target_shape);
  }
  return true;
}

inline bool TopKShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector *in_attrs,
                      mxnet::ShapeVector *out_attrs) {
  const TopKParam& param = nnvm::get<TopKParam>(attrs.parsed);
  return TopKShapeImpl(param, in_attrs, out_attrs);
}

inline bool SortType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1);
  CHECK_EQ(out_attrs->size(), 2);
  CHECK(type_assign(&(*out_attrs)[1], mshadow::kInt32))
          << "Failed to set the type of ret_indices to int32.";
  return AssignTypes(in_attrs, out_attrs);
}

inline bool SortShape(const nnvm::NodeAttrs& attrs,
                      mxnet::ShapeVector *in_attrs,
                      mxnet::ShapeVector *out_attrs) {
  const SortParam& param = nnvm::get<SortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnValue;
  return TopKShapeImpl(topk_param, in_attrs, out_attrs);
}

inline bool ArgSortType(const nnvm::NodeAttrs& attrs,
                        std::vector<int> *in_attrs,
                        std::vector<int> *out_attrs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  CHECK(type_assign(&(*out_attrs)[0], param.dtype))
          << "Failed to set the type of ret_indices";
  return true;
}

inline bool ArgSortShape(const nnvm::NodeAttrs& attrs,
                         mxnet::ShapeVector *in_attrs,
                         mxnet::ShapeVector *out_attrs) {
  const ArgSortParam& param = nnvm::get<ArgSortParam>(attrs.parsed);
  TopKParam topk_param;
  topk_param.axis = param.axis;
  topk_param.is_ascend = param.is_ascend;
  topk_param.k = 0;
  topk_param.ret_typ = topk_enum::kReturnIndices;
  return TopKShapeImpl(topk_param, in_attrs, out_attrs);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ORDERING_OP_INL_H_
