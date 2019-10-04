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
 *  Copyright (c) 2017 by Contributors
 * \file sort_op-inl.cuh
 * \brief CUDA implementations for sort_op.h
 */
#ifndef MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
#define MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
#include <type_traits>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if defined(_MSC_VER) && __CUDACC_VER_MAJOR__ == 8 && __CUDACC_VER_BUILD__ != 44
// Many CUDA 8 compilers other than V8.0.44 crash on Windows
#pragma warning("Potential crash on CUDA compiler detected. Switching sorting from CUB to Thrust")
#define SORT_WITH_THRUST
#else
#include <cub/device/device_radix_sort.cuh>
#undef SORT_WITH_THRUST
#endif
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

namespace mxnet {
namespace op {
namespace cuda {

#define m_half mshadow::half::half_t

template<typename T>
struct less_half
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return static_cast<m_half>(lhs) < static_cast<m_half>(rhs);
  }
};

template<typename T>
struct greater_half
{
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) const {
    return static_cast<m_half>(lhs) > static_cast<m_half>(rhs);
  }
};
}

#ifndef SORT_WITH_THRUST
template <typename KDType, typename VDType>
inline void WorkspaceSize4KeysAndValues(
  const size_t num_keys, size_t *pKeys_bytes, size_t *pValues_bytes) {
  const size_t alignment = std::max(sizeof(KDType), sizeof(VDType));
  *pKeys_bytes = PadBytes(num_keys * sizeof(KDType), alignment);
  *pValues_bytes = PadBytes(num_keys * sizeof(VDType), alignment);
}

template <typename KDType, typename VDType>
inline typename std::enable_if<!std::is_same<KDType, m_half>::value, size_t>::type
SortPairsWorkspaseSize(const size_t num_keys) {
  size_t sortpairs_bytes = 0;
  cub::DeviceRadixSort::SortPairs<KDType, VDType>(NULL, sortpairs_bytes,
    NULL, NULL, NULL, NULL, num_keys);
  return sortpairs_bytes;
}

template <typename KDType, typename VDType>
inline typename std::enable_if<std::is_same<KDType, m_half>::value, size_t>::type
SortPairsWorkspaseSize(const size_t num_keys) {
  size_t sortpairs_bytes = 0;
  cub::DeviceRadixSort::SortPairs<__half, VDType>(NULL, sortpairs_bytes,
    NULL, NULL, NULL, NULL, num_keys);
  return sortpairs_bytes;
}
#endif

template <typename KDType, typename VDType, typename xpu>
inline typename std::enable_if<std::is_same<xpu, gpu>::value, size_t>::type
SortByKeyWorkspaceSize(const size_t num_keys) {
#ifdef SORT_WITH_THRUST
  return 0;
#else
  size_t keys_bytes, values_bytes;
  WorkspaceSize4KeysAndValues<KDType, VDType>(num_keys, &keys_bytes, &values_bytes);
  return keys_bytes + values_bytes + SortPairsWorkspaseSize<KDType, VDType>(num_keys);
#endif
}

template<typename KDType, typename VDType>
inline typename std::enable_if<!(std::is_same<KDType, m_half>::value ||
                                 std::is_same<VDType, m_half>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit) {
  const size_t num_keys = keys.size(0);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
#ifndef SORT_WITH_THRUST
  if (workspace != NULL) {
    // Workspace given, sort using CUB
    CHECK_EQ(workspace->CheckContiguous(), true);
    // workspace = [keys_out, values_out, temporary_storage]
    size_t keys_bytes, values_bytes;
    WorkspaceSize4KeysAndValues<KDType, VDType>(num_keys, &keys_bytes, &values_bytes);
    // Get the size of internal storage (for checking purposes only)
    size_t sortpairs_bytes = 0;
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs<KDType, VDType>(NULL, sortpairs_bytes,
        NULL, NULL, NULL, NULL, num_keys, begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending<KDType, VDType>(NULL, sortpairs_bytes,
        NULL, NULL, NULL, NULL, num_keys, begin_bit, end_bit, stream);
    }

    // Check that we have enough storage
    CHECK_GE(workspace->size(0), keys_bytes + values_bytes + sortpairs_bytes);
    //
    KDType* keys_out_ptr = reinterpret_cast<KDType *>(workspace->dptr_);
    VDType* values_out_ptr = reinterpret_cast<VDType *>(workspace->dptr_ + keys_bytes);
    void* temp_storage = reinterpret_cast<void *>(workspace->dptr_ + keys_bytes + values_bytes);
    // Sort
    if (is_ascend) {
      cub::DeviceRadixSort::SortPairs(temp_storage, sortpairs_bytes,
        keys.dptr_, keys_out_ptr, values.dptr_, values_out_ptr,
        num_keys, begin_bit, end_bit, stream);
    } else {
      cub::DeviceRadixSort::SortPairsDescending(temp_storage, sortpairs_bytes,
        keys.dptr_, keys_out_ptr, values.dptr_, values_out_ptr,
        num_keys, begin_bit, end_bit, stream);
    }
    // Copy result back to [keys, values]
    mshadow::Tensor<gpu, 1, KDType> keys_out(keys_out_ptr, mshadow::Shape1(num_keys),
      keys.stream_);
    mshadow::Tensor<gpu, 1, VDType> values_out(values_out_ptr, mshadow::Shape1(num_keys),
      keys.stream_);
    mshadow::Copy(keys, keys_out, keys.stream_);
    mshadow::Copy(values, values_out, values.stream_);
  } else {
#endif // SORT_WITH_THRUST
    // No workspace, sort using thrust
    const auto key_iter = thrust::device_pointer_cast(keys.dptr_);
    const auto value_iter = thrust::device_pointer_cast(values.dptr_);
    if (is_ascend) {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + num_keys, value_iter, thrust::less<KDType>());
    } else {
      thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        key_iter, key_iter + num_keys, value_iter, thrust::greater<KDType>());
    }
#ifndef SORT_WITH_THRUST
  }
#endif // SORT_WITH_THRUST
}

template<typename KDType, typename VDType>
inline typename std::enable_if<((!std::is_same<KDType, m_half>::value) &&
                                std::is_same<VDType, m_half>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit) {
  using half = ::half;
  const size_t num_keys = keys.size(0);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
  const auto key_iter = thrust::device_pointer_cast(keys.dptr_);
  const auto value_iter = thrust::device_pointer_cast(reinterpret_cast<half*>(values.dptr_));
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter.get(), key_iter.get() + num_keys, value_iter.get(), thrust::less<KDType>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter.get(), key_iter.get() + num_keys, value_iter.get(), thrust::greater<KDType>());
  }
}

template<typename KDType, typename VDType>
inline typename std::enable_if<(std::is_same<KDType, m_half>::value &&
                                (!std::is_same<VDType, m_half>::value)), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit) {
  using half = ::half;
  const size_t num_keys = keys.size(0);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
  const auto key_iter = thrust::device_pointer_cast(reinterpret_cast<half*>(keys.dptr_));
  const auto value_iter = thrust::device_pointer_cast(values.dptr_);
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, value_iter.get(), cuda::less_half<half>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, value_iter.get(), cuda::greater_half<half>());
  }
}

// use thrust sorting when keys or values are half_t
template<typename KDType, typename VDType>
inline typename std::enable_if<(std::is_same<KDType, m_half>::value &&
                                std::is_same<VDType, m_half>::value), void>::type
SortByKeyImpl(mshadow::Tensor<gpu, 1, KDType> keys,
              mshadow::Tensor<gpu, 1, VDType> values, bool is_ascend,
              mshadow::Tensor<gpu, 1, char>* workspace,
              const int begin_bit, const int end_bit) {
  using half = ::half;
  const size_t num_keys = keys.size(0);
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(keys.stream_);
  const auto key_iter = thrust::device_pointer_cast(reinterpret_cast<half*>(keys.dptr_));
  const auto value_iter = thrust::device_pointer_cast(reinterpret_cast<half*>(values.dptr_));
  if (is_ascend) {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, value_iter.get(), cuda::less_half<half>());
  } else {
    thrust::stable_sort_by_key(
      thrust::cuda::par.on(stream),
      key_iter, key_iter + num_keys, value_iter.get(), cuda::greater_half<half>());
  }
}

template<typename KDType, typename VDType>
inline void SortByKey(mshadow::Tensor<gpu, 1, KDType> keys, mshadow::Tensor<gpu, 1, VDType> values,
                      bool is_ascend, mshadow::Tensor<gpu, 1, char>* workspace,
                      const int begin_bit, const int end_bit) {
#if CUDA_VERSION < 9000
#if CUDA_VERSION < 7000
  LOG(FATAL) << "SortByKey is only supported for CUDA version >=7.0!";
#endif
  if (std::is_same<KDType, m_half>::value || std::is_same<VDType, m_half>::value)
    LOG(FATAL) << "SortByKey with fp16 keys and values is only supported for CUDA version >= 9.0";
#endif

  CHECK_EQ(keys.CheckContiguous(), true);
  CHECK_EQ(values.CheckContiguous(), true);
  SortByKeyImpl(keys, values, is_ascend, workspace, begin_bit, end_bit);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SortByKey);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_SORT_OP_INL_CUH_
