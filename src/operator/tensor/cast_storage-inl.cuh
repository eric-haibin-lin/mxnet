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
 * \file cast_storage-inl.cuh
 * \brief implementation of cast_storage op on GPU
 */
#ifndef MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_
#define MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_

#include <mxnet/base.h>
#include <mxnet/operator.h>
#include <nnvm/tuple.h>

#include <cub/cub.cuh>
#include "cast_storage-inl.h"

namespace mxnet {
namespace op {

/*!
 * \brief Thread kernel for marking non-zero rows of a tensor.
 * Parallelized by tensor rows: 1 thread/row
 */
struct MarkRspRowIdxThreadKernel {
  /*!
   * \brief
   * \param tid         global thread id
   * \param row_flg     row flag array to mark non-zero rows
   * \param dns         dense matrix data
   * \param num_rows    number of rows (size of first dimension of tensor)
   * \param row_length  number of elements per row
   */
  template<typename DType, typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             RType* row_flg,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    if (tid < num_rows) {
      dim_t j = 0;
      dim_t offset = tid * row_length;
      for (; j < row_length; ++j) {
        if (dns[offset+j] != 0) {
          break;
        }
      }
      if (j < row_length) {
        row_flg[tid] = 1;  // mark as one for non-zero row
      } else {
        row_flg[tid] = 0;  // mark as zero for zero row
      }
    }
  }
};

/*!
 * \brief Warp kernel for marking non-zero rows of a tensor.
 * Parallelized by tensor rows: 1 warp/row
 */
struct MarkRspRowIdxWarpKernel {
  template<typename DType, typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             RType* row_flg,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    typedef cub::WarpReduce<dim_t> WarpReduce;
    const dim_t warps_per_block = mshadow::cuda::kBaseThreadNum / 32;
    __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];

    const dim_t warp_id   = tid / 32;          // global warp   id
    const dim_t warp_lane = threadIdx.x / 32;  // local  warp   id within thread block
    const dim_t lane      = tid & (32-1);      // local  thread id within warp

    if (warp_id < num_rows) {
      dim_t flg = 0;
      dim_t offset = warp_id * row_length;
      for (dim_t j = lane; j < row_length; j+=32) {
        if (dns[offset+j] != 0) {
          // avoid break: causes slower performance on sparse tensors (<20% density),
          // due to thread divergence
          flg++;
        }
      }
      dim_t aggr = WarpReduce(temp_storage[warp_lane]).Sum(flg);
      if (lane == 0) {
        if (aggr > 0) {
          row_flg[warp_id] = 1;  // mark as one for non-zero row
        } else {
          row_flg[warp_id] = 0;  // mark as zero for zero row
        }
      }
    }
  }
};

/*!
 * \brief Block kernel for marking non-zero rows of a tensor.
 * Parallelized by tensor rows: 1 threadBlock/row
 */
struct MarkRspRowIdxBlockKernel {
  template<typename DType, typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             RType* row_flg,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    using mshadow::cuda::kBaseThreadNum;
    typedef cub::BlockReduce<dim_t, kBaseThreadNum> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    if (blockIdx.x < num_rows) {
      dim_t flg = 0;
      dim_t offset = blockIdx.x * row_length;
      for (dim_t j = threadIdx.x; j < row_length; j+=kBaseThreadNum) {
        if (dns[offset+j] != 0) {
          // avoid break: causes slower performance on sparse tensors (<20% density),
          // due to thread divergence
          flg++;
        }
      }
      dim_t aggr = BlockReduce(temp_storage).Sum(flg);
      if (threadIdx.x == 0) {
        if (aggr > 0) {
          row_flg[blockIdx.x] = 1;  // mark as one for non-zero row
        } else {
          row_flg[blockIdx.x] = 0;  // mark as zero for zero row
        }
      }
    }
  }
};

/*!
 * \brief Kernel for filling the row index array of the rsp tensor.
 * Parallelized by tensor rows: 1 thread/row
 */
struct FillRspRowIdxKernel {
  /*!
   * \brief
   * \param tid          global thread id
   * \param row_idx      row index array to store indices of non-zero rows
   * \param row_flg_sum  inclusive prefix sum array over marked row flag array
   * \param num_rows     number of rows (size of first dimension of tensor)
   */
  template<typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             RType* row_idx,
                                             const RType* row_flg_sum,
                                             const nnvm::dim_t num_rows) {
    if (tid < num_rows) {
      nnvm::dim_t prev = (tid == 0)? 0 : row_flg_sum[tid-1];
      if (row_flg_sum[tid] > prev) {
        row_idx[prev] = tid;
      }
    }
  }
};

/*!
 * \brief Kernel for filling the value array of the rsp tensor.
 * Parallelized by rsp tensor elements: 1 thread/element
 */
struct FillRspValsKernel {
  /*!
   * \brief
   * \param tid         global thread id
   * \param rsp_val     value array of rsp tensor to store data
   * \param row_idx     indices of non-zero rows
   * \param dns         dense matrix data
   * \param nnr         number of non-zero rows
   * \param row_length  number of elements per row
   */
  template<typename DType, typename RType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* rsp_val,
                                             const RType* row_idx,
                                             const DType* dns,
                                             const nnvm::dim_t nnr,
                                             const nnvm::dim_t row_length) {
    using nnvm::dim_t;
    if (tid < nnr*row_length) {
      const dim_t row_id = tid / row_length;
      const dim_t row_el = tid % row_length;
      const dim_t dns_idx = row_idx[row_id] * row_length + row_el;
      rsp_val[tid] = dns[dns_idx];
    }
  }
};

/*!
 * \brief GPU implementation of casting a dns tensor to rsp type.
 */
inline void CastStorageDnsRspImpl(const OpContext& ctx,
                                  const gpu& gpu_dev,
                                  const TBlob& dns,
                                  NDArray* rsp) {
  CHECK(rsp != nullptr);
  CHECK_EQ(rsp->storage_type(), kRowSparseStorage);
  CHECK_EQ(dns.shape_, rsp->shape());
  using mshadow::Shape1;
  using mxnet_op::Kernel;
  using nnvm::dim_t;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(rsp->aux_type(rowsparse::kIdx), RType, {  // row idx type
      const dim_t num_rows = dns.shape_[0];
      const dim_t row_length = dns.shape_.ProdShape(1, dns.shape_.ndim());
      const dim_t threads_per_warp = mxnet_op::cuda_get_device_prop().warpSize;
      const dim_t threads_per_block = mshadow::cuda::kBaseThreadNum;
      const dim_t min_num_warps = 512;
      dim_t num_threads;
      // TODO: remove kernel dependency on warpSize=32
      if (threads_per_warp != 32) {
        LOG(FATAL) << "CastStorageDnsRspImpl GPU kernels expect warpSize=32";
      }
      // Determine temporary device storage requirements
      RType* row_flg = NULL;
      void* d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    row_flg,
                                    row_flg,
                                    num_rows,
                                    mshadow::Stream<gpu>::GetStream(s));

      // Allocate temp storage for marking non-zero rows and for cub's prefix sum
      auto workspace = AllocateTempDataForSparseHandling<gpu, 1, char>(ctx, Shape1(num_rows*sizeof(RType)
                                                                          + temp_storage_bytes));
      row_flg = reinterpret_cast<RType*>(workspace.dptr_);
      d_temp_storage = workspace.dptr_ + num_rows*sizeof(RType);

      // Mark non-zero rows as 'one' in row_flg
      // Different kernel versions are optimized for different matrix instances
      // (1) 'Thread kernel' (one thread       computing one row)
      // (2) 'Warp kernel'   (one warp         computing one row)
      // (3) 'Block kernel'  (one thread block computing one row)
      const int kernel_version = 0;
      switch (kernel_version) {
        case 1:
          num_threads = num_rows;
          Kernel<MarkRspRowIdxThreadKernel, gpu>::Launch(s, num_threads,
              row_flg, dns.dptr<DType>(), num_rows, row_length);
          break;
        case 2:
          num_threads = num_rows * threads_per_warp;
          Kernel<MarkRspRowIdxWarpKernel, gpu>::Launch(s, num_threads,
              row_flg, dns.dptr<DType>(), num_rows, row_length);
          break;
        case 3:
          num_threads = num_rows * threads_per_block;
          Kernel<MarkRspRowIdxBlockKernel, gpu>::Launch(s, num_threads,
              row_flg, dns.dptr<DType>(), num_rows, row_length);
          break;
        default:
          if (row_length < threads_per_warp) {
            num_threads = num_rows;
            Kernel<MarkRspRowIdxThreadKernel, gpu>::Launch(s, num_threads,
                row_flg, dns.dptr<DType>(), num_rows, row_length);
          } else if (row_length < threads_per_block || num_rows > min_num_warps) {
            num_threads = num_rows * threads_per_warp;
            Kernel<MarkRspRowIdxWarpKernel, gpu>::Launch(s, num_threads,
                row_flg, dns.dptr<DType>(), num_rows, row_length);
          } else {
            num_threads = num_rows * threads_per_block;
            Kernel<MarkRspRowIdxBlockKernel, gpu>::Launch(s, num_threads,
                row_flg, dns.dptr<DType>(), num_rows, row_length);
          }
          break;
      }
      // Compute non-zero row indices through inclusive prefix sum
      cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    row_flg,
                                    row_flg,
                                    num_rows,
                                    mshadow::Stream<gpu>::GetStream(s));

      // Get total number of non-zero rows from device
      RType nnr = 0;
      CUDA_CALL(cudaMemcpy(&nnr, &row_flg[num_rows-1], sizeof(RType), cudaMemcpyDeviceToHost));

      // Allocate rsp tensor row index array and fill
      rsp->CheckAndAllocAuxData(rowsparse::kIdx, Shape1(static_cast<dim_t>(nnr)));
      if (0 == nnr) return;
      RType* row_idx = rsp->aux_data(rowsparse::kIdx).dptr<RType>();
      num_threads = num_rows;
      Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_threads,
          row_idx, row_flg, num_rows);

      // Construct shape of rsp tensor data, allocate, and fill
      auto storage_shape = dns.shape_;
      storage_shape[0] = nnr;
      rsp->CheckAndAllocData(storage_shape);
      num_threads = nnr * row_length;
      Kernel<FillRspValsKernel, gpu>::Launch(s, num_threads,
          rsp->data().dptr<DType>(), row_idx, dns.dptr<DType>(), nnr, row_length);
    });
  });
}

/*!
 * \brief Thread kernel for initializing the indptr in a csr matrix.
 * Parallelized by matrix rows: 1 thread/row
 */
struct FillCsrIndPtrThreadKernel {
  /*!
   * \brief
   * \param tid       global thread id
   * \param indptr    index pointer array of the csr matrix
   * \param dns       dense matrix
   * \param num_rows  number of rows of the dense matrix
   * \param num_cols  number of columns of the dense matrix
   */
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             IType* indptr,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    if (tid == 0) {
      indptr[tid] = 0;
    }
    if (tid < num_rows) {
      dim_t nnz = 0;
      const dim_t offset = tid * num_cols;
      for (dim_t j = 0; j < num_cols; ++j) {
        if (dns[offset+j] != 0) {
          nnz++;
        }
      }
      indptr[tid+1] = nnz;
    }
  }
};

/*!
 * \brief Thread kernel for initializing the col_idx and value array of the csr matrix.
 * Parallelized by matrix rows: 1 thread/row
 */
struct FillCsrColIdxAndValsThreadKernel {
  /*!
   * \brief
   * \param tid       global thread id
   * \param val       data array of the csr matrix
   * \param col_idx   column index array of the csr matrix
   * \param indptr    index pointer array of the csr matrix
   * \param dns       dense matrix
   * \param num_rows  number of rows of the dense matrix
   * \param num_cols  number of columns of the dense matrix
   */
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* val,
                                             CType* col_idx,
                                             const IType* indptr,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    if (tid < num_rows) {
      const dim_t offset = tid * num_cols;
      dim_t k = indptr[tid];
      for (dim_t j = 0; j < num_cols; ++j) {
        if (dns[offset+j] != 0) {
          val[k] = dns[offset+j];
          col_idx[k] = j;
          ++k;
        }
      }
    }
  }
};

/*!
 * \brief Warp kernel for initializing the indptr in a csr matrix.
 * Parallelized by matrix rows: 1 warp/row
 */
struct FillCsrIndPtrWarpKernel {
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             IType* indptr,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    typedef cub::WarpReduce<dim_t> WarpReduce;
    const dim_t warps_per_block = mshadow::cuda::kBaseThreadNum / 32;
    __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];

    if (tid == 0) {
      indptr[tid] = 0;
    }
    const dim_t warp_id   = tid / 32;          // global warp   id
    const dim_t warp_lane = threadIdx.x / 32;  // local  warp   id within thread block
    const dim_t lane      = tid & (32-1);      // local  thread id within warp
    if (warp_id < num_rows) {
      dim_t lane_nnz = 0;
      const dim_t offset = warp_id * num_cols;
      for (dim_t j = lane; j < num_cols; j+=32) {
        if (dns[offset+j] != 0) {
          lane_nnz++;
        }
      }
      dim_t aggr = WarpReduce(temp_storage[warp_lane]).Sum(lane_nnz);
      if (lane == 0) {
        indptr[warp_id+1] = aggr;
      }
    }
  }
};

/*!
 * \brief Warp kernel for initializing the col_idx and value array of the csr matrix.
 * Parallelized by matrix rows: 1 warp/row
 */
struct FillCsrColIdxAndValsWarpKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* val,
                                             CType* col_idx,
                                             const IType* indptr,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using nnvm::dim_t;
    typedef cub::WarpScan<dim_t> WarpScan;
    const dim_t warps_per_block = mshadow::cuda::kBaseThreadNum / 32;
    __shared__ typename WarpScan::TempStorage temp_storage[warps_per_block];
    __shared__ volatile dim_t warp_nnz[warps_per_block];

    const dim_t warp_id   = tid / 32;          // global warp   id
    const dim_t warp_lane = threadIdx.x / 32;  // local  warp   id within thread block
    const dim_t lane      = tid & (32-1);      // local  thread id within warp
    if (warp_id < num_rows) {
      const dim_t offset = warp_id * num_cols;
      dim_t k = indptr[warp_id];
      dim_t nnz;
      for (dim_t j = lane; j < num_cols+lane; j+=32) {
        nnz = 0;
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            nnz++;
          }
        }
        if (lane == 31) {
          warp_nnz[warp_lane] = nnz;
        }
        // Compute index each thread has to write to
        WarpScan(temp_storage[warp_lane]).ExclusiveSum(nnz, nnz);
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            val[k+nnz] = dns[offset+j];
            col_idx[k+nnz] = j;
          }
        }
        if (lane == 31) {
          warp_nnz[warp_lane] += nnz;
        }
        __syncwarp();
        k += warp_nnz[warp_lane];
      }
    }
  }
};

/*!
 * \brief Block kernel for initializing the indptr in a csr matrix.
 * Parallelized by matrix rows: 1 threadBlock/row
 */
struct FillCsrIndPtrBlockKernel {
  template<typename DType, typename IType>
  __device__ __forceinline__ static void Map(int tid,
                                             IType* indptr,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using mshadow::cuda::kBaseThreadNum;
    using nnvm::dim_t;
    typedef cub::BlockReduce<dim_t, kBaseThreadNum> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if (tid == 0) {
      indptr[tid] = 0;
    }
    if (blockIdx.x < num_rows) {
      dim_t lane_nnz = 0;
      const dim_t offset = blockIdx.x * num_cols;
      for (dim_t j = threadIdx.x; j < num_cols; j+=kBaseThreadNum) {
        if (dns[offset+j] != 0) {
          lane_nnz++;
        }
      }
      dim_t aggr = BlockReduce(temp_storage).Sum(lane_nnz);
      if (threadIdx.x == 0) {
        indptr[blockIdx.x+1] = aggr;
      }
    }
  }
};

/*!
 * \brief Block kernel for initializing the col_idx and value array of the csr matrix.
 * Parallelized by matrix rows: 1 threadBlock/row
 */
struct FillCsrColIdxAndValsBlockKernel {
  template<typename DType, typename IType, typename CType>
  __device__ __forceinline__ static void Map(int tid,
                                             DType* val,
                                             CType* col_idx,
                                             const IType* indptr,
                                             const DType* dns,
                                             const nnvm::dim_t num_rows,
                                             const nnvm::dim_t num_cols) {
    using mshadow::cuda::kBaseThreadNum;
    using nnvm::dim_t;
    typedef cub::BlockScan<dim_t, kBaseThreadNum> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ volatile dim_t block_nnz;

    if (blockIdx.x < num_rows) {
      const dim_t offset = blockIdx.x * num_cols;
      dim_t k = indptr[blockIdx.x];
      dim_t nnz;
      for (dim_t j = threadIdx.x; j < num_cols+threadIdx.x; j+=kBaseThreadNum) {
        nnz = 0;
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            nnz++;
          }
        }
        if (threadIdx.x == kBaseThreadNum-1) {
          block_nnz = nnz;
        }
        // Compute index each thread has to write to
        BlockScan(temp_storage).ExclusiveSum(nnz, nnz);
        if (j < num_cols) {
          if (dns[offset+j] != 0) {
            val[k+nnz] = dns[offset+j];
            col_idx[k+nnz] = j;
          }
        }
        if (threadIdx.x == kBaseThreadNum-1) {
          block_nnz += nnz;
        }
        __syncthreads();
        k += block_nnz;
      }
    }
  }
};

/*!
 * \brief GPU implementation of casting a dense matrix to csr type.
 */
inline void CastStorageDnsCsrImpl(const OpContext& ctx,
                                  const gpu& gpu_dev,
                                  const TBlob& dns,
                                  NDArray* csr) {
  CHECK(csr != nullptr);
  CHECK_EQ(csr->storage_type(), kCSRStorage);
  CHECK_EQ(dns.shape_.ndim(), 2);
  CHECK_EQ(dns.shape_, csr->shape());
  using mshadow::Shape1;
  using mxnet_op::Kernel;
  using nnvm::dim_t;
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {                     // data type
    MSHADOW_IDX_TYPE_SWITCH(csr->aux_type(csr::kIndPtr), IType, {  // indptr type
      MSHADOW_IDX_TYPE_SWITCH(csr->aux_type(csr::kIdx), CType, {   // col_idx type
        const dim_t num_rows = dns.shape_[0];
        const dim_t num_cols = dns.shape_[1];
        const dim_t threads_per_warp  = mxnet_op::cuda_get_device_prop().warpSize;
        const dim_t threads_per_block = mshadow::cuda::kBaseThreadNum;
        const dim_t min_num_warps = 512;
        dim_t num_threads;
        // TODO: remove kernel dependency on warpSize=32
        if (threads_per_warp != 32) {
          LOG(FATAL) << "CastStorageDnsCsrImpl GPU kernels expect warpSize=32";
        }
        csr->CheckAndAllocAuxData(csr::kIndPtr, Shape1(num_rows+1));
        IType* indptr = csr->aux_data(csr::kIndPtr).dptr<IType>();
        DType* dns_data = dns.dptr<DType>();

        // Different kernel versions are optimized for different matrix instances
        // (1) 'Thread kernel' (one thread       computing one row)
        // (2) 'Warp kernel'   (one warp         computing one row)
        // (3) 'Block kernel'  (one thread block computing one row)
        const int kernel_version = 0;
        switch (kernel_version) {
          case 1:
            num_threads = num_rows;
            Kernel<FillCsrIndPtrThreadKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            break;
          case 2:
            num_threads = num_rows * threads_per_warp;
            Kernel<FillCsrIndPtrWarpKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            break;
          case 3:
            num_threads = num_rows * threads_per_block;
            Kernel<FillCsrIndPtrBlockKernel, gpu>::Launch(s, num_threads,
                indptr, dns_data, num_rows, num_cols);
            break;
          default:
            if (num_cols < threads_per_warp) {
              num_threads = num_rows;
              Kernel<FillCsrIndPtrThreadKernel, gpu>::Launch(s, num_threads,
                  indptr, dns_data, num_rows, num_cols);
            } else if (num_cols < threads_per_block || num_rows > min_num_warps) {
              num_threads = num_rows * threads_per_warp;
              Kernel<FillCsrIndPtrWarpKernel, gpu>::Launch(s, num_threads,
                  indptr, dns_data, num_rows, num_cols);
            } else {
              num_threads = num_rows * threads_per_block;
              Kernel<FillCsrIndPtrBlockKernel, gpu>::Launch(s, num_threads,
                  indptr, dns_data, num_rows, num_cols);
            }
            break;
        }

        // Determine temporary device storage requirements
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      indptr,
                                      indptr,
                                      num_rows+1,
                                      mshadow::Stream<gpu>::GetStream(s));

        // Allocate temporary storage
        auto workspace = AllocateTempDataForSparseHandling<gpu, 1, char>(ctx, Shape1(temp_storage_bytes));

        d_temp_storage = workspace.dptr_;

        // Compute indptr through inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      indptr,
                                      indptr,
                                      num_rows+1,
                                      mshadow::Stream<gpu>::GetStream(s));

        // Receive total number of nnz values from device
        IType nnz = 0;
        CUDA_CALL(cudaMemcpy(&nnz, &(indptr[num_rows]), sizeof(IType), cudaMemcpyDeviceToHost));

        // Allocate column index array and data array of the csr matrix
        csr->CheckAndAllocAuxData(csr::kIdx, Shape1(static_cast<dim_t>(nnz)));
        csr->CheckAndAllocData(Shape1(static_cast<dim_t>(nnz)));

        // Compute and fill column index array and data array of the csr matrix
        switch (kernel_version) {
          case 1:
            num_threads = num_rows;
            Kernel<FillCsrColIdxAndValsThreadKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            break;
          case 2:
            num_threads = num_rows * threads_per_warp;
            Kernel<FillCsrColIdxAndValsWarpKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            break;
          case 3:
            num_threads = num_rows * threads_per_block;
            Kernel<FillCsrColIdxAndValsBlockKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            break;
          default:
            if (num_cols < threads_per_warp) {
              num_threads = num_rows;
              Kernel<FillCsrColIdxAndValsThreadKernel, gpu>::Launch(s, num_threads,
                  csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                  indptr, dns_data, num_rows, num_cols);
            } else if (num_cols < threads_per_block || num_rows > min_num_warps) {
              num_threads = num_rows * threads_per_warp;
              Kernel<FillCsrColIdxAndValsWarpKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            } else {
              num_threads = num_rows * threads_per_block;
              Kernel<FillCsrColIdxAndValsBlockKernel, gpu>::Launch(s, num_threads,
                csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
                indptr, dns_data, num_rows, num_cols);
            }
            break;
        }
      });
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_TENSOR_CAST_STORAGE_INL_CUH_
