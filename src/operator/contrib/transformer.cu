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
 *  Copyright (c) 2018 by Contributors
 * \file transformer.cu
 * \brief GPU implementation of the operators used in Transformer
 * \authors Clement Fuji Tsang
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#include <mxnet/base.h>
#include "./transformer-inl.h"
#include "../../common/cuda_utils.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/wmma_matrix.h"
#ifdef CUTLASS_USE_WMMA_API
#include "cutlass/gemm/wmma_gemm_traits.h"

namespace mxnet {
namespace op {

// gemm_switch_fp32accum and the functions called are almost fully copied from:
// MLPerf v0.6 submission repository from NVIDIA by https://github.com/kevinstephano
template<typename DType>
void CublasStridedBatchedGemm(mshadow::Stream<gpu>* s, bool transA, bool transB,
                              int32_t m, int32_t n, int32_t k,
                              float alpha, const DType* a, int32_t lda, int32_t strideA,
                              const DType *b, int32_t ldb, int32_t strideB, float beta,
                              DType *c, int32_t ldc, int32_t strideC, int32_t batchCount,
                              cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  using namespace mxnet::common::cuda;
  CHECK_EQ(s->blas_handle_ownership_, mshadow::Stream<gpu>::OwnHandle)
      << "Must init CuBLAS handle in stream";

  cublasHandle_t blas_handle = mshadow::Stream<gpu>::GetBlasHandle(s);
  auto err = CUBLAS_STATUS_SUCCESS;
  // TODO(cfujitsang): handle computation_precision
  err = cublasGemmStridedBatchedEx(
      blas_handle, CublasTransposeOp(transA), CublasTransposeOp(transB),
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      reinterpret_cast<void*>(&alpha),
      a, CublasType<DType>::kCudaFlag, static_cast<int>(lda), strideA,
      b, CublasType<DType>::kCudaFlag, static_cast<int>(ldb), strideB,
      reinterpret_cast<void*>(&beta),
      c, CublasType<DType>::kCudaFlag, static_cast<int>(ldc), strideC,
      static_cast<int>(batchCount), CUDA_R_32F, algo);
  CHECK_EQ(err, CUBLAS_STATUS_SUCCESS) << "Cublas gemmEx fail.";
}

template<::cutlass::MatrixLayout::Kind A_LAYOUT,
         ::cutlass::MatrixLayout::Kind B_LAYOUT, int SRC_A, int SRC_B, int DST_C>
void CutlassGemm_FP32Accum(cudaStream_t stream, int32_t m, int32_t n, int32_t k,
                          float alpha, const mshadow::half::half_t *a, int32_t lda,
                          int32_t strideA, const mshadow::half::half_t *b, int32_t ldb,
                          int32_t strideB, float beta, mshadow::half::half_t *c, int32_t ldc,
                          int32_t strideC, int32_t batchCount) {
  typedef cutlass::gemm::WmmaGemmTraits<
    A_LAYOUT,
    B_LAYOUT,
    cutlass::Shape<32, 16, 16>,
    half,
    half,
    half,
    cutlass::gemm::LinearScaling<float>,
    float,
    typename cutlass::gemm::WmmaGemmAccumulatorsPerWarp<
      typename cutlass::Shape<32, 16, 16> >::Shape,
      typename cutlass::Shape<16, 16, 16>,
      SRC_A,   // kScalarsPerLdgA_
      SRC_B,   // kScalarsPerLdgB_
      SRC_A,   // KScalarsPerLdsA_
      SRC_B,   // KScalarsPerLdsB_
      DST_C,   // kScalarsPerLdgCAndStgD_
      DST_C/2,  // kScalarsPerStsD_
      DST_C/2  // kScalarsPerLdsD_
    >
    WmmaGemmTraits;

  typedef cutlass::gemm::Gemm<WmmaGemmTraits> Gemm;
  typename Gemm::Params params;


  int result = params.initialize(
    m,  // M dimension for each batch
    n,  // N dimension for each batch
    k,  // K dimension for each batch
    alpha,  // scalar alpha
    reinterpret_cast<const __half*>(a),
    lda,
    strideA,  // distance in memory between the first element of neighboring batch
    reinterpret_cast<const __half*>(b),
    ldb,
    strideB,  // distance in memory between the first element of neighboring batch
    beta,  // scalar beta
    reinterpret_cast<__half*>(c),  // source matrix C
    ldc,
    strideC,  // distance in memory between the first element of neighboring batch
    reinterpret_cast<__half*>(c),  // destination matrix C (may be different memory than C)
    ldc,
    strideC,  // distance in memory between the first element of neighboring batch
    batchCount);

  CHECK_EQ(result, 0) << "Failed to initialize CUTLASS Gemm::Params object.";

  // Launch the CUTLASS GEMM kernel.
  Gemm::launch(params);
}

void gemm_switch_fp32accum(mshadow::Stream<gpu>* s, bool transA, bool transB,
                           int32_t m, int32_t n, int32_t k,
                           float alpha, const mshadow::half::half_t *a, int32_t lda,
                           int32_t strideA, const mshadow::half::half_t *b, int32_t ldb,
                           int32_t strideB, float beta, mshadow::half::half_t *c, int32_t ldc,
                           int32_t strideC, int32_t batchCount) {
  using cutlass::MatrixLayout::kRowMajor;
  using cutlass::MatrixLayout::kColumnMajor;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  if (transA && (!transB)) {
    if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CublasStridedBatchedGemm(s, transA, transB, m, n, k, alpha, a, lda, strideA, b, ldb,
        strideB, beta, c, ldc, strideC, batchCount, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 8, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 8, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 4, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 8, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kRowMajor, kColumnMajor, 2, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else {
      CublasStridedBatchedGemm(s, transA, transB, m, n, k, alpha, a, lda, strideA, b, ldb,
        strideB, beta, c, ldc, strideC, batchCount);
    }
  } else if ((!transA) && (!transB)) {
    if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CublasStridedBatchedGemm(s, transA, transB, m, n, k, alpha, a, lda, strideA, b, ldb,
        strideB, beta, c, ldc, strideC, batchCount, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 8, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 8, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 4, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 8, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kColumnMajor, 2, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else {
      CublasStridedBatchedGemm(s, transA, transB, m, n, k, alpha, a, lda, strideA, b, ldb,
        strideB, beta, c, ldc, strideC, batchCount);
    }
  } else if ((!transA) && transB) {
    if      (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CublasStridedBatchedGemm(s, transA, transB, m, n, k, alpha, a, lda, strideA, b, ldb,
        strideB, beta, c, ldc, strideC, batchCount, CUBLAS_GEMM_ALGO0_TENSOR_OP);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x7) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 8, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 8, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x3) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 4, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 8, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 8, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x7) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 8, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 4, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 4, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x3) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 4, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x7)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 2, 8>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x3)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 2, 4>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else if (!(lda & 0x1) && !(ldb & 0x1) && !(ldc & 0x1)) {
      CutlassGemm_FP32Accum<kColumnMajor, kRowMajor, 2, 2, 2>(stream, m, n, k, alpha, a, lda,
        strideA, b, ldb, strideB, beta, c, ldc, strideC, batchCount);
    } else {
      CublasStridedBatchedGemm(s, transA, transB, m, n, k, alpha, a, lda, strideA, b, ldb,
                               strideB, beta, c, ldc, strideC, batchCount);
    }
  } else {
    LOG(FATAL) << "transA and transB are invalid";
  }
  CHECK_CUDA_ERROR("Error at InterleavedMatMul");
}

void FakeGrad(const nnvm::NodeAttrs& attrs,
              const OpContext &ctx,
              const std::vector<TBlob> &inputs,
              const std::vector<OpReqType> &req,
              const std::vector<TBlob> &outputs) {
  LOG(FATAL) << "FakeGrad";
}
// TODO(cfujitsang): use scaled as optional ?
// TODO(cfujitsang): set alpha / beta if kAddTo (look at FullyConnected)
void InterleavedMatMulSelfAttQKGPU(const nnvm::NodeAttrs& attrs,
                                   const OpContext &ctx,
                                   const std::vector<TBlob> &inputs,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    typedef mshadow::half::half_t DType;
    const DType* queries_keys_values = inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    DType* output = outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const int32_t qkv_seq_len    = inputs[0].shape_[0];
    const int32_t sequences      = inputs[0].shape_[1];
    const int32_t output_lin_dim = inputs[0].shape_[2];
    const int32_t embed_dim      = output_lin_dim / 3;
    const int32_t head_dim       = embed_dim / params.heads;
    const int32_t attn_batches   = params.heads * sequences;
    const int32_t lead_dim       = attn_batches * 3 * head_dim;
    const int32_t batch_stride   = 3 * head_dim;
    if (req[0] == kNullOp)
      return;
    if (req[0] == kWriteTo) {
      cudaMemsetAsync(output, 0, outputs[0].shape_.Size() * sizeof(DType),
                      mshadow::Stream<gpu>::GetStream(s));
    }
    const float beta          = req[0] == kAddTo ? 1.f : 0.f;
    const float scale         = 1.0 / sqrt(static_cast<float>(head_dim));
    gemm_switch_fp32accum(s,
                          true,
                          false,
                          qkv_seq_len,
                          qkv_seq_len,
                          head_dim,
                          scale,
                          queries_keys_values + head_dim,
                          lead_dim,
                          batch_stride,
                          queries_keys_values,
                          lead_dim,
                          batch_stride,
                          beta,
                          output,
                          qkv_seq_len,
                          qkv_seq_len * qkv_seq_len,
                          attn_batches);
  } else {
    LOG(INFO) << "Not implemented with this type";
  }
}

void BackwardInterleavedMatMulSelfAttQKGPU(const nnvm::NodeAttrs& attrs,
                                           const OpContext &ctx,
                                           const std::vector<TBlob> &inputs,
                                           const std::vector<OpReqType> &req,
                                           const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    typedef mshadow::half::half_t DType;
    const DType* output_grads        = inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* queries_keys_values = inputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    DType* queries_keys_values_grads = outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const int32_t qkv_seq_len    = inputs[1].shape_[0];
    const int32_t sequences      = inputs[1].shape_[1];
    const int32_t output_lin_dim = inputs[1].shape_[2];
    const int32_t embed_dim      = output_lin_dim / 3;
    const int32_t head_dim       = embed_dim / params.heads;
    const int32_t attn_batches   = params.heads * sequences;
    const int32_t lead_dim       = attn_batches * 3 * head_dim;
    const int32_t batch_stride   = 3 * head_dim;
    if (req[0] == kNullOp)
      return;
    const float beta             = req[0] == kAddTo ? 1.f : 0.f;
    if (req[0] == kWriteTo) {
      cudaMemsetAsync(queries_keys_values_grads, 0, outputs[0].shape_.Size() * sizeof(DType),
                      mshadow::Stream<gpu>::GetStream(s));
    }
    const float scale            = 1.0 / sqrt(static_cast<float>(head_dim));
    gemm_switch_fp32accum(s,
                          false,
                          false,
                          head_dim,
                          qkv_seq_len,
                          qkv_seq_len,
                          scale,
                          queries_keys_values + head_dim,
                          lead_dim,
                          batch_stride,
                          output_grads,
                          qkv_seq_len,
                          qkv_seq_len * qkv_seq_len,
                          beta,
                          queries_keys_values_grads,
                          lead_dim,
                          batch_stride,
                          attn_batches);
    gemm_switch_fp32accum(s,
                          false,
                          true,
                          head_dim,
                          qkv_seq_len,
                          qkv_seq_len,
                          scale,
                          queries_keys_values,
                          lead_dim,
                          batch_stride,
                          output_grads,
                          qkv_seq_len,
                          qkv_seq_len * qkv_seq_len,
                          beta,
                          queries_keys_values_grads + head_dim,
                          lead_dim,
                          batch_stride,
                          attn_batches);
  } else {
    LOG(INFO) << "Not implemented with this type";
  }
}

void InterleavedMatMulSelfAttValAttGPU(const nnvm::NodeAttrs& attrs,
                                       const OpContext &ctx,
                                       const std::vector<TBlob> &inputs,
                                       const std::vector<OpReqType> &req,
                                       const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    typedef mshadow::half::half_t DType;
    const DType* queries_keys_values = inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* attention_maps      = inputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    DType* output                    = outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const int32_t qkv_seq_len    = inputs[0].shape_[0];
    const int32_t sequences      = inputs[0].shape_[1];
    const int32_t output_lin_dim = inputs[0].shape_[2];
    const int32_t embed_dim      = output_lin_dim / 3;
    const int32_t head_dim       = embed_dim / params.heads;
    const int32_t attn_batches   = params.heads * sequences;
    const int32_t lead_dim       = attn_batches * 3 * head_dim;
    const int32_t batch_stride   = 3 * head_dim;
    const float alpha            = 1.f;
    if (req[0] == kNullOp)
      return;
    if (req[0] == kWriteTo) {
      cudaMemsetAsync(output, 0, outputs[0].shape_.Size() * sizeof(DType),
                      mshadow::Stream<gpu>::GetStream(s));
    }
    const float beta             = req[0] == kAddTo ? 1.f : 0.f;
    gemm_switch_fp32accum(s,
                          false,
                          false,
                          head_dim,
                          qkv_seq_len,
                          qkv_seq_len,
                          alpha,
                          queries_keys_values + 2 * head_dim,
                          lead_dim,
                          batch_stride,
                          attention_maps,
                          qkv_seq_len,
                          qkv_seq_len * qkv_seq_len,
                          beta,
                          output,
                          head_dim * attn_batches,
                          head_dim,
                          attn_batches);
  } else {
    LOG(INFO) << "Not implemented with this type";
  }
}

void BackwardInterleavedMatMulSelfAttValAttGPU(const nnvm::NodeAttrs& attrs,
                                               const OpContext &ctx,
                                               const std::vector<TBlob> &inputs,
                                               const std::vector<OpReqType> &req,
                                               const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    typedef mshadow::half::half_t DType;
    const DType* output_grads              = inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* queries_keys_values       = inputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* attention_maps            = inputs[2].FlatTo2D<gpu, DType>(s).dptr_;
    DType* queries_keys_values_grads       = outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    DType* attention_maps_grads            = outputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    const int32_t qkv_seq_len    = inputs[1].shape_[0];
    const int32_t sequences      = inputs[1].shape_[1];
    const int32_t output_lin_dim = inputs[1].shape_[2];
    const int32_t embed_dim      = output_lin_dim / 3;
    const int32_t head_dim       = embed_dim / params.heads;
    const int32_t attn_batches   = params.heads * sequences;
    const int32_t lead_dim       = attn_batches * 3 * head_dim;
    const int32_t batch_stride   = 3 * head_dim;
    const float alpha            = 1.f;
    if (req[0] != kNullOp) {
      if (req[0] == kWriteTo) {
        cudaMemsetAsync(queries_keys_values_grads, 0, outputs[0].shape_.Size() * sizeof(DType),
                        mshadow::Stream<gpu>::GetStream(s));
      }

      const float beta = req[0] == kAddTo ? 1.f : 0.f;
      gemm_switch_fp32accum(s,
                            false,
                            true,
                            head_dim,
                            qkv_seq_len,
                            qkv_seq_len,
                            alpha,
                            output_grads,
                            head_dim * attn_batches,
                            head_dim,
                            attention_maps,
                            qkv_seq_len,
                            qkv_seq_len * qkv_seq_len,
                            beta,
                            queries_keys_values_grads + 2 * head_dim,
                            lead_dim,
                            batch_stride,
                            attn_batches);
    }
    if (req[1] != kNullOp) {
      if (req[1] == kWriteTo) {
        cudaMemsetAsync(attention_maps_grads, 0, outputs[1].shape_.Size() * sizeof(DType),
                        mshadow::Stream<gpu>::GetStream(s));
      }

      const float beta = req[1] == kAddTo ? 1.f : 0.f;
      gemm_switch_fp32accum(s,
                            true,
                            false,
                            qkv_seq_len,
                            qkv_seq_len,
                            head_dim,
                            alpha,
                            queries_keys_values + 2 * head_dim,
                            lead_dim,
                            batch_stride,
                            output_grads,
                            head_dim * attn_batches,
                            head_dim,
                            beta,
                            attention_maps_grads,
                            qkv_seq_len,
                            qkv_seq_len * qkv_seq_len,
                            attn_batches);
    }
  } else {
    LOG(INFO) << "Not implemented with this type";
  }
}


void InterleavedMatMulEncDecQKGPU(const nnvm::NodeAttrs& attrs,
                                  const OpContext &ctx,
                                  const std::vector<TBlob> &inputs,
                                  const std::vector<OpReqType> &req,
                                  const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    typedef mshadow::half::half_t DType;
    const DType* queries     = inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* keys_values = inputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    DType* output            = outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const int32_t q_seq_len         = inputs[0].shape_[0];
    const int32_t sequences         = inputs[0].shape_[1];
    const int32_t output_lin_q_dim  = inputs[0].shape_[2];
    const int32_t kv_seq_len        = inputs[1].shape_[0];
    const int32_t output_lin_kv_dim = inputs[1].shape_[2];
    const int32_t embed_dim         = output_lin_q_dim;
    const int32_t head_dim          = embed_dim / params.heads;
    const int32_t attn_batches      = params.heads * sequences;
    const int32_t lead_dim_q        = attn_batches * head_dim;
    const int32_t lead_dim_kv       = attn_batches * 2 * head_dim;
    const int32_t batch_stride_q    = head_dim;
    const int32_t batch_stride_kv   = head_dim * 2;
    if (req[0] == kNullOp)
      return;
    const float beta                = req[0] == kAddTo ? 1.f : 0.f;
    const float scale               = 1.f / sqrt(static_cast<float>(head_dim));
    gemm_switch_fp32accum(s,
                          true,
                          false,
                          kv_seq_len,
                          q_seq_len,
                          head_dim,
                          scale,
                          keys_values,
                          lead_dim_kv,
                          batch_stride_kv,
                          queries,
                          lead_dim_q,
                          batch_stride_q,
                          beta,
                          output,
                          kv_seq_len,
                          kv_seq_len * q_seq_len,
                          attn_batches);
  } else {
    LOG(INFO) << "Not implemented with this type";
  }
}

void InterleavedMatMulEncDecValAttGPU(const nnvm::NodeAttrs& attrs,
                                      const OpContext &ctx,
                                      const std::vector<TBlob> &inputs,
                                      const std::vector<OpReqType> &req,
                                      const std::vector<TBlob> &outputs) {
  const auto& params = nnvm::get<InterleavedMatMulParam>(attrs.parsed);
  mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
  if (inputs[0].type_flag_ == mshadow::kFloat16) {
    typedef mshadow::half::half_t DType;
    const DType* keys_values    = inputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const DType* attention_maps = inputs[1].FlatTo2D<gpu, DType>(s).dptr_;
    DType* output               = outputs[0].FlatTo2D<gpu, DType>(s).dptr_;
    const int32_t kv_seq_len        = inputs[0].shape_[0];
    const int32_t sequences         = inputs[0].shape_[1];
    const int32_t output_lin_kv_dim = inputs[0].shape_[2];
    const int32_t attn_batches      = inputs[1].shape_[0];
    const int32_t q_seq_len         = inputs[1].shape_[1];
    const int32_t embed_dim         = output_lin_kv_dim / 2;
    int32_t head_dim          = embed_dim / params.heads;
    const int32_t lead_dim_kv       = attn_batches * head_dim * 2;
    const int32_t batch_stride_kv   = 2 * head_dim;
    const float alpha      = 1.f;
    const float beta       = 0.f;
    gemm_switch_fp32accum(s,
                          false,
                          false,
                          head_dim,
                          q_seq_len,
                          kv_seq_len,
                          alpha,
                          keys_values + head_dim,
                          lead_dim_kv,
                          batch_stride_kv,
                          attention_maps,
                          kv_seq_len,
                          kv_seq_len * q_seq_len,
                          beta,
                          output,
                          head_dim * attn_batches,
                          head_dim,
                          attn_batches);
  } else {
    LOG(INFO) << "Not implemented with this type";
  }
}


NNVM_REGISTER_OP(interleaved_matmul_selfatt_qk)
.set_attr<FCompute>("FCompute<gpu>", InterleavedMatMulSelfAttQKGPU);

NNVM_REGISTER_OP(interleaved_matmul_selfatt_valatt)
.set_attr<FCompute>("FCompute<gpu>", InterleavedMatMulSelfAttValAttGPU);

NNVM_REGISTER_OP(interleaved_matmul_encdec_qk)
.set_attr<FCompute>("FCompute<gpu>", InterleavedMatMulEncDecQKGPU);

NNVM_REGISTER_OP(interleaved_matmul_encdec_valatt)
.set_attr<FCompute>("FCompute<gpu>", InterleavedMatMulEncDecValAttGPU);

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_qk)
.set_attr<FCompute>("FCompute<gpu>", BackwardInterleavedMatMulSelfAttQKGPU);

NNVM_REGISTER_OP(_backward_interleaved_matmul_selfatt_valatt)
.set_attr<FCompute>("FCompute<gpu>", BackwardInterleavedMatMulSelfAttValAttGPU);

NNVM_REGISTER_OP(fake_grad1)
.set_attr<FCompute>("FCompute<gpu>", FakeGrad);

NNVM_REGISTER_OP(fake_grad2)
.set_attr<FCompute>("FCompute<gpu>", FakeGrad);

// relu
NNVM_REGISTER_OP(_contrib_div_sqrt_dim)
.set_attr<FCompute>("FCompute<gpu>", DivSqrtDimForward_<gpu>);

}  // namespace op
}  // namespace mxnet

#endif  // defined CUTLASS_USE_WMMA_API
