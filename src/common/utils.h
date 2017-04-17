/*!
 * Copyright (c) 2015 by Contributors
 * \file utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_COMMON_UTILS_H_
#define MXNET_COMMON_UTILS_H_

#if DMLC_USE_CXX11
#include <memory>
#include <vector>
#include <type_traits>
#include <utility>
#include <random>
#include <thread>
#include <algorithm>
#endif  // DMLC_USE_CXX11

#include <dmlc/logging.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <nnvm/graph_attr_types.h>

namespace mxnet {
namespace common {

#if DMLC_USE_CXX11
template <typename xpu>
inline void PrepDefaultBlobs(const std::vector<NDArray>& ndinputs,
                             const std::vector<NDArray>& ndoutputs,
                             std::vector<TBlob> *input_blobs,
                             std::vector<TBlob> *output_blobs,
                             std::vector<NDArray> *tmp_nds,
                             bool alloc_outputs,
                             mshadow::Stream<xpu> *s) {
  for (auto& i : ndinputs) {
    if (i.storage_type() != kDefaultStorage) {
      NDArray tmp_nd = i.ConvertTo<xpu>(kDefaultStorage, s);
      tmp_nds->push_back(tmp_nd);
      input_blobs->push_back(tmp_nd.data());
    } else {
      input_blobs->push_back(i.data());
    }
  }
  for (auto& i : ndoutputs) {
    if (alloc_outputs) i.CheckAndAlloc();
    output_blobs->push_back(i.data());
  }
}

inline void PrepVars(const std::vector<NDArray> &nds,
                     std::vector<Engine::VarHandle> *vars) {
  for (auto& i : nds) {
    auto v = i.var();
    vars->push_back(v);
  }
}

inline NDArrayStorageType GetDispatchStorageType(const nnvm::StorageTypeVector& vstorage_type) {
  NDArrayStorageType dispatch_storage_type = kDefaultStorage;
  for (auto& i : vstorage_type) {
    if (i != kDefaultStorage) {
      // TODO(haibin) the check is not necessary?
      CHECK_NE(i, -1);
      dispatch_storage_type = NDArrayStorageType(i);
      break;
    }
  }
  return dispatch_storage_type;
}

// heuristic to dermine number of threads per GPU
inline int GetNumThreadPerGPU() {
  // This is resource efficient option.
  return dmlc::GetEnv("MXNET_GPU_WORKER_NTHREADS", 2);
}

// heuristic to get number of matching colors.
// this decides how much parallelism we can get in each GPU.
inline int GetExecNumMatchColor() {
  // This is resource efficient option.
  int num_match_color = dmlc::GetEnv("MXNET_EXEC_NUM_TEMP", 1);
  return std::min(num_match_color, GetNumThreadPerGPU());
}

/*!
 * \brief Random Engine
 */
typedef std::mt19937 RANDOM_ENGINE;

/*!
 * \brief Helper functions.
 */
namespace helper {

/*!
 * \brief Helper for non-array type `T`.
 */
template <class T>
struct UniqueIf {
  /*!
   * \brief Type of `T`.
   */
  using SingleObject = std::unique_ptr<T>;
};

/*!
 * \brief Helper for an array of unknown bound `T`.
 */
template <class T>
struct UniqueIf<T[]> {
  /*!
   * \brief Type of `T`.
   */
  using UnknownBound = std::unique_ptr<T[]>;
};

/*!
 * \brief Helper for an array of known bound `T`.
 */
template <class T, size_t kSize>
struct UniqueIf<T[kSize]> {
  /*!
   * \brief Type of `T`.
   */
  using KnownBound = void;
};

}  // namespace helper

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param args List of arguments with which an instance of `T` will be
 *             constructed.
 * \return `std``::``unique_ptr` of an instance of type `T`.
 *
 * Constructs a non-array type `T`. The arguments `args` are passed to the
 * constructor of `T`. The function does not participate in the overload
 * resolution if `T` is an array type.
 */
template <class T, class... Args>
typename helper::UniqueIf<T>::SingleObject MakeUnique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param n The size of the array to construct.
 * \return `std``::``unique_ptr` of an instance of type `T`.
 *
 * Constructs an array of unknown bound `T`. The function does not participate
 * in the overload resolution unless `T` is an array of unknown bound.
 */
template <class T>
typename helper::UniqueIf<T>::UnknownBound MakeUnique(size_t n) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[n]{});
}

/*!
 * \brief Constructs an object of type `T` and wraps it in a
 *        `std``::``unique_ptr`.
 * \param args List of arguments with which an instance of `T` will be
 *             constructed.
 *
 * Constructs an arrays of known bound is disallowed.
 */
template <class T, class... Args>
typename helper::UniqueIf<T>::KnownBound MakeUnique(Args&&... args) = delete;

#endif  // DMLC_USE_CXX11

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_UTILS_H_
