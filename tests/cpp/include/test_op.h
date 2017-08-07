/*!
 * Copyright (c) 2017 by Contributors
 * \file test_op.h
 * \brief operator unit test utility functions
 * \author Chris Olivier
 *
 * These classes offer a framework for developing, testing and debugging operators
 * in C++.  They work for both CPU and GPU modes, as well as offer a timing
 * infrastructure in order to test inidividual operator performance.
 *
 * Operator data can be validated against general logic,
 * stored scalar values (which can be generated by this code from an existing operator via
 * BasicOperatorData::dumpC(), as well as against each other (ie check that
 * GPU, CPU, MKL, and CUDNN operators produce the same output given the same input.
 *
 * test_util.h: General testing utility functionality
 * test_perf.h: Performance-related classes
 * test_op.h:   Operator-specific testing classes
 */
#ifndef TEST_OP_H_
#define TEST_OP_H_

#include "test_perf.h"
#include "test_util.h"

#include <ndarray/ndarray_function.h>
#include <mshadow/base.h>
#include <mshadow/stream_gpu-inl.h>
#include <atomic>
#include <algorithm>
#include <map>
#include <list>
#include <string>
#include <vector>
#include <utility>

namespace mxnet {
namespace test {
namespace op {

#if MXNET_USE_CUDA
#define MXNET_CUDA_ONLY(__i$) __i$
#else
#define MXNET_CUDA_ONLY(__i$) ((void)0)
#endif

#if MXNET_USE_CUDA
struct GPUStreamScope {
  explicit inline GPUStreamScope(OpContext *opContext)
    : opContext_(*opContext) {
    CHECK_EQ(opContext_.run_ctx.stream == nullptr, true)
      << "Invalid runtime context stream state";
    opContext_.run_ctx.stream = mshadow::NewStream<gpu>(true, true);
    CHECK_EQ(opContext_.run_ctx.stream != nullptr, true)
      << "Unable to allocate a GPU stream";
  }
  inline ~GPUStreamScope() {
    if (opContext_.run_ctx.stream) {
      mshadow::DeleteStream<gpu>(static_cast<mshadow::Stream<gpu> *>(opContext_.run_ctx.stream));
      opContext_.run_ctx.stream = nullptr;
    }
  }
  OpContext& opContext_;
};
#endif  // MXNET_USE_CUDA

/*!
 * \brief Manage test blobs and context, and universal logic
 * Create an operator from its "Prop" class and sets up the operator
 * and resources for both forward and backward passes
 * \tparam DType
 */
template <typename DType, typename AccReal>
class BasicOperatorData {
 public:
  /*! \brief Manage test blobs and context */
  BasicOperatorData(const bool isGPU, const TShape& topShape)
#if !MXNET_USE_CUDA
    : isGPU_(false)
#else
    : isGPU_(isGPU)
#endif
      , initializeForward_(0)   // unit testing may call inits in any order based
      , initializeBackward_(0)  // upon its use-case (ie may not want to run forward pass first)
      , initializeCallback_(0) {
    opContext_.is_train = true;
    opContext_.run_ctx.stream = nullptr;

    shape_input_vec_.push_back(topShape);
  }

  inline mxnet::Context getContext() {
    return isGPU_ ? mxnet::Context::GPU(0) : mxnet::Context{};
  }

  /*! \brief Initialize forward blob data values */
  virtual void resetForward() {}

  /*! \brief Initialize backward blob data values */
  virtual void resetBackward() {}

  /*! \brief Initialize auxiliary and output blobs */
  virtual bool initForward(const OperatorProperty &opProp, std::vector<int> *in_type) {
    if (!initializeForward_++) {
      shape_input_vec_.resize(opProp.ListArguments().size());
      op_.reset(opProp.CreateOperatorEx(getContext(), &shape_input_vec_, in_type));
      if (op_) {
        // Figure out what sort of blobs we need to allocate
        std::vector<TShape> out_shape, aux_shape;
        opProp.InferShape(&shape_input_vec_, &out_shape, &aux_shape);
        std::vector<int> out_type, aux_type;
        opProp.InferType(in_type, &out_type, &aux_type);

        // Allocate top blobs (input)
        for (size_t x = 0, n = shape_input_vec_.size(); x < n; ++x) {
          int type;
          if (x < in_type->size()) {
            type = (*in_type)[x];
          } else {
            type = x ? mshadow::DataType<AccReal>::kFlag : mshadow::DataType<DType>::kFlag;
          }

          allocateBlob(&c_.blob_input_vec_, shape_input_vec_[x], false, type);
        }

        // Allocate aux blobs (scratch, hidden, etc.)
        for (size_t x = 0, n = aux_shape.size(); x < n; ++x) {
          CHECK(x < aux_type.size());
          allocateBlob(&c_.blob_aux_states_, aux_shape[x], false, aux_type[x]);
        }

        // Allocate bottom blobs (output)
        for (size_t x = 0, n = out_shape.size(); x < n; ++x) {
          CHECK(x < out_type.size());
          allocateBlob(&c_.blob_output_vec_, out_shape[x], false, out_type[x]);
        }

        // Get the resource of temporal space
        std::vector<TShape> inputShapes;
        for (size_t x = 0, n = shape_input_vec_.size(); x < n; ++x) {
          inputShapes.push_back(shape_input_vec_[x]);
        }
        allocateResources(opProp.ForwardResource(inputShapes));

        resetForward();
        return true;
      }
      return false;
    } else {
      return true;
    }
  }

  /*! \brief Initialize auxiliary and output blobs */
  virtual bool initBackward(const OperatorProperty &opProp, std::vector<int> *in_type) {
    initForward(opProp, in_type);
    if (!initializeBackward_++) {
      for (size_t x = 0, n = static_cast<size_t>(opProp.NumVisibleOutputs()); x < n; ++x) {
        CHECK_LT(x, c_.blob_input_vec_.size());
        allocateBlob(&c_.blob_out_grad_, c_.blob_input_vec_[x].shape_,
                     false, c_.blob_input_vec_[x].type_flag_);
      }

      for (size_t x = 0, n = c_.blob_input_vec_.size(); x < n; ++x) {
        allocateBlob(&c_.blob_in_grad_,  c_.blob_input_vec_[x].shape_,
                     false, c_.blob_input_vec_[x].type_flag_);
      }

      // Get the resource of temporal space
      std::vector<TShape> ishapes;
      allocateResources(opProp.BackwardResource(ishapes));

      resetBackward();
      return false;
    } else {
      return true;
    }
  }

  /*! \brief Run operator forward */
  void forward(const size_t count = 1) {
    // Possibly move data to/from CPU and GPU (outside of timing scope)
    MXNET_CUDA_ONLY(std::unique_ptr<GPUOpData> gpuData(isGPU_ ?
                       new GPUOpData(c_, &opContext_) : nullptr));
    perf::TimingItem timeF(&timing_, Forward, "Forward", count);
    if (!isGPU_) {
      VTuneResume profile;  // VTune sample only this scope
      for (size_t x = 0; x < count; ++x) {
        op()->Forward(opContext_,
                      c_.blob_input_vec_,
                      {kWriteTo, kWriteTo, kWriteTo},
                      c_.blob_output_vec_,
                      c_.blob_aux_states_);
      }
    } else {
      for (size_t x = 0; x < count; ++x) {
        MXNET_CUDA_ONLY(op()->Forward(opContext_,
                                      gpuData->blob_input_vec_,
                                      {kWriteTo, kWriteTo, kWriteTo},
                                      gpuData->blob_output_vec_,
                                      gpuData->blob_aux_states_));
      }
    }
  }

  /*! \brief Run operator backwards */
  void backward(const size_t count = 1) {
    // Possibly move data to/from CPU and GPU (outside of timing scope)
    MXNET_CUDA_ONLY(std::unique_ptr<GPUOpData> gpuData(isGPU_ ?
                      new GPUOpData(c_, &opContext_) : nullptr));
    perf::TimingItem timeB(&timing_, Backward, "Backward", count);
    if (!isGPU_) {
      VTuneResume profile;  // VTune sample only this scope
      for (size_t x = 0; x < count; ++x) {
        op()->Backward(opContext_,
                       c_.blob_out_grad_,
                       c_.blob_input_vec_,
                       c_.blob_output_vec_,
                       {kWriteTo, kWriteTo, kWriteTo},
                       c_.blob_in_grad_,
                       c_.blob_aux_states_);
      }
    } else {
      for (size_t x = 0; x < count; ++x) {
        MXNET_CUDA_ONLY(op()->Backward(opContext_,
                                       gpuData->blob_out_grad_,
                                       gpuData->blob_input_vec_,
                                       gpuData->blob_output_vec_,
                                       {kWriteTo, kWriteTo, kWriteTo},
                                       gpuData->blob_in_grad_,
                                       gpuData->blob_aux_states_));
      }
    }
  }

  /*! \brief Getter functions for the operator */
  inline Operator *op() { return op_.get(); }
  inline const Operator *op() const { return op_.get(); }

  enum BlobVectorType {
    kInput,
    kOutput,
    kAux,
    kInGrad,
    kOutGrad,
    kBlobVectorTypeCount
  };

  #define CASE_STR(__v$) case (__v$): return #__v$

  /*! \brief Convert BlobVectorType enum into a string */
  static inline const char *bvt2String(const BlobVectorType bvt) {
    switch (bvt) {
      CASE_STR(kInput);
      CASE_STR(kOutput);
      CASE_STR(kAux);
      CASE_STR(kInGrad);
      CASE_STR(kOutGrad);
      default:
      CHECK(false);
      return "";
    }
  }
  #undef CASE_STR

  /*! \brief Return a particular blob in a test data set */
  inline const std::vector<TBlob>& getBlobVect(const BlobVectorType bvt) const {
    switch (bvt) {
      case kInput:
        return c_.blob_input_vec_;
      case kOutput:
        return c_.blob_output_vec_;
      case kAux:
        return c_.blob_aux_states_;
      case kInGrad:
        return c_.blob_in_grad_;
      case kOutGrad:
        return c_.blob_out_grad_;
      default:
        CHECK(false);
        return c_.blob_input_vec_;
    }
  }

  /*! \brief Dump an operator's data set into compilable C++ data code for runtime validation
   * When writing an operator test, you can generate a "known good operator data state" in C++
   * code with this function, and then use load() to load the blob states into this
   * class (BasicOperatorData).
   * After that, you can compare with the "actual" operator state (BasicOperatorData) of
   * the operator that you are testing.
   */
  template<typename Stream>
  inline void dumpC(Stream *_os, const std::string& label) {
    Stream& os = *_os;
    os << "static const std::vector< std::vector< std::vector<float> > > ___"
       << label << "_data_shape_";
    const TShape& shape = shape_input_vec_[0];
    for (size_t i = 0, n = shape.ndim(); i < n; ++i) {
      os << shape[i] << "_";
    }
    os << "__ =" << std::endl << "{" << std::endl;
    for (size_t x = 0; x < kBlobVectorTypeCount; ++x) {
      os << "  { /* " << bvt2String(BlobVectorType(x)) << " */" << std::endl;
      const std::vector<TBlob>& blobVect = getBlobVect(BlobVectorType(x));
      for (size_t i = 0, n = blobVect.size(); i < n; ++i) {
        os << "    { ";
        test::dump<DType>(&os, blobVect[i]);
        os << " }";
        if (i < n - 1) {
          os << ",";
        }
        os << std::endl;
      }
      os << "  }";
      if (x < kBlobVectorTypeCount - 1) {
        os << ",";
      }
      os << std::endl;
    }
    os << "};" << std::endl;
  }

  static inline void copy(const TBlob& blob, const DType array[],
                          const size_t start, const size_t end) {
    const size_t blobSize = blob.Size();
    DType *p = blob.dptr<DType>();
    for (size_t i = 0, n = end - start; i < n; ++i) {
      CHECK_LT(i, blobSize);
      p[i] = array[i + start];
    }
  }

  /*! \brief Runtime load of the C++ data code generated by dumpC() */
  void load(const std::vector<std::vector<std::vector<DType>>>& cData) {
    for (size_t i = 0, ni = cData.size(); i < ni; ++i) {
      for (size_t j = 0, nj = cData[i].size(); j < nj; ++j)  {
        const TBlob& blob = getBlobVect(BlobVectorType(i))[j];
        const size_t sourceDataSize = cData[i][j].size();
        CHECK_EQ(sourceDataSize, blob.Size());
        const DType *sourceData = &cData[i][j][0];
        copy(blob, sourceData, 0, sourceDataSize);
      }
    }
  }

  /*! \brief Runtime load of the C++ data code generated by dumpC() */
  void load(const std::vector<std::vector<std::vector<DType>>>& cData,
            const BlobVectorType type) {
    CHECK_LT(type, cData.size());
    for (size_t j = 0, nj = cData[type].size(); j < nj; ++j)  {
      const TBlob& blob = getBlobVect(type)[j];
      const size_t sourceDataSize = cData[type][j].size();
      CHECK_EQ(sourceDataSize, blob.Size());
      const DType *sourceData = &cData[type][j][0];
      copy(blob, sourceData, 0, sourceDataSize);
    }
  }

  /*! \brief Runtime load of the C++ data code generated by dumpC() */
  void load(const std::vector<std::vector<std::vector<DType>>>& cData,
            const BlobVectorType type, const int idx) {
    CHECK_LT(type, cData.size());
    CHECK_LT(idx, cData[type].size());
    const TBlob& blob = getBlobVect(type)[idx];
    const size_t sourceDataSize = cData[type][idx].size();
    CHECK_EQ(sourceDataSize, blob.Size());
    const DType *sourceData = &cData[type][idx][0];
    copy(blob, sourceData, 0, sourceDataSize);
  }

  /*! \brief Input and output blobs */
  OpContext                 opContext_;

  std::vector<TShape>       shape_input_vec_;

  struct OpData {
    std::vector<TBlob> blob_input_vec_;
    std::vector<TBlob> blob_output_vec_;
    std::vector<TBlob> blob_aux_states_;
    std::vector<TBlob> blob_in_grad_;
    std::vector<TBlob> blob_out_grad_;  // Remaining err (loss) pushing back upstream

    std::vector<std::vector<TBlob> *> all_blob_vects_;
    inline OpData() {
      all_blob_vects_.push_back(&blob_input_vec_);
      all_blob_vects_.push_back(&blob_output_vec_);
      all_blob_vects_.push_back(&blob_aux_states_);
      all_blob_vects_.push_back(&blob_in_grad_);
      all_blob_vects_.push_back(&blob_out_grad_);  // Remaining err (loss) pushing back upstream
    }
    virtual ~OpData() {}
  };

#if MXNET_USE_CUDA
  class GPUOpData : public OpData {
    GPUOpData() = delete;
    GPUOpData(const GPUOpData& o) = delete;

   public:
    inline GPUOpData(const OpData& cpuData, OpContext *opContext)
    : cpuData_(cpuData)
      , allocGPUStream_(opContext) {
      // Copy CPU->GPU
      CHECK_EQ(gpuBlobs_.size(), 0U);
      CHECK_EQ(cpuData_.all_blob_vects_.size(), this->all_blob_vects_.size());
      for (size_t bvt = 0, nbvt = cpuData_.all_blob_vects_.size(); bvt < nbvt; ++bvt) {
        std::vector<TBlob>& bv_src = *cpuData_.all_blob_vects_[bvt];
        std::vector<TBlob>& bvt_dest = *this->all_blob_vects_[bvt];
        for (size_t i = 0, n = bv_src.size(); i < n; ++i) {
          const TBlob& srcBlob = bv_src[i];
          TBlob *destBlob = allocateBlob(&gpuBlobs_, &bvt_dest, srcBlob.shape_,
                                         true, srcBlob.type_flag_);

          Context cpu_ctx, gpu_ctx;
          cpu_ctx.dev_type = Context::kCPU;
          gpu_ctx.dev_type = Context::kGPU;
          cpu_ctx.dev_id = gpu_ctx.dev_id = 0;

          mxnet::ndarray::Copy<cpu, gpu>(srcBlob, destBlob, cpu_ctx,
                                         gpu_ctx, allocGPUStream_.opContext_.run_ctx);
        }
      }
      cudaDeviceSynchronize();
    }
    inline ~GPUOpData() {
      // Copy GPU->CPU
      cudaDeviceSynchronize();
      for (size_t bvt = 0, nbvt = this->all_blob_vects_.size(); bvt < nbvt; ++bvt) {
        std::vector<TBlob>& bv_src = *this->all_blob_vects_[bvt];
        std::vector<TBlob>& bvt_dest = *cpuData_.all_blob_vects_[bvt];
        for (size_t i = 0, n = bv_src.size(); i < n; ++i) {
          const TBlob& srcBlob = bv_src[i];
          TBlob *destBlob = &bvt_dest[i];

          Context cpu_ctx, gpu_ctx;
          cpu_ctx.dev_type = Context::kCPU;
          gpu_ctx.dev_type = Context::kGPU;
          cpu_ctx.dev_id = gpu_ctx.dev_id = 0;

          mxnet::ndarray::Copy<gpu, cpu>(srcBlob, destBlob, gpu_ctx,
                                         cpu_ctx, allocGPUStream_.opContext_.run_ctx);
        }
      }
      gpuBlobs_.clear();  // Force deallocation of the GPU blob data
      cudaDeviceSynchronize();
    }

   private:
    /*! \brief Reference to the src/dest CPU data */
    const OpData& cpuData_;
    /*! \brief The GPU-allocated blobs */
    std::list<std::unique_ptr<test::StandaloneBlob>> gpuBlobs_;
    /*! \brief Scoped GPU stream */
    GPUStreamScope allocGPUStream_;
  };
#endif  // MXNET_USE_CUDA

  OpData                    c_;

 protected:
  /*! \brief Allocate the operator's resource requests */
  void allocateResources(const std::vector<ResourceRequest>& reqs) {
    std::map<Context, Resource> cached_temp;

    Context ctx;
    ctx.dev_type = isGPU_ ? Context::kGPU : Context::kCPU;
    ctx.dev_id = 0;

    for (const ResourceRequest& req : reqs) {
      if (req.type == ResourceRequest::kTempSpace) {
        if (cached_temp.count(ctx) != 0) {
          opContext_.requested.push_back(cached_temp.at(ctx));
        } else {
          Resource r = ResourceManager::Get()->Request(ctx, req);
          opContext_.requested.push_back(r);
          cached_temp[ctx] = r;
        }
      } else if (req.type == ResourceRequest::kRandom) {
        opContext_.requested.push_back(ResourceManager::Get()->Request(ctx, req));
      } else {
        LOG(FATAL) << "resource type not yet supported";
      }
    }
  }

  /*! \brief Locally allocate a managed TBlob and insert into the supplied vector */
  static TBlob *allocateBlob(std::list<std::unique_ptr<test::StandaloneBlob>> *standalone_blobs,
                             std::vector<TBlob> *dest,
                             const TShape& shape,
                             const bool isGPU,
                             const int dtype) {
    test::StandaloneBlob *blob = new test::StandaloneBlob(shape, isGPU, dtype);
    CHECK_NE(blob, static_cast<TBlob *>(nullptr));
    standalone_blobs->push_back(std::unique_ptr<test::StandaloneBlob>(blob));
    (*dest).push_back(*blob);
    return blob;
  }

  /*! \brief Locally allocate a managed TBlob and insert into the supplied vector */
  inline TBlob *allocateBlob(std::vector<TBlob> *dest, const TShape& shape,
                             const bool isGPU, const int dtype) {
    return allocateBlob(&standalone_blobs_, dest, shape, isGPU, dtype);
  }

  /*! \brief Performance timing categories */
  enum TimingId {
    Forward,
    Backward
  };

  /*! \brief The operator */
  std::unique_ptr<Operator>   op_;
  /*! \brief Is this for a GPU? */
  const bool                  isGPU_;
  /*! \brief Assure that the Forward initialized only once */
  std::atomic<int>            initializeForward_;
  /*! \brief Assure that the Forward initialized only once */
  std::atomic<int>            initializeBackward_;
  /*! \brief Assure that the callback is initialized only once */
  std::atomic<int>            initializeCallback_;
  /*! \brief scoped lifecycle management of allocated blobs */
  std::list<std::unique_ptr<test::StandaloneBlob>> standalone_blobs_;

 public:
  /*! Timing instrumentation */
  test::perf::TimingInstrument timing_;
};

/*! \brief Top-level operator test state info structure */
template<typename OperatorProp, typename DType, typename AccReal>
struct OpInfo {
  /*! \brief The operator data */
  std::shared_ptr< test::op::BasicOperatorData<DType, AccReal> > data_;
  /*! \brief The operator prop class */
  std::shared_ptr<OperatorProp>                         prop_;
  /*! \brief The input type(s) */
  std::vector<int>                                      in_type_;
};

/*! \brief Pair of op info objects, generally for validating ops against each other */
template<typename OperatorProp1, typename OperatorProp2, typename DType, typename AccReal>
struct OpInfoPair {
  /*! \brief Operator item 1 */
  test::op::OpInfo<OperatorProp1, DType, AccReal>  info_1_;
  /*! \brief Operator item 2 */
  test::op::OpInfo<OperatorProp2, DType, AccReal>  info_2_;
};

/*! \brief Base validator class for validating test data */
template<typename DType, typename AccReal>
class Validator {
 public:
  static inline DType ERROR_BOUND() {
    switch (sizeof(DType)) {
      case sizeof(mshadow::half::half_t):
        return 0.01f;
      default:
        return 0.001f;
    }
  }

  static inline DType ErrorBound(const TBlob *blob) {
    // Due to eps, for a small number of entries, the error will be a bit higher for one pass
    if (blob->shape_.ndim() >= 3) {
      return (blob->Size() / blob->shape_[1]) <= 4 ? (ERROR_BOUND() * 15) : ERROR_BOUND();
    } else {
      // Probably just a vector
      return ERROR_BOUND();
    }
  }

  /*! \brief Adjusted error based upon significant digits */
  template<typename DTypeX>
  static inline DType ErrorBound(const TBlob *blob, const DTypeX v1, const DTypeX v2) {
    const DType initialErrorBound = ErrorBound(blob);
    DType kErrorBound = initialErrorBound;  // This error is based upon the range [0.1x, 0.9x]
    DTypeX avg = static_cast<DTypeX>((fabs(v1) + fabs(v2)) / 2);
    if (avg >= 1) {
      uint64_t vv = static_cast<uint64_t>(avg + 0.5);
      do {
        kErrorBound *= 10;  // shift error to the right one digit
      } while (vv /= 10);
    }
    return kErrorBound;
  }

  template<typename DTypeX>
  static bool isNear(const DTypeX v1, const DTypeX v2, const AccReal error) {
    return error >= fabs(v2 - v1);
  }

  /*! \brief Convenient setpoint for macro-expanded failures */
  template<typename Type1, typename Type2>
  static void on_failure(const size_t i, const size_t n,
                         const Type1 v1, const Type1 v2, const Type2 kErrorBound) {
    LOG(WARNING)
      << "Near test failure: at i = " << i << ", n = "
      << n << ", kErrorBound = " << kErrorBound << std::endl
      << std::flush;
  }

  /*! \brief Compare blob data */
  static bool compare(const TBlob& b1, const TBlob& b2) {
    if (b1.shape_ == b2.shape_) {
      MSHADOW_REAL_TYPE_SWITCH(b1.type_flag_, DTypeX, {
        CHECK_EQ(b1.type_flag_, b2.type_flag_) << "Can't compare blobs of different data types";
        const DTypeX *d1 = b1.dptr<DTypeX>();
        const DTypeX *d2 = b2.dptr<DTypeX>();
        CHECK_NE(d1, d2);  // don't compare the same memory
        for (size_t i = 0, n = b1.Size(), warningCount = 0; i < n; ++i) {
          const DTypeX v1 = *d1++;
          const DTypeX v2 = *d2++;
          const DType kErrorBound = ErrorBound(&b1, v1, v2);
          EXPECT_NEAR(v1, v2, kErrorBound);
          if (!isNear(v1, v2, kErrorBound) && !warningCount++) {
            on_failure(i, n, v1, v2, kErrorBound);
            return false;
          }
        }
      });
      return true;
    }
    return false;
  }

  /*! \brief Compare blob data to a pointer to data */
  template<typename DTypeX>
  static bool compare(const TBlob& b1, const DTypeX *valuePtr) {
    const DTypeX *d1 = b1.dptr<DType>();
    CHECK_NE(d1, valuePtr);  // don't compare the same memory
    const DType kErrorBound = ErrorBound(&b1);
    for (size_t i = 0, n = b1.Size(), warningCount = 0; i < n; ++i) {
      const DTypeX v1 = *d1++;
      const DTypeX v2 = *valuePtr++;
      EXPECT_NEAR(v1, v2, kErrorBound);
      if (!isNear(v1, v2, kErrorBound) && !warningCount++) {
        LOG(WARNING) << "Near test failure: " << i << ", " << n << std::endl << std::flush;
      }
    }
    return true;
  }

  /*! \brief Compare similar blobs in two operator data structs */
  static bool compare(
    const test::op::BasicOperatorData<DType, AccReal>& i1,
    const test::op::BasicOperatorData<DType, AccReal>& i2,
    const typename test::op::BasicOperatorData<DType, AccReal>::BlobVectorType bvt,
    const size_t idx, bool print = false) {
    const std::vector<TBlob>& bv1 = i1.getBlobVect(bvt);
    const std::vector<TBlob>& bv2 = i2.getBlobVect(bvt);

    // If this is an invalid index, at least make sure the two blob vects
    // are similarly too small for the index
    if (bv1.size() <= idx) {
      CHECK(bv1.size() == bv2.size());
      return true;
    }
    const TBlob& b1 = bv1[idx];
    const TBlob& b2 = bv2[idx];
    if (print && test::debugOutput) {
      test::print(&(std::cout << "Blob 1:"), b1, true, true);
      test::print(&(std::cout << "Blob 2:"), b2, true, true);
    }
    return compare(b1, b2);
  }
};

/*! \brief Operator Prop argument key/value pairs */
typedef std::vector<std::pair<std::string, std::string> > kwargs_t;

/*! \brief Create operator data, prop, the operator itself and init default forward input */
template<typename OperatorProp, typename OperatorData, typename DType, typename AccReal>
static test::op::OpInfo<OperatorProp, DType, AccReal> createOpAndInfoF(const bool isGPU,
                                                              const TShape &inputShape,
                                                              const kwargs_t &kwargs) {
  test::op::OpInfo<OperatorProp, DType, AccReal> info;
  info.data_ = std::make_shared<OperatorData>(isGPU, inputShape);
  info.prop_ = std::make_shared<OperatorProp>();
  info.in_type_ = { mshadow::DataType<DType>::kFlag };
  info.prop_->Init(kwargs);
  info.data_->initForward(*info.prop_, &info.in_type_);
  return info;
}

}  // namespace op
}  // namespace test
}  // namespace mxnet

#endif  // TEST_OP_H_
