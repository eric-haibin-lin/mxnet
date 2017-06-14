/*!
 * Copyright (c) 2017 by Contributors
 * \file test_utils.h
 * \brief operator unit test utility functions
 * \author Haibin Lin
*/
#ifndef TESTS_CPP_INCLUDE_TEST_NDARRAY_UTILS_H_
#define TESTS_CPP_INCLUDE_TEST_NDARRAY_UTILS_H_

#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <cstdlib>

#include "../src/operator/tensor/elemwise_binary_op.h"
#include "../src/operator/tensor/elemwise_unary_op.h"
#include "../src/operator/optimizer_op-inl.h"
#include "../src/operator/tensor/init_op.h"

#include "test_util.h"
#include "test_op.h"

namespace mxnet
{
namespace test
{

using namespace mxnet;
#define TEST_DTYPE float
#define TEST_ITYPE int32_t

void CheckDataRegion(const TBlob &src, const TBlob &dst) {
  auto size = src.shape_.Size() * mshadow::mshadow_sizeof(src.type_flag_);
  auto equals = memcmp(src.dptr_, dst.dptr_, size);
  EXPECT_EQ(equals, 0);
}

float RandFloat() {
  double v = rand() * 1.0 / RAND_MAX;
  return static_cast<float>(v);
}

// Get an NDArray with provided indices, prepared for a RowSparse NDArray.
NDArray RspIdxND(const TShape shape, const Context ctx, const std::vector<TEST_ITYPE> &values) {
  NDArray nd(shape, ctx, false, ROW_SPARSE_IDX_TYPE);
  size_t num_val = values.size();
  MSHADOW_TYPE_SWITCH(nd.dtype(), DType, {
    auto tensor = nd.data().FlatTo1D<cpu, DType>();
    for (size_t i = 0; i < num_val; i++) {
      tensor[i] = values[i];
    }
  });
  return nd;
}

// Get a dense NDArray with provided values.
NDArray DnsND(const TShape shape, const Context ctx, std::vector<TEST_DTYPE> vs) {
  NDArray nd(shape, ctx, false);
  size_t num_val = shape.Size();
  // generate random values
  while (vs.size() < num_val) {
    auto v = RandFloat();
    vs.push_back(v);
  }
  CHECK_EQ(vs.size(), nd.shape().Size());
  MSHADOW_TYPE_SWITCH(nd.dtype(), DType, {
    auto tensor = nd.data().FlatTo1D<cpu, DType>();
    for (size_t i = 0; i < num_val; i++) {
      tensor[i] = vs[i];
    }
  });
  return nd;
}

// Get a RowSparse NDArray with provided indices and values
NDArray RspND(const TShape shape, const Context ctx, const std::vector<TEST_ITYPE> idx,
              std::vector<TEST_DTYPE> vals) {
  CHECK(shape.ndim() <= 2) << "High dimensional row sparse not implemented yet";
  index_t num_rows = idx.size();
  index_t num_cols = vals.size() / idx.size();
  // create index NDArray
  NDArray index = RspIdxND(mshadow::Shape1(num_rows), ctx, idx);
  CHECK_EQ(vals.size() % idx.size(), 0);
  // create value NDArray
  NDArray data = DnsND(mshadow::Shape2(num_rows, num_cols), ctx, vals);
  // create result nd
  NDArray nd(kRowSparseStorage, shape, ctx, false, mshadow::default_type_flag,
             {}, {mshadow::Shape1(num_rows)});
  // assign values
  NDArray nd_aux = nd.aux_ndarray(0);
  NDArray nd_data = nd.data_ndarray();
  CopyFromTo(index, &nd_aux);
  CopyFromTo(data, &nd_data);
  return nd;
}

// TODO(haibin) support other types
NDArray Convert(NDArrayStorageType type, NDArray src) {
  CHECK_EQ(type, kDefaultStorage);
  NDArray converted(src.shape(), src.ctx(), false);
  Engine::Get()->PushSync([src, converted](RunContext ctx) {
                            // TODO provide type in attrs, which is empty now
                            OpContext op_ctx;
                            op_ctx.run_ctx = ctx;
                            if (src.storage_type() == kRowSparseStorage) {
                              std::vector<NDArray> inputs({src}), outputs({converted});
                              mxnet::op::CastStorageComputeEx<cpu>({}, op_ctx, inputs, {}, outputs);
                            } else if (src.storage_type() == kDefaultStorage) {
                              std::vector<TBlob> inputs({src.data()}), outputs({converted.data()});
                              mxnet::op::UnaryOp::IdentityCompute<cpu>({}, op_ctx, inputs, {kWriteTo}, outputs);
                            } else {
                              LOG(FATAL) << "unsupported storage type";
                            }
                          }, src.ctx(), {src.var()}, {converted.var()},
                          FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  converted.WaitToRead();
  return converted;
}

/*! \brief Array - utility class to construct sparse arrays
 *  \warning This class is not meant to run in a production environment.  Since it is for unit tests only,
 *           simplicity has been chosen over performance.
 **/
template<typename DType>
class Array
{
  typedef std::map<size_t, std::map<size_t, DType> > TItems;
  static constexpr DType EPSILON = 1e-5;

  static const char *st2str(const NDArrayStorageType storageType) {
    switch (storageType) {
      case kDefaultStorage:
        return "kDefaultStorage";
      case kRowSparseStorage:
        return "kRowSparseStorage";
      case kCSRStorage:
        return "kCSRStorage";
      case kUndefinedStorage:
        return "kUndefinedStorage";
      default:
        LOG(FATAL) << "Unsupported storage type: " << storageType;
        return "<INVALID>";
    }
  }

  /*! \brief Remove all zero entries */
  void Prune() {
    for (typename TItems::iterator i = items_.begin(), e = items_.end();
         i != e;) {
      const size_t y = i->first;
      std::map<size_t, DType> &m = i->second;
      ++i;
      for (typename std::map<size_t, DType>::const_iterator j = m.begin(), jn = m.end();
           j != jn;) {
        const size_t x = j->first;
        const DType v = j->second;
        ++j;
        if (IsZero(v)) {
          m.erase(x);
        }
      }
      if (m.empty()) {
        items_.erase(y);
      }
    }
  }

  /*! \brief Create a dense NDArray from our mapped data */
  NDArray CreateDense(const Context& ctx) const {
    NDArray array(shape_, Context::CPU(-1));
    TBlob data = array.data();
    DType *p_data = data.dptr<DType>();
    memset(p_data, 0, array.shape().Size() * sizeof(DType));
    for (typename TItems::const_iterator i = items_.begin(), e = items_.end();
         i != e; ++i) {
      const size_t y = i->first;
      const std::map<size_t, DType> &m = i->second;
      for (typename std::map<size_t, DType>::const_iterator j = m.begin(), jn = m.end();
           j != jn; ++j) {
        const size_t x = j->first;
        const DType v = j->second;
        if (!IsZero(v)) {
          const size_t offset = mxnet::test::offset(shape_, {y, x});
          p_data[offset] = v;
        }
      }
    }
    if(ctx.dev_type == Context::kGPU) {
      NDArray argpu(shape_, ctx);
      CopyFromTo(array, &argpu);
      return argpu;
    } else {
      return array;
    }
  }

 public:

  Array() = default;

  Array(const TShape &shape)
    : shape_(shape) {}

  void clear() {
    items_.clear();
    shape_ = TShape(0);
  }

  static inline bool IsNear(const DType v1, const DType v2) { return fabs(v2 - v1) <= EPSILON; }
  static inline bool IsZero(const DType v) { return IsNear(v, DType(0)); }

  /*! Index into value maps via: [y][x] (row, col) */
  std::map<size_t, DType> &operator[](const size_t idx) { return items_[idx]; }

  const std::map<size_t, DType> &operator[](const size_t idx) const {
    typename TItems::const_iterator i = items_.find(idx);
    if (i != items_.end()) {
      return i->second;
    }
    CHECK(false) << "Attempt to access a non-existent key in a constant map";
    return *static_cast<std::map<size_t, DType> *>(nullptr);
  }

  bool Contains(const size_t row, const size_t col) const {
    typename TItems::const_iterator i = items_.find(row);
    if(i != items_.end()) {
      typename std::map<size_t, DType>::const_iterator j = i->second.find(col);
      if(j != i->second.end()) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Convert from one storage type NDArray to another */
  static NDArray Convert(const Context& ctx, const NDArray& src, const NDArrayStorageType storageType) {
    std::unique_ptr<NDArray> pArray(
      storageType == kDefaultStorage
      ? new NDArray(src.shape(), ctx)
      : new NDArray(storageType, src.shape(), ctx));
    OpContext opContext;
    MXNET_CUDA_ONLY(std::unique_ptr<test::op::GPUStreamScope> gpuScope;);
    switch(ctx.dev_type) {
#if MNXNET_USE_CUDA
      case Context::kGPU:
        gpuScope.reset(new test::op::GPUStreamScope(&opContext));
        mxnet::op::CastStorageComputeImpl<gpu>(s, src, dest);
        break;
#endif  // MNXNET_USE_CUDA
      default:  // CPU
        mxnet::op::CastStorageComputeImpl<cpu>(nullptr, src, *pArray);
        break;
    }
    return *pArray;
  }

  /*! \brief Return NDArray of given storage type representing the value maps */
  NDArray Save(const Context& ctx, const NDArrayStorageType storageType) const {
    switch (storageType) {
      case kDefaultStorage:
        return CreateDense(ctx);
      case kRowSparseStorage:
      case kCSRStorage:
        return Convert(ctx, CreateDense(ctx), storageType);
      case kUndefinedStorage:
      default:
        LOG(ERROR) << "Unsupported storage type: " << storageType;
        return NDArray(TShape(0), ctx);
    }
  }

  void Load(NDArray array) {
    clear();
    shape_ = array.shape();
    if(array.storage_type() != kDefaultStorage) {
      array = Convert(array.ctx(), array, kDefaultStorage);
    }
#if MXNET_USE_CUDA
    if(array.ctx().dev_type == Context::kGPU) {
      NDArray tmp(array.shape(), Context::CPU(-1));
      CopyFromTo(array, &tmp, int priority);
      array = tmp;
    }
#endif  // MXNET_USE_CUDA
    const TBlob blob = array.data();
    DType *p = blob.dptr<DType>();
    CHECK_EQ(shape_.ndim(), 2U);
    for(size_t row = 0, nrow = shape_[0]; row < nrow; ++row) {
      for(size_t col = 0, ncol = shape_[1]; col < ncol; ++col) {
        const size_t off = test::offset(shape_, {row, col});
        if(!IsZero(p[off])) {
          (*this)[row][col] = p[off];
        }
      }
    }
  }

  void print() const {
    for (typename TItems::const_iterator i = items_.begin(), e = items_.end();
         i != e; ++i) {
      const size_t y = i->first;
      const std::map<size_t, DType> &m = i->second;
      CHECK_EQ(m.empty(), false);  // How did it get to have an empty map?
      for (typename std::map<size_t, DType>::const_iterator j = m.begin(), jn = m.end();
           j != jn; ++j) {
        const size_t x = j->first;
        const DType v = j->second;
        if (!IsZero(v)) {
          std::cout << "[row=" << y << ", col=" << x << "]: " << v << std::endl;
        }
      }
    }
    std::cout << std::flush;
  }

 private:
  TShape shape_;
  TItems items_;
};

}  // namespace test
}  // namespace mxnet

#endif  // TESTS_CPP_INCLUDE_TEST_NDARRAY_UTILS_H_
