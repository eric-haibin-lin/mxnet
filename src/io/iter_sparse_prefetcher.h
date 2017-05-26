/*!
 *  Copyright (c) 2017 by Contributors
 * \file iter_sparse_prefetcher.h
 * \brief define a prefetcher using threaditer to keep k batch fetched
 */
#ifndef MXNET_IO_ITER_SPARSE_PREFETCHER_H_
#define MXNET_IO_ITER_SPARSE_PREFETCHER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <dmlc/logging.h>
#include <dmlc/threadediter.h>
#include <dmlc/optional.h>
#include <mshadow/tensor.h>
#include <climits>
#include <utility>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include "./inst_vector.h"
#include "./image_iter_common.h"
#include "./iter_prefetcher.h"

namespace mxnet {
namespace io {
// iterator on sparse data
class SparsePrefetcherIter : public PrefetcherIter {
 public:
  explicit SparsePrefetcherIter(SparseIIterator<TBlobBatch>* base)
      : PrefetcherIter(base), sparse_loader_(base) {}

  ~SparsePrefetcherIter() {}

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    // TODO refactor this
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init image rec param
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // use the kwarg to init batch loader
    sparse_loader_->Init(kwargs);
    // maximum prefetch threaded iter internal size
    const int kMaxPrefetchBuffer = 16;
    // init thread iter
    iter_.set_max_capacity(kMaxPrefetchBuffer);

    iter_.Init([this](DataBatch **dptr) {
        if (!sparse_loader_->Next()) return false;
        const TBlobBatch& batch = sparse_loader_->Value();
        if (*dptr == nullptr) {
          // allocate databatch
          *dptr = new DataBatch();
          (*dptr)->num_batch_padd = batch.num_batch_padd;
          // (*dptr)->data.at(0) => data
          // (*dptr)->data.at(1) => label
          (*dptr)->data.resize(2);
          (*dptr)->index.resize(batch.batch_size);
          size_t j = 0;
          for (size_t i = 0; i < (*dptr)->data.size(); ++i) {
            bool is_data = i == 0;
            auto stype = this->GetStorageType(is_data);
            auto dtype = param_.dtype ? param_.dtype.value() : batch.data[j].type_flag_;
            if (stype == kDefaultStorage) {
              (*dptr)->data.at(i) = NDArray(batch.data[j].shape_, Context::CPU(), false, dtype);
            } else {
              (*dptr)->data.at(i) = NDArray(stype, this->GetShape(is_data),
                                            Context::CPU(), false, dtype);
            }
            j += NDArray::NumAuxData(stype) + 1;
          }
        }
        // copy data over
        size_t j = 0;
        for (size_t i = 0; i < (*dptr)->data.size(); ++i) {
          auto& nd = ((*dptr)->data)[i];
          auto stype = nd.storage_type();
          auto& data_i = ((*dptr)->data)[i];
          if (stype == kDefaultStorage) {
            CopyFromTo(data_i.data(), batch.data[j]);
          } else if (stype == kCSRStorage){
            auto& values = batch.data[j];
            auto& indices = batch.data[j + 1];
            auto& indptr = batch.data[j + 2];
            // allocate memory
            CHECK_EQ(indices.shape_.Size(),values.shape_.Size());
            nd.CheckAndAllocAuxData(csr::kIdx, indices.shape_);
            nd.CheckAndAllocData(values.shape_);
            nd.CheckAndAllocAuxData(csr::kIndPtr, indptr.shape_);
            // copy values, indices and indptr
            CopyFromTo(data_i.data(), values);
            CopyFromTo(data_i.aux_data(csr::kIdx), indices);
            CopyFromTo(data_i.aux_data(csr::kIndPtr), indptr);
          } else {
            LOG(FATAL) << "Storage type not implemented: " << stype;
          }
          j += NDArray::NumAuxData(stype) + 1;
          (*dptr)->num_batch_padd = batch.num_batch_padd;
        }
        if (batch.inst_index) {
          std::copy(batch.inst_index,
                    batch.inst_index + batch.batch_size,
                    (*dptr)->index.begin());
        }
       return true;
      },
      [this]() { sparse_loader_->BeforeFirst(); });
  }

  virtual void BeforeFirst(void) {
    PrefetcherIter::BeforeFirst();
  }

  virtual bool Next(void) {
    return PrefetcherIter::Next();
  }
  virtual const DataBatch &Value(void) const {
    return PrefetcherIter::Value();
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    return sparse_loader_->GetStorageType(is_data);
  }

  virtual const TShape GetShape(bool is_data) const {
    return sparse_loader_->GetShape(is_data);
  }

 protected:
  /*! \brief internal sparse batch loader */
  SparseIIterator<TBlobBatch>* sparse_loader_;

 private:
  inline void CopyFromTo(TBlob dst, const TBlob src) {
    MSHADOW_TYPE_SWITCH(src.type_flag_, DType, {
      mshadow::Copy(dst.FlatTo1D<cpu, DType>(), src.FlatTo1D<cpu, DType>());
    });
  }

};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_SPARSE_PREFETCHER_H_
