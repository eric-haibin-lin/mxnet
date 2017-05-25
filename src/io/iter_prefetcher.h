/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_prefetcher.h
 * \brief define a prefetcher using threaditer to keep k batch fetched
 */
#ifndef MXNET_IO_ITER_PREFETCHER_H_
#define MXNET_IO_ITER_PREFETCHER_H_

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

namespace mxnet {
namespace io {
// iterator on image recordio
class PrefetcherIter : public IIterator<DataBatch> {
 public:
  // TODO move stype to non-optional
  explicit PrefetcherIter(IIterator<TBlobBatch>* base, NDArrayStorageType stype = kDefaultStorage)
      : loader_(base), out_(nullptr), stype_(stype) {
  }

  ~PrefetcherIter() {
    while (recycle_queue_.size() != 0) {
      DataBatch *batch = recycle_queue_.front();
      recycle_queue_.pop();
      delete batch;
    }
    delete out_;
    iter_.Destroy();
  }

  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init image rec param
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // use the kwarg to init batch loader
    loader_->Init(kwargs);
    // maximum prefetch threaded iter internal size
    const int kMaxPrefetchBuffer = 16;
    // init thread iter
    iter_.set_max_capacity(kMaxPrefetchBuffer);

    iter_.Init([this](DataBatch **dptr) {
        if (!loader_->Next()) return false;
        const TBlobBatch& batch = loader_->Value();
        size_t num_aux = NDArray::num_aux(stype_);
        if (*dptr == nullptr) {
          // allocate databatch
          *dptr = new DataBatch();
          (*dptr)->num_batch_padd = batch.num_batch_padd;
          // assume label is always in dense format, for now
          bool contains_label = batch.data.size() > num_aux;
          // (*dptr)->data.at(0) => data
          // (*dptr)->data.at(1) => label
          (*dptr)->data.resize(contains_label ? 2 : 1);
          (*dptr)->index.resize(batch.batch_size);
          for (size_t i = 0; i < (*dptr)->data.size(); ++i) {
            size_t j = i * num_aux;
            //TODO for labels, also infer batch.data[j].shape_
            auto dtype = param_.dtype ? param_.dtype.value() : batch.data[j].type_flag_;
            if (stype_ == kDefaultStorage || i == 1) {
              (*dptr)->data.at(i) = NDArray(mshadow::Shape2(batch.batch_size, 1),
                                            Context::CPU(), false, 0);
            } else {
              // FIXME the shape is not correct and dtype, too
              (*dptr)->data.at(i) = NDArray(stype_, mshadow::Shape2(batch.batch_size, 10),
                                            Context::CPU(), false, 0);
            }
          }
        }
        // copy data over
        for (size_t i = 0; i < (*dptr)->data.size(); ++i) {
          size_t j = i * num_aux;
          auto &nd = ((*dptr)->data)[i];
          auto stype = nd.storage_type();
          if (stype == kDefaultStorage) {
    //TODO fix idx
            MSHADOW_TYPE_SWITCH(batch.data[2].type_flag_, DType, {
                mshadow::Copy(((*dptr)->data)[i].data().FlatTo1D<cpu, DType>(),
                          batch.data[2].FlatTo1D<cpu, DType>());
            });
          } else if (stype == kCSRStorage){
            size_t values_offset = j;
            size_t indices_offset = j + 1;
            size_t indptr_offset = batch.data.size() - 1;
            // allocate memory
            nd.CheckAndAllocAuxData(csr::kIdx, batch.data[indices_offset].shape_);
            nd.CheckAndAllocData(batch.data[values_offset].shape_);
            CHECK_EQ(batch.data[indices_offset].shape_.Size(), batch.data[values_offset].shape_.Size());
            CHECK_EQ(batch.data.size(), 4);
            // the last one is indptr
            nd.CheckAndAllocAuxData(csr::kIndPtr, batch.data[indptr_offset].shape_);
            // copy idx
            MSHADOW_TYPE_SWITCH(batch.data[indices_offset].type_flag_, DType, {
              mshadow::Copy(((*dptr)->data)[i].aux_data(csr::kIdx).FlatTo1D<cpu, DType>(),
                            batch.data[indices_offset].FlatTo1D<cpu, DType>());
            });
            // copy value
            MSHADOW_TYPE_SWITCH(batch.data[values_offset].type_flag_, DType, {
              mshadow::Copy(((*dptr)->data)[i].data().FlatTo1D<cpu, DType>(),
                            batch.data[values_offset].FlatTo1D<cpu, DType>());
            });
            // copy indptr
            MSHADOW_TYPE_SWITCH(batch.data[indptr_offset].type_flag_, DType, {
              mshadow::Copy(((*dptr)->data)[i].aux_data(csr::kIndPtr).FlatTo1D<cpu, DType>(),
                            batch.data[indptr_offset].FlatTo1D<cpu, DType>());
            });
          } else {
            LOG(FATAL) << "Storage type not implemented: " << stype;
          }
          (*dptr)->num_batch_padd = batch.num_batch_padd;
        }
        if (batch.inst_index) {
          std::copy(batch.inst_index,
                    batch.inst_index + batch.batch_size,
                    (*dptr)->index.begin());
        }
       return true;
      },
      [this]() { loader_->BeforeFirst(); });
  }

  virtual void BeforeFirst(void) {
    iter_.BeforeFirst();
  }

  virtual bool Next(void) {
    if (out_ != nullptr) {
      recycle_queue_.push(out_); out_ = nullptr;
    }
    // do recycle
    if (recycle_queue_.size() == param_.prefetch_buffer) {
      DataBatch *old_batch =  recycle_queue_.front();
      // can be more efficient on engine
      for (NDArray& arr : old_batch->data) {
        arr.WaitToWrite();
      }
      recycle_queue_.pop();
      iter_.Recycle(&old_batch);
    }
    return iter_.Next(&out_);
  }
  virtual const DataBatch &Value(void) const {
    return *out_;
  }

 protected:
  /*! \brief prefetcher parameters */
  PrefetcherParam param_;
  /*! \brief internal batch loader */
  std::unique_ptr<IIterator<TBlobBatch> > loader_;

 private:
  /*! \brief output data */
  DataBatch *out_;
  /*! \brief queue to be recycled */
  std::queue<DataBatch*> recycle_queue_;
  /*! \brief backend thread */
  dmlc::ThreadedIter<DataBatch> iter_;
  /*! \brief storage type of NDArray */
  // TODO add stype for label and data separately
  NDArrayStorageType stype_;
};
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_PREFETCHER_H_
