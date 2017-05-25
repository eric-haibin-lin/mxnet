/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_batchloader.h
 * \brief define a batch adapter to create tblob batch
 */
#ifndef MXNET_IO_ITER_BATCHLOADER_H_
#define MXNET_IO_ITER_BATCHLOADER_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <dmlc/logging.h>
#include <mshadow/tensor.h>
#include <utility>
#include <vector>
#include <string>
#include "./inst_vector.h"
#include "./image_iter_common.h"

namespace mxnet {
namespace io {

/*! \brief create a batch iterator from single instance iterator */
class BatchLoader : public IIterator<TBlobBatch> {
 public:
  explicit BatchLoader(IIterator<DataInst> *base):
      base_(base), head_(1), num_overflow_(0) {
  }

  virtual ~BatchLoader(void) {
    delete base_;
  }

  inline void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    std::vector<std::pair<std::string, std::string> > kwargs_left;
    // init batch param, it could have similar param with
    kwargs_left = param_.InitAllowUnknown(kwargs);
    // Init space for out_
    out_.inst_index = new unsigned[param_.batch_size];
    out_.batch_size = param_.batch_size;
    out_.data.clear();
    // init base iterator
    base_->Init(kwargs);
    data_stype_ = base_->GetDataStorageType();
    label_stype_ = base_->GetLabelStorageType();
    if (param_.round_batch == 0) {
      LOG(FATAL) << "CSR batch loader doesn't support round_batch == false";
    }
  }

  virtual void BeforeFirst(void) {
    if (param_.round_batch == 0 || num_overflow_ == 0) {
      // otherise, we already called before first
      base_->BeforeFirst();
    } else {
      num_overflow_ = 0;
    }
    head_ = 1;
  }

  virtual bool Next(void) {
    if (data_stype_ == kCSRStorage) {
       return NextCSRBatch();
    } else {
      return NextDefaultBatch();
    }
  }

  virtual const TBlobBatch &Value(void) const {
    return out_;
  }

 private:
  /*! \brief batch parameters */
  BatchParam param_;
  /*! \brief output data */
  TBlobBatch out_;
  /*! \brief base iterator */
  IIterator<DataInst> *base_;
  /*! \brief on first */
  int head_;
  /*! \brief number of overflow instances that readed in round_batch mode */
  int num_overflow_;
  /*! \brief data shape */
  std::vector<TShape> shape_;
  /*! \brief unit size */
  std::vector<size_t> unit_size_;
  std::vector<size_t> offsets_;
  std::vector<DataInst> inst_cache_;

  /*! \brief tensor to hold data */
  std::vector<TBlobContainer> data_;
  NDArrayStorageType data_stype_;
  NDArrayStorageType label_stype_;

  bool NextCSRBatch(void) {
    out_.num_batch_padd = 0;
    out_.batch_size = param_.batch_size;
    this->head_ = 0;
    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;
    inst_cache_.clear();
    while (base_->Next()) {
      inst_cache_.emplace_back(base_->Value());
      if (inst_cache_.size() >= param_.batch_size) break;
    }
    // no more data instance
    if (inst_cache_.size() == 0) {
      return false;
    }
    if (inst_cache_.size() < param_.batch_size) {
      CHECK_GT(param_.round_batch, 0);
      num_overflow_ = 0;
      base_->BeforeFirst();
      for (; inst_cache_.size() < param_.batch_size; ++num_overflow_) {
        CHECK(base_->Next()) << "number of input must be bigger than batch size";
        inst_cache_.emplace_back(base_->Value());
      }
    }
    out_.num_batch_padd = num_overflow_;
    CHECK_EQ(inst_cache_.size(), param_.batch_size);
    this->InitDataFromBatch();
    int32_t acc = 0;
    using IType = int32_t;
    data_[data_.size() - 1].get<cpu, 1, IType>()[0] = acc;
    for (size_t j = 0; j < inst_cache_.size(); j++) {
      const auto& d = inst_cache_[j];
      out_.inst_index[top] = d.index;
      size_t unit_size = 0;
      for (size_t i = 0; i < d.data.size(); ++i) {
        unit_size = d.data[i].shape_.Size();
        MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
          mshadow::Copy(data_[i].get<cpu, 1, DType>().Slice(offsets_[i], offsets_[i] + unit_size),
                        d.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size)));
          });
        offsets_[i] += unit_size;
        CHECK_NE(unit_size, 0);
        // Update the indptr tensor, only for indices and values
        // FIXME hard code index
        if (i == 0) {
          acc += (int32_t) unit_size;
          data_[d.data.size()].get<cpu, 1, IType>()[j + 1] = acc;
        }
      }
    }
    return true;
  }

  // TODO compare with original code
  bool NextDefaultBatch(void) {
    out_.num_batch_padd = 0;
    out_.batch_size = param_.batch_size;
    this->head_ = 0;

    // if overflow from previous round, directly return false, until before first is called
    if (num_overflow_ != 0) return false;
    index_t top = 0;

    while (base_->Next()) {
      const DataInst& d = base_->Value();
      out_.inst_index[top] = d.index;
      if (data_.size() == 0) {
        this->InitDataFromInst(d);
      }
      for (size_t i = 0; i < d.data.size(); ++i) {
        // TODO remove unit_size_[i]
        CHECK_EQ(unit_size_[i], d.data[i].Size());
        MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
            mshadow::Copy(
              data_[i].get<cpu, 1, DType>().Slice(top * unit_size_[i],
                                                  (top + 1) * unit_size_[i]),
              d.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size_[i])));
          });
      }
      if (++top >= param_.batch_size) {
        return true;
      }
    }
    if (top != 0) {
      if (param_.round_batch != 0) {
        num_overflow_ = 0;
        base_->BeforeFirst();
        for (; top < param_.batch_size; ++top, ++num_overflow_) {
          CHECK(base_->Next()) << "number of input must be bigger than batch size";
          const DataInst& d = base_->Value();
          out_.inst_index[top] = d.index;
          // copy data
          for (size_t i = 0; i < d.data.size(); ++i) {
            CHECK_EQ(unit_size_[i], d.data[i].Size());
            MSHADOW_TYPE_SWITCH(data_[i].type_flag_, DType, {
                mshadow::Copy(
                  data_[i].get<cpu, 1, DType>().Slice(top * unit_size_[i],
                                                      (top + 1) * unit_size_[i]),
                  d.data[i].get_with_shape<cpu, 1, DType>(mshadow::Shape1(unit_size_[i])));
              });
          }
        }
        out_.num_batch_padd = num_overflow_;
      } else {
        out_.num_batch_padd = param_.batch_size - top;
      }
      return true;
    }
    return false;
  }


  // initialize the data holder by using from the batch
  inline void InitDataFromBatch() {
    // TODO also init indptr for label, if any
    LOG(INFO) << "init data ";
    CHECK(inst_cache_.size() > 0);
    size_t num_data = inst_cache_[0].data.size();
    out_.data.clear();
    //const auto& first_inst = inst_cache_[0];
    //shape_.resize(first_inst.data.size());
    //data_.clear();
    data_.resize(num_data + 1);
    CHECK(data_[num_data].dev_mask_ == 1);
    LOG(INFO) << "Num_data: " << num_data;
    offsets_.clear();
    offsets_.resize(num_data + 1, 0);
    //unit_size_.resize(first_inst.data.size());
    std::vector<size_t> sizes(num_data + 1, 0);
    // accumulate the memory space required for a batch
    for (size_t i = 0; i < num_data; ++i) {
      size_t size = 0;
      for (const auto &d : inst_cache_) size += d.data[i].shape_.Size();
      sizes[i] = size;
    }

    for (size_t i = 0; i < num_data; ++i) {
      int src_type_flag = inst_cache_[0].data[i].type_flag_;
      // init object attributes
      TShape dst_shape(mshadow::Shape1(sizes[i]));
      data_[i].resize(mshadow::Shape1(sizes[i]), src_type_flag);
      //unit_size_[i] = src_shape.Size();
      CHECK(data_[i].dptr_ != nullptr);
      out_.data.push_back(TBlob(data_[i].dptr_, dst_shape, cpu::kDevMask, src_type_flag));
    }

    // reserve last position for indptr
    sizes[num_data] = param_.batch_size + 1;
    data_[num_data].resize(mshadow::Shape1(sizes[num_data]), mshadow::DataType<int32_t>::kFlag);
    out_.data.push_back(TBlob(data_[num_data].dptr_, mshadow::Shape1(sizes[num_data]), cpu::kDevMask,mshadow::DataType<int32_t>::kFlag));
    LOG(INFO) << "done init";
  }

  // initialize the data holder by using from the first data instance
  inline void InitDataFromInst(const DataInst& first_batch) {
    shape_.resize(first_batch.data.size());
    data_.resize(first_batch.data.size());
    unit_size_.resize(first_batch.data.size());
    offsets_.clear();
    offsets_.resize(first_batch.data.size(), 0);
    for (size_t i = 0; i < first_batch.data.size(); ++i) {
      TShape src_shape = first_batch.data[i].shape_;
      int src_type_flag = first_batch.data[i].type_flag_;
      // init object attributes
      std::vector<index_t> shape_vec;
      shape_vec.push_back(param_.batch_size);
      for (index_t dim = 0; dim < src_shape.ndim(); ++dim) {
        shape_vec.push_back(src_shape[dim]);
      }
      TShape dst_shape(shape_vec.begin(), shape_vec.end());
      shape_[i] = dst_shape;
      data_[i].resize(mshadow::Shape1(dst_shape.Size()), src_type_flag);
      unit_size_[i] = src_shape.Size();
      out_.data.push_back(TBlob(data_[i].dptr_, dst_shape, cpu::kDevMask, src_type_flag));
    }
  }

};  // class BatchLoader
}  // namespace io
}  // namespace mxnet
#endif  // MXNET_IO_ITER_BATCHLOADER_H_
