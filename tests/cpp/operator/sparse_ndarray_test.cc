/*!
 * Copyright (c) 2017 by Contributors
 * \file ndarray_test.cc
 * \brief ndarray unit test utility functions
 * \author Haibin Lin
*/
#include <unistd.h>
#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <mxnet/operator_util.h>
#include <mxnet/ndarray.h>
#include "../src/operator/mshadow_op.h"
#include "../src/operator/tensor/indexing_op.h"
#include "../src/operator/mshadow_op.h"
#include "test_ndarray_utils.h"
int __static_result_putenv = putenv(const_cast<char *>("MXNET_ENGINE_TYPE=NaiveEngine"));
#if 1
using namespace mxnet;

// Conversion Tests
void CastDnsDnsTest() {
  Context ctx;
  TShape shape({2, 2});
  NDArray nd = test::DnsND(shape, ctx, {});
  auto nd_copy = test::Convert(kDefaultStorage, nd);
  test::CheckDataRegion(nd_copy.data(), nd.data());
}

void CastRspDnsTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({2, 2});
  float v1 = test::RandFloat();
  float v2 = test::RandFloat();
  NDArray nd = test::RspND(shape, ctx, {0}, {v1, v2});
  // Dense ndarray
  NDArray dense_nd = test::DnsND(shape, ctx, {v1, v2, 0, 0});
  NDArray converted = test::Convert(kDefaultStorage, nd);
  test::CheckDataRegion(converted.data(), dense_nd.data());
}

// NDArray function tests
void SetValueTest() {
  Context ctx = Context::CPU();
  TShape data_shape({2, 2});
  float v = test::RandFloat();
  NDArray nd0 = test::DnsND(data_shape, ctx, {v, v, v, v});
  NDArray nd1(data_shape, ctx, false);
  nd1 = v;
  nd1.WaitToRead();
  test::CheckDataRegion(nd0.data(), nd1.data());
}

// InferStorage
void InferElemwiseStorageTest() {
  Context ctx;
  nnvm::NodeAttrs attrs;
  attrs.name = "test_op";
  std::vector<int> in_attrs({kRowSparseStorage, kDefaultStorage});
  std::vector<int> out_attrs({kUndefinedStorage});
  // rsp, default -> default
  op::ElemwiseStorageType<2, 1>(attrs, ctx, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultStorage);
  // default, rsp -> default
  in_attrs = {kDefaultStorage, kRowSparseStorage};
  out_attrs = {kUndefinedStorage};
  op::ElemwiseStorageType<2, 1>(attrs, ctx, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultStorage);
  // rsp, rsp -> rsp
  in_attrs = {kRowSparseStorage};
  out_attrs = {kUndefinedStorage, kUndefinedStorage};
  op::ElemwiseStorageType<1, 2>(attrs, ctx, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kRowSparseStorage);
  EXPECT_EQ(out_attrs[1], kRowSparseStorage);
}

void CopyFromToRspDnsTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({2, 2});
  NDArray nd = test::RspND(shape, ctx, {0}, {1, 1});
  // Dense ndarray
  NDArray dns_nd = test::DnsND(shape, ctx, {});
  CopyFromTo(nd, &dns_nd);
  dns_nd.WaitToRead();
  test::CheckDataRegion(nd.data(), dns_nd.data());
}

void CopyFromToRspRspReuseTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({3, 2});
  NDArray nd = test::RspND(shape, ctx, {0}, {1, 2});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = test::RspND(shape, ctx, {0, 1, 2}, {6, 6, 6, 6, 6, 6});
  nd.WaitToRead();
  CopyFromTo(nd, &dst_nd);
  dst_nd.WaitToRead();
  test::CheckDataRegion(nd.data(), dst_nd.data());
  CHECK_EQ(dst_nd.aux_shape(rowsparse::kIdx)[0], 1);
  CHECK_EQ(dst_nd.storage_shape()[0], 1);
  CHECK_EQ(dst_nd.storage_shape()[1], 2);
}


void CopyFromToRspRspFreeTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({3, 2});
  NDArray nd = test::RspND(shape, ctx, {0, 1}, {1, 1, 1, 1});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = test::RspND(shape, ctx, {0}, {2, 2});
  nd.WaitToRead();
  CopyFromTo(nd, &dst_nd);
  dst_nd.WaitToRead();
  test::CheckDataRegion(nd.data(), dst_nd.data());
}

template<typename xpu, typename OP>
void BinaryOpForwardRspRsp() {
  Context ctx = Context::CPU();

  TShape output_shape({4, 2});
  NDArray input_nd0 = test::RspND(output_shape, ctx, {0, 1}, {10, 10, 10, 10});
  NDArray input_nd1 = test::RspND(output_shape, ctx, {0, 2}, {5, 5, 5, 5});

  NDArray output(kRowSparseStorage, output_shape, ctx);
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(input_nd0.var());
  const_vars.push_back(input_nd1.var());

  Engine::Get()->PushSync([input_nd0, input_nd1, output](RunContext ctx) {
      OpContext op_ctx;
      std::vector<NDArray> inputs, outputs;
      std::vector<OpReqType> req = { kWriteTo };
      inputs.push_back(input_nd0);
      inputs.push_back(input_nd1);
      outputs.push_back(output);
      op::ElemwiseBinaryOp::LaunchEx<xpu, OP>({}, op_ctx, inputs, req, outputs);
    }, input_nd0.ctx(), const_vars, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);

  // Check the data region of output ndarray
  NDArray dense_output = test::DnsND(output_shape, ctx, {15, 15, 10, 10, 5, 5, 0, 0});
  NDArray copy = test::Convert(kDefaultStorage, output);
  test::CheckDataRegion(dense_output.data(), copy.data());
}

class TestOldCode {
  template<OpReqType req>
  struct EmbeddingBackwardRsp {
    template<typename DType, typename IType>
    // each thread i is responsible for target gradient row ids in [segment_start, segment_end)
    MSHADOW_XINLINE static void Map(int i, const size_t width, IType* dst_idx, DType* dst_val,
                                    const IType* idx, const size_t num_idx, const DType* src,
                                    const size_t segment_len, const size_t num_rows) {
      OpReqType req_type = req;
      size_t segment_start = i * segment_len;
      size_t segment_end = (i + 1) * segment_len;
      for (size_t y = 0; y < num_idx; y++) {
        size_t j = idx[y];
        if (j >= num_rows) j = num_rows - 1;
        if (j < segment_start || j >= segment_end) continue;
        dst_idx[j] = j;
        for (size_t k = 0; k < width; k++) {
          if (req_type == kWriteTo) req_type = kAddTo;
          KERNEL_ASSIGN(dst_val[j * width + k], req_type, src[y * width + k]);
        }
      }
    }
  };

/*
 * for sparse embedding, the storage type for weight gradient is row_sparse.
 * we don't care about the storage type for data gradient, since it is not
 * differentiable.
 */
  static inline bool SparseEmbeddingBackwardStorageType(const nnvm::NodeAttrs& attrs,
                                                 std::vector<int> *in_attrs,
                                                 std::vector<int> *out_attrs) {
    CHECK_EQ((*in_attrs)[0], kDefaultStorage);
    CHECK_EQ((*in_attrs)[1], kDefaultStorage);
    (*out_attrs)[0] = kRowSparseStorage;
    (*out_attrs)[1] = kRowSparseStorage;
    return true;
  }

  template<typename xpu>
  static void SparseEmbeddingOpBackwardDnsDnsRsp(const nnvm::NodeAttrs& attrs,
                                          const OpContext& ctx,
                                          const std::vector<NDArray>& inputs,
                                          const std::vector<OpReqType>& req,
                                          const std::vector<NDArray>& outputs) {
    using namespace mxnet::op;
    using namespace mxnet::op::mxnet_op;
    using namespace mshadow::expr;
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 2U);
    if (req[1] == kNullOp) return;
    // check storage types
    auto idx = inputs[1];  // idx shape (d1, d2 .. dk)
    auto grad = inputs[0];  // grad shape (d1, d2, .. dk, out_dim)
    auto output = outputs[1];  // weight shape (in_dim, out_dim)
    CHECK_EQ(idx.storage_type(), kDefaultStorage);
    CHECK_EQ(grad.storage_type(), kDefaultStorage);
    CHECK_EQ(output.dtype(), grad.dtype());
    CHECK_EQ(idx.dtype(), output.aux_type(rowsparse::kIdx)) << "Index type doesn't match";

    const nnvm::TShape& ishape = idx.shape();
    const nnvm::TShape& oshape = grad.shape();

    mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
    CHECK_EQ(idx.dtype(), output.aux_type(rowsparse::kIdx))
      << "embedding input index and gradient row sparse type doesn't match!";
    // Alloc dense output
    unsigned int num_rows = output.shape()[0];
    output.CheckAndAlloc({mshadow::Shape1(num_rows)});
    MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
      MSHADOW_IDX_TYPE_SWITCH(idx.dtype(), IType, {
        MXNET_ASSIGN_REQ_SWITCH(req[1], req_type, {
          // input embedding indice, each idx in [0, input_dim)
          auto idx_data = idx.data().FlatTo1D<xpu, IType>(s);
          auto grad_data = grad.data().get_with_shape<xpu, 2, DType>(
          mshadow::Shape2(oshape.ProdShape(0, oshape.ndim()-1), oshape[oshape.ndim()-1]), s);
          auto output_idx = output.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
          auto output_val = output.data().FlatTo2D<xpu, DType>(s);
          int num_threads = omp_get_num_threads();
          size_t width = output.shape()[1];
          size_t segment_len = (num_rows + num_threads - 1) / num_threads;
          // fill indices with invalid row ids
          Kernel<mxnet_op::fill, xpu>::Launch(s, num_rows, output_idx.dptr_,
          static_cast<IType>(num_rows));
          // fill zeros if needed
          if (req_type == kWriteTo) {
            Kernel<mxnet_op::set_zero, xpu>::Launch(s, output_val.shape_.Size(), output_val.dptr_);
          }
          Kernel<EmbeddingBackwardRsp<req_type>, xpu>::Launch(s, num_threads, width,
          output_idx.dptr_,
          output_val.dptr_, idx_data.dptr_,
          ishape.Size(), grad_data.dptr_,
          segment_len, num_rows);
        });
      });
    });
  }

 public:
  // todo replace xpu with cpu
  template<typename xpu>
  static void SparseEmbeddingOpBackwardEx(const nnvm::NodeAttrs& attrs,
                                   const OpContext& ctx,
                                   const std::vector<NDArray>& inputs,
                                   const std::vector<OpReqType>& req,
                                   const std::vector<NDArray>& outputs) {
    CHECK_EQ(inputs.size(), 2U);
    CHECK_EQ(outputs.size(), 2U);
    // idx shape (d1, d2 .. dk)
    auto idx_stype = inputs[1].storage_type();
    // grad shape (d1, d2, .. dk, out_dim)
    auto grad_stype = inputs[0].storage_type();
    // weight shape (in_dim, out_dim)
    auto output_stype = outputs[1].storage_type();
    if (idx_stype == kDefaultStorage && grad_stype == kDefaultStorage &&
        output_stype == kRowSparseStorage) {
      SparseEmbeddingOpBackwardDnsDnsRsp<xpu>(attrs, ctx, inputs, req, outputs);
    } else {
      LOG(FATAL) << "Not implemented";
    }
  }
};

static void SparseEmbeddingBackwardTest() {
  Context ctx = Context::CPU();
  // d1 .. dk
  // idx shape : (2, 3)
  // input dim 4, output dim 2
  int input_dim = 4;
  int output_dim = 2;
  TShape idx_shape({2, 3});
  NDArray idx = test::RspIdxND(idx_shape, ctx, {1, 2, 3, 1, 2, 3});
  TShape grad_shape({2, 3, 2});
  NDArray grad = test::DnsND(grad_shape, ctx, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                               0.7, 0.8, 0.9, 1.0, 1.1, 1.2});
  TShape out_shape({4, 2});
  NDArray output = NDArray(kRowSparseStorage, out_shape, ctx);
  op::EmbeddingParam param;
  param.input_dim = input_dim;
  param.output_dim = output_dim;
  param.dtype = 0;

  Engine::Get()->PushSync([idx, grad, output, param](RunContext ctx) {
      std::vector<NDArray> inputs{grad, idx}, outputs{output, output};
      // this is a hack
      std::vector<OpReqType> req({kNullOp, kAddTo});
      TestOldCode::SparseEmbeddingOpBackwardEx<cpu>({}, {}, inputs, req, outputs);
    }, output.ctx(), {grad.var(), idx.var()}, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);

  NDArray expected = test::DnsND(out_shape, ctx, {0, 0, 0, 0, 0, 0, 0, 0});
  Engine::Get()->PushSync([idx, grad, expected, param](RunContext ctx) {
      std::vector<TBlob> inputs{grad.data(), idx.data()}, outputs{expected.data(), expected.data()};
      std::vector<OpReqType> req({kNullOp, kWriteTo});
      op::EmbeddingOpBackward<cpu>({}, {}, inputs, req, outputs);
    }, expected.ctx(), {grad.var(), idx.var()}, {expected.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  NDArray converted = test::Convert(kDefaultStorage, output);
  expected.WaitToRead();
}

TEST(NDArray, BinaryOps) {
  BinaryOpForwardRspRsp<cpu, mshadow::op::plus>();
  BinaryOpForwardRspRsp<cpu, mshadow::op::minus>();
  BinaryOpForwardRspRsp<cpu, mshadow::op::mul>();
  BinaryOpForwardRspRsp<cpu, mshadow::op::div>();
}

TEST(NDArray, conversion) {
  CastDnsDnsTest();
  CastRspDnsTest();
}

TEST(NDArray, functions) {
  SetValueTest();
}

TEST(NDArray, copy) {
  CopyFromToRspDnsTest();
  CopyFromToRspRspReuseTest();
  CopyFromToRspRspFreeTest();
}

TEST(NDArray, infer_storage) {
  InferElemwiseStorageTest();
}

TEST(NDArray, ArrayStruct) {
  typedef float DType;
  const TShape shape({150, 250});
  test::Array<DType> array(shape);  // shape is H, W
  array[2][5] = 0.1;  // [x][y] <-- [col][row]
  array[6][17] = 0.2;
  array[6][52] = 0.3;
  array[115][220] = 0.4;

  Context ctx ; Context::CPU(-1);

  const NDArray row_sparse = array.Save(ctx, kRowSparseStorage);
  const NDArray csr = array.Save(ctx, kCSRStorage);

  const NDArray dense_1 = test::Array<DType>::Convert(ctx, row_sparse, kDefaultStorage);
  const NDArray dense_2 = test::Array<DType>::Convert(ctx, csr, kDefaultStorage);

  const DType *p1 = dense_1.data().dptr<DType>();
  const DType *p2 = dense_2.data().dptr<DType>();
  CHECK_EQ(memcmp(p1, p2, dense_1.shape().Size() * sizeof(DType)), 0);
  test::Array<DType> array1;
  array1.Load(dense_1);
  CHECK_EQ(test::Array<DType>::IsNear(array1[2][5], 0.1), true);
  CHECK_EQ(test::Array<DType>::IsNear(array1[6][17], 0.2), true);
  CHECK_EQ(test::Array<DType>::IsNear(array1[6][52], 0.3), true);
  CHECK_EQ(test::Array<DType>::IsNear(array1[115][220], 0.4), true);
}

TEST(NDArray, sparse_embedding) {
  putenv(const_cast<char *>("MXNET_ENGINE_TYPE=NaiveEngine"));
  SparseEmbeddingBackwardTest();
}

struct TPosition : public TShape {
  inline explicit TPosition(const TShape& o) : TShape(o) {}
};

template<typename DType>
class ArrayIterator {
 public:
  virtual DType *Begin() = 0;
  virtual DType *Next() = 0;
  virtual TPosition Position() const = 0;
  virtual DType *Current() = 0;
  virtual const DType *Current() const = 0;
};

template<typename DType, typename IType>
class RowSparseArrayIterator : public ArrayIterator<DType> {
 public:
  explicit RowSparseArrayIterator(const NDArray& array)
  : array_(array)
    , position_(array.shape().ndim())
    , row_count_(array_.aux_shape(rowsparse::kIdx)[0])
    , dptr_(array.data().dptr<DType>())  {
    CHECK_EQ(array.storage_type(), kRowSparseStorage);
    DCHECK_NE(array_.shape().Size(), 0);
    DCHECK_GT(row_count_, 0);
    for (size_t i = 0, n = position_.ndim(); i < n; ++i) {
      position_[i] = 0;
    }
  }

  DType *Begin() override {
    for (size_t i = 0, n = position_.ndim(); i < n; ++i) {
      position_[i] = 0;
    }
    dptr_ = array_.data().template dptr<DType>();
    return dptr_;
  }

  DType *Next() override {
    CHECK_NE(dptr_, static_cast<DType *>(nullptr));
    const int dim = position_.ndim();
    const TShape &shape = array_.shape();
    DCHECK_GT(dim, 1);
    bool lastFlipped = true;
    for (int i = dim - 1; i >= 0; --i) {
      nnvm::dim_t curr = position_[i];
      if (lastFlipped) {
        ++curr;
      }
      const int limit = i ? shape[i] : row_count_;
      if (curr >= limit) {
        if (!i) {
          return nullptr;
        }
        curr = 0;
        CHECK_GT(i, 0);
        lastFlipped = true;
        position_[i] = curr;
      } else {
        position_[i] = curr;
        break;
      }
    }
    return dptr_++;
  }

  TPosition Position() const override {
    TPosition pos = position_;
    IType *it = array_.aux_data(rowsparse::kIdx).template dptr<IType>();
    pos[0] = it[position_[0]];
    return pos;
  }

  DType *Current() { return dptr_; }
  const DType *Current() const { return dptr_; }

 private:
  const NDArray& array_;
  TPosition      position_;  // Use dim optimization characteristics
                             // (ie up to 4 pos don't need new allocations)
  const int      row_count_;
  DType *        dptr_;
};

TEST(NDArray, SparseArrayIteratorRSP) {
  typedef float DType;
  const TShape shape({3, 2});
  test::Array<DType> array(shape);  // shape is H, W
  array[1][0] = 0.1;
  array[2][1] = 0.2;

  const Context ctx = Context::CPU(-1);

  const NDArray row_sparse = array.Save(ctx, kRowSparseStorage);

  RowSparseArrayIterator<DType, int> rsp_iter(row_sparse);
  do {
    TPosition position = rsp_iter.Position();
    std::cout << "( ";
    for (size_t i = 0, n = position.ndim(); i < n; ++i) {
      if (i) {
        std::cout << ", ";
      }
      std::cout << position[i];
    }
    std::cout << " ): ";
    const DType val = *rsp_iter.Current();
    std::cout << val << std::endl << std::flush;
  } while (rsp_iter.Next());
}

template<typename DType, typename IType>
class CSRSparseArrayIterator : public ArrayIterator<DType> {
 public:
  explicit CSRSparseArrayIterator(const NDArray& array)
    : array_(array)
      , position_(array.shape().ndim())
      , dptr_(array.data().dptr<DType>())  {
  }

  virtual DType *Begin() {
    return nullptr;
  }
  virtual DType *Next()  {
    return nullptr;
  }
  virtual TPosition Position() const {
    return position_;
  }

  virtual DType *Current() { return dptr_; }
  virtual const DType *Current() const { return dptr_; }

 private:
  const NDArray& array_;
  TPosition      position_;  // Use dim optimization characteristics
                             // (ie up to 4 pos don't need new allocations)
  DType *        dptr_;
};

TEST(NDArray, SparseArrayIteratorCSR) {
  typedef float DType;
  const TShape shape({3, 2});
  test::Array<DType> array(shape);  // shape is H, W
  array[1][0] = 0.1;
  array[2][1] = 0.2;

  const Context ctx = Context::CPU(-1);

  const NDArray csr = array.Save(ctx, kCSRStorage);

  CSRSparseArrayIterator<DType, int> rsp_iter(csr);
  do {
    TPosition position = rsp_iter.Position();
    std::cout << "( ";
    for (size_t i = 0, n = position.ndim(); i < n; ++i) {
      if (i) {
        std::cout << ", ";
      }
      std::cout << position[i];
    }
    std::cout << " ): ";
    const DType val = *rsp_iter.Current();
    std::cout << val << std::endl << std::flush;
  } while (rsp_iter.Next());
}

#endif
