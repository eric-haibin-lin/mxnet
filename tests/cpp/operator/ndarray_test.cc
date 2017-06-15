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
#include "../src/executor/graph_executor.h"
#include "../src/operator/tensor/elemwise_unary_op.h"
#include "../src/operator/tensor/elemwise_binary_op.h"
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
  nnvm::NodeAttrs attrs;
  attrs.name = "test_op";
  std::vector<int> in_attrs({kRowSparseStorage, kDefaultStorage});
  std::vector<int> out_attrs({kUndefinedStorage});
  // rsp, default -> default
  op::ElemwiseStorageType<2, 1>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultStorage);
  // default, rsp -> default
  in_attrs = {kDefaultStorage, kRowSparseStorage};
  out_attrs = {kUndefinedStorage};
  op::ElemwiseStorageType<2, 1>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultStorage);
  // rsp, rsp -> rsp
  in_attrs = {kRowSparseStorage};
  out_attrs = {kUndefinedStorage, kUndefinedStorage};
  op::ElemwiseStorageType<1, 2>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kRowSparseStorage);
  EXPECT_EQ(out_attrs[1], kRowSparseStorage);
}

// Optimizer
//void SGDDnsRspTest() {
//  TShape shape({4, 2});
//  Context ctx = Context::CPU();
//  NDArray weight = DnsND(shape, ctx, {1, 2, 3, 4, 5, 6, 7, 8});
//  NDArray rsp_grad = RspND(shape, ctx, {0, 3}, {1, 2, 3, 4});
//  NDArray output = weight;
//  float lr = RandFloat();
//  float wd = RandFloat();
//  float rescale = RandFloat();
//  op::SGDParam param;
//  param.lr = lr;
//  param.wd = wd;
//  param.rescale_grad = rescale;
//  param.clip_gradient = -1.0f;
//  Engine::Get()->PushSync([weight, rsp_grad, output, param](RunContext ctx) {
//      std::vector<NDArray> inputs{weight, rsp_grad}, outputs{output};
//      std::vector<OpReqType> req({kAddTo});
//      op::SparseSGDUpdateDnsRspImpl<cpu>(param, {}, inputs, req, outputs);
//    }, weight.ctx(), {rsp_grad.var()}, {output.var()},
//    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
//  auto sgd = [lr, wd, rescale] (TEST_DTYPE weight, TEST_DTYPE grad) {
//     return (1.f-lr*wd)*weight - (lr*rescale)*grad;
//    };
//
//  NDArray expected = DnsND(shape, ctx,
//                           {1 + sgd(1, 1), 2 + sgd(2, 2), 3, 4, 5, 6,
//                           7 + sgd(7, 3), 8 + sgd(8, 4)});
//  output.WaitToRead();
//  CheckDataRegion(output.data(), expected.data());
//}

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
  NDArray nd = test::RspND(shape, ctx, {0}, {1,2});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = test::RspND(shape, ctx, {0, 1, 2}, {6,6,6,6,6,6});
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
  NDArray nd = test::RspND(shape, ctx, {0, 1}, {1,1,1,1});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = test::RspND(shape, ctx, {0}, {2,2});
  nd.WaitToRead();
  CopyFromTo(nd, &dst_nd);
  dst_nd.WaitToRead();
  test::CheckDataRegion(nd.data(), dst_nd.data());
}

void BinaryAddRspRsp() {
  Context ctx = Context::CPU();

  TShape output_shape({4, 2});
  NDArray input_nd0 = test::RspND(output_shape, ctx, {0, 1}, {10,10,10,10});
  NDArray input_nd1 = test::RspND(output_shape, ctx, {0, 2}, {5,5,5,5});

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
      op::BinaryOp::ComputeRspRsp<cpu, mshadow::op::plus>({}, op_ctx, inputs, req, outputs);
    }, input_nd0.ctx(), const_vars, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);

  // Check the data region of output ndarray
  NDArray dense_output = test::DnsND(output_shape, ctx, {15, 15, 10, 10, 5, 5, 0, 0});
  NDArray copy = test::Convert(kDefaultStorage, output);
  test::CheckDataRegion(dense_output.data(), copy.data());
}

//void SparseEmbeddingBackwardTest() {
//  Context ctx = Context::CPU();
//  // d1 .. dk
//  // idx shape : (2, 3)
//  // input dim 4, output dim 2
//  int input_dim = 4;
//  int output_dim = 2;
//  TShape idx_shape({2, 3});
//  NDArray idx = RspIdxND(idx_shape, ctx, {1, 2, 3, 1, 2, 3});
//  TShape grad_shape({2, 3, 2});
//  NDArray grad = DnsND(grad_shape, ctx, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2});
//  TShape out_shape({4, 2});
//  NDArray output = NDArray(kRowSparseStorage, out_shape, ctx);
//  op::EmbeddingParam param;
//  param.input_dim = input_dim;
//  param.output_dim = output_dim;
//  param.dtype = 0;
//
//  Engine::Get()->PushSync([idx, grad, output, param](RunContext ctx) {
//      std::vector<NDArray> inputs{grad, idx}, outputs{output, output};
//      // this is a hack
//      std::vector<OpReqType> req({kNullOp, kAddTo});
//      op::SparseEmbeddingOpBackwardEx<cpu>({}, {}, inputs, req, outputs);
//    }, output.ctx(), {grad.var(), idx.var()}, {output.var()},
//    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
//
//  NDArray expected = DnsND(out_shape, ctx, {0,0,0,0,0,0,0,0});
//  Engine::Get()->PushSync([idx, grad, expected, param](RunContext ctx) {
//      std::vector<TBlob> inputs{grad.data(), idx.data()}, outputs{expected.data(), expected.data()};
//      std::vector<OpReqType> req({kNullOp, kWriteTo});
//      op::EmbeddingOpBackward<cpu>({}, {}, inputs, req, outputs);
//    }, expected.ctx(), {grad.var(), idx.var()}, {expected.var()},
//    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
//  NDArray converted = Convert(kDefaultStorage, output);
//  expected.WaitToRead();
//  CheckDataRegion(converted.data(), expected.data());
//}

TEST(NDArray, binary_add) {
  BinaryAddRspRsp();
}

TEST(NDArray, conversion) {
  CastDnsDnsTest();
  CastRspDnsTest();
}

TEST(NDArray, functions) {
  SetValueTest();
}

//TEST(NDArray, optimizer) {
//  SGDDnsRspTest();
//}

TEST(NDArray, copy) {
  CopyFromToRspDnsTest();
  CopyFromToRspRspReuseTest();
  CopyFromToRspRspFreeTest();
}

TEST(NDArray, infer_storage) {
  InferElemwiseStorageTest();
}

//TEST(NDArray, sparse_embedding) {
//  putenv("MXNET_ENGINE_TYPE=NaiveEngine");
//  SparseEmbeddingBackwardTest();
//}

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

struct TPosition : public TShape {
  inline TPosition(const TShape& o) : TShape(o) {}
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
  RowSparseArrayIterator(const NDArray& array)
  : array_(array)
    , position_(array.shape().ndim())
    , row_count_(array_.aux_shape(rowsparse::kIdx)[0])
    , dptr_(array.data().dptr<DType>())  {
    CHECK_EQ(array.storage_type(), kRowSparseStorage);
    DCHECK_NE(array_.shape().Size(), 0);
    DCHECK_GT(row_count_, 0);
    for(size_t i = 0, n = position_.ndim(); i < n; ++i) {
      position_[i] = 0;
    }
  }

  DType *Begin() override {
    for(size_t i = 0, n = position_.ndim(); i < n; ++i) {
      position_[i] = 0;
    }
    dptr_ = array_.data().dptr<DType>();
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
      if(curr >= limit) {
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
    IType *it = array_.aux_data(rowsparse::kIdx).dptr<IType>();
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
  //const TShape shape({150, 55});
  test::Array<DType> array(shape);  // shape is H, W
//  array[2][5] = 0.1;  // [x][y] <-- [col][row]
//  array[6][17] = 0.2;
//  array[6][52] = 0.3;
//  array[115][220] = 0.4;
  array[1][0] = 0.1;
  array[2][1] = 0.2;

  const Context ctx = Context::CPU(-1);

  const NDArray row_sparse = array.Save(ctx, kRowSparseStorage);

  RowSparseArrayIterator<DType, int> rsp_iter(row_sparse);
  do {
    TPosition position = rsp_iter.Position();
    std::cout << "( ";
    for(size_t i = 0, n = position.ndim(); i < n; ++i) {
      if(i) {
        std::cout << ", ";
      }
      std::cout << position[i];
    }
    std::cout << " ): ";
    const DType val = *rsp_iter.Current();
    std::cout << val << std::endl << std::flush;
  } while(rsp_iter.Next());
}

template<typename DType, typename IType>
class CSRSparseArrayIterator : public ArrayIterator<DType> {
 public:
  CSRSparseArrayIterator(const NDArray& array)
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
  //const TShape shape({150, 55});
  test::Array<DType> array(shape);  // shape is H, W
//  array[2][5] = 0.1;  // [x][y] <-- [col][row]
//  array[6][17] = 0.2;
//  array[6][52] = 0.3;
//  array[115][220] = 0.4;
  array[1][0] = 0.1;
  array[2][1] = 0.2;

  const Context ctx = Context::CPU(-1);

  const NDArray csr = array.Save(ctx, kCSRStorage);

  CSRSparseArrayIterator<DType, int> rsp_iter(csr);
  do {
    TPosition position = rsp_iter.Position();
    std::cout << "( ";
    for(size_t i = 0, n = position.ndim(); i < n; ++i) {
      if(i) {
        std::cout << ", ";
      }
      std::cout << position[i];
    }
    std::cout << " ): ";
    const DType val = *rsp_iter.Current();
    std::cout << val << std::endl << std::flush;
  } while(rsp_iter.Next());
}

#endif
