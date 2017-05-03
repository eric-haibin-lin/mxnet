#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include "../src/executor/graph_executor.h"
#include "../src/operator/tensor/elemwise_binary_op.h"
#include "../src/operator/tensor/elemwise_unary_op.h"
#include "../src/operator/tensor/indexing_op.h"
#include "../src/operator/optimizer_op-inl.h"
#include "../src/operator/tensor/init_op.h"
#include "test_utils.h"

using namespace mxnet;

// Conversion Tests
void CastDnsDnsTest() {
  Context ctx;
  TShape shape({2, 2});
  NDArray nd = DnsND(shape, ctx, {});
  auto nd_copy = Convert(kDefaultStorage, nd);
  CheckDataRegion(nd_copy.data(), nd.data());
}

void CastRspDnsTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({2, 2});
  float v1 = RandFloat();
  float v2 = RandFloat();
  NDArray nd = RspND(shape, ctx, {0}, {v1, v2});
  // Dense ndarray
  NDArray dense_nd = DnsND(shape, ctx, {v1, v2, 0, 0});
  NDArray converted = Convert(kDefaultStorage, nd);
  CheckDataRegion(converted.data(), dense_nd.data());
}

// NDArray function tests
void SetValueTest() {
  Context ctx = Context::CPU();
  TShape data_shape({2, 2});
  float v = RandFloat();
  NDArray nd0 = DnsND(data_shape, ctx, {v, v, v, v});
  NDArray nd1(data_shape, ctx, false);
  nd1 = v;
  nd1.WaitToRead();
  CheckDataRegion(nd0.data(), nd1.data());
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
void SGDDnsRspTest() {
  TShape shape({4, 2});
  Context ctx = Context::CPU();
  NDArray weight = DnsND(shape, ctx, {1, 2, 3, 4, 5, 6, 7, 8});
  NDArray rsp_grad = RspND(shape, ctx, {0, 3}, {1, 2, 3, 4});
  NDArray output = weight;
  float lr = RandFloat();
  float wd = RandFloat();
  float rescale = RandFloat();
  op::SGDParam param;
  param.lr = lr;
  param.wd = wd;
  param.rescale_grad = rescale;
  param.clip_gradient = -1.0f;
  Engine::Get()->PushSync([weight, rsp_grad, output, param](RunContext ctx) {
      std::vector<NDArray> inputs{weight, rsp_grad}, outputs{output};
      std::vector<OpReqType> req({kAddTo});
      op::SparseSGDUpdateDnsRspImpl<cpu>(param, {}, inputs, req, outputs);
    }, weight.ctx(), {rsp_grad.var()}, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  auto sgd = [lr, wd, rescale] (TEST_DTYPE weight, TEST_DTYPE grad) {
     return (1.f-lr*wd)*weight - (lr*rescale)*grad;
    };

  NDArray expected = DnsND(shape, ctx,
                           {1 + sgd(1, 1), 2 + sgd(2, 2), 3, 4, 5, 6,
                           7 + sgd(7, 3), 8 + sgd(8, 4)});
  output.WaitToRead();
  CheckDataRegion(output.data(), expected.data());
}

void CopyFromToRspDnsTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({2, 2});
  NDArray nd = RspND(shape, ctx, {0}, {1, 1});
  // Dense ndarray
  NDArray dns_nd = DnsND(shape, ctx, {});
  CopyFromTo(nd, &dns_nd);
  dns_nd.WaitToRead();
  CheckDataRegion(nd.data(), dns_nd.data());
}

void CopyFromToRspRspReuseTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({3, 2});
  NDArray nd = RspND(shape, ctx, {0}, {1,2});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = RspND(shape, ctx, {0, 1, 2}, {6,6,6,6,6,6});
  nd.WaitToRead();
  CopyFromTo(nd, &dst_nd);
  dst_nd.WaitToRead();
  CheckDataRegion(nd.data(), dst_nd.data());
  CHECK_EQ(dst_nd.aux_shape(rowsparse::kIdx)[0], 1);
  CHECK_EQ(dst_nd.storage_shape()[0], 1);
  CHECK_EQ(dst_nd.storage_shape()[1], 2);
}


void CopyFromToRspRspFreeTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({3, 2});
  NDArray nd = RspND(shape, ctx, {0, 1}, {1,1,1,1});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = RspND(shape, ctx, {0}, {2,2});
  nd.WaitToRead();
  CopyFromTo(nd, &dst_nd);
  dst_nd.WaitToRead();
  CheckDataRegion(nd.data(), dst_nd.data());
}

void BinaryAddRspRsp() {
  Context ctx = Context::CPU();

  TShape output_shape({4, 2});
  NDArray input_nd0 = RspND(output_shape, ctx, {0, 1}, {10,10,10,10});
  NDArray input_nd1 = RspND(output_shape, ctx, {0, 2}, {5,5,5,5});

  NDArray output(kRowSparseStorage, output_shape, ctx);
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(input_nd0.var());
  const_vars.push_back(input_nd1.var());

  Engine::Get()->PushSync([input_nd0, input_nd1, output](RunContext ctx) {
      OpContext op_ctx;
      std::vector<NDArray> inputs, outputs;
      std::vector<OpReqType> req;
      inputs.push_back(input_nd0);
      inputs.push_back(input_nd1);
      outputs.push_back(output);
      op::BinaryComputeRspRsp<cpu, cpu>({}, op_ctx, inputs, req, outputs);
    }, input_nd0.ctx(), const_vars, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);

  // Check the data region of output ndarray
  NDArray dense_output = DnsND(output_shape, ctx, {15, 15, 10, 10, 5, 5, 0, 0});
  NDArray copy = Convert(kDefaultStorage, output);
  CheckDataRegion(dense_output.data(), copy.data());
}

void SparseEmbeddingBackwardTest() {
  Context ctx = Context::CPU();
  // d1 .. dk
  // idx shape : (2, 3)
  // input dim 4, output dim 2
  int input_dim = 4;
  int output_dim = 2;
  TShape idx_shape({2, 3});
  NDArray idx = RspIdxND(idx_shape, ctx, {1, 2, 3, 1, 2, 3});
  TShape grad_shape({2, 3, 2});
  NDArray grad = DnsND(grad_shape, ctx, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2});
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
      op::SparseEmbeddingOpBackwardEx<cpu>({}, {}, inputs, req, outputs);
    }, output.ctx(), {grad.var(), idx.var()}, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);

  NDArray expected = DnsND(out_shape, ctx, {0,0,0,0,0,0,0,0});
  Engine::Get()->PushSync([idx, grad, expected, param](RunContext ctx) {
      std::vector<TBlob> inputs{grad.data(), idx.data()}, outputs{expected.data(), expected.data()};
      std::vector<OpReqType> req({kNullOp, kWriteTo});
      op::EmbeddingOpBackward<cpu>({}, {}, inputs, req, outputs);
    }, expected.ctx(), {grad.var(), idx.var()}, {expected.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  NDArray converted = Convert(kDefaultStorage, output);
  expected.WaitToRead();
  CheckDataRegion(converted.data(), expected.data());
}


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

TEST(NDArray, optimizer) {
  SGDDnsRspTest();
}

TEST(NDArray, copy) {
  CopyFromToRspDnsTest();
  CopyFromToRspRspReuseTest();
  CopyFromToRspRspFreeTest();
}

TEST(NDArray, infer_storage) {
  InferElemwiseStorageTest();
}

TEST(NDArray, sparse_embedding) {
  SparseEmbeddingBackwardTest();
}
