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

void CopyFromToRspRspTest() {
  Context ctx;
  // Sparse ndarray
  TShape shape({3, 2});
  NDArray nd = RspND(shape, ctx, {0}, {1, 1});
  // Sparse ndarray with enough memory. It's expected to reuse the memory
  NDArray dst_nd = RspND(shape, ctx, {0, 1, 2}, {});
  nd.WaitToRead();
  CopyFromTo(nd, &dst_nd);
  dst_nd.WaitToRead();
  CheckDataRegion(nd.data(), dst_nd.data());
}

/*
void BinaryDenseSparseTest() {
  Context ctx = Context::CPU();

  TShape output_shape({3, 2});
  NDArray input_nd0 = RspND(output_shape, ctx, {0, 1}, {10, 10, 10, 10});
  NDArray input_nd1 = DnsND(output_shape, ctx, {1, 2, 3, 4, 5, 6});
  NDArray output(kRowSparseStorage, output_shape, ctx);

  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(input_nd0.var());
  const_vars.push_back(input_nd1.var());
  Engine::Get()->PushSync([input_nd0, input_nd1, output](RunContext ctx) {
      nnvm::NodeAttrs attrs;
      OpContext op_ctx;
      std::vector<NDArray> inputs, outputs;
      std::vector<OpReqType> req;
      inputs.push_back(input_nd0);
      inputs.push_back(input_nd1);
      outputs.push_back(output);
      op::BinaryComputeEx<cpu, mshadow::op::plus>(attrs, op_ctx, inputs, req, outputs);
    }, input_nd0.ctx(), const_vars, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  std::vector<TEST_DTYPE> output_vals({11, 12, 3, 4, 15, 16});
  NDArray out_data = DnsND(output_shape, ctx, output_vals);
  Engine::Get()->WaitForAll();
  CheckDataRegion(out_data.data(), output.data());
  // TODO(haibin) also check with zeros..
}*/

void BinaryRsRsTest() {
  Context ctx = Context::CPU();

  TShape index_shape({2});
  NDArray index0 = RspIdxND(index_shape, ctx, {0, 1});
  NDArray index1 = RspIdxND(index_shape, ctx, {0, 2});

  TShape data_shape({2, 2});
  NDArray raw_data0 = DnsND(data_shape, ctx, {10, 10, 10, 10});
  NDArray raw_data1 = DnsND(data_shape, ctx, {5, 5, 5, 5});

  NDArray input_nd0(raw_data0, {index0}, ctx, kRowSparseStorage, data_shape);
  NDArray input_nd1(raw_data1, {index1}, ctx, kRowSparseStorage, data_shape);

  TShape output_shape({4, 2});
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
  CheckDataRegion(input_nd0.data(), raw_data0.data());
  CheckDataRegion(input_nd1.data(), raw_data1.data());
  CheckDataRegion(dense_output.data(), copy.data());
}

TEST(NDArray, binary_add) {
  BinaryRsRsTest();
  //BinaryDenseSparseTest();

  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  InferElemwiseStorageTest();
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
  CopyFromToRspRspTest();
}
