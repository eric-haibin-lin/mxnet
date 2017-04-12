//#include <time.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include "../src/executor/graph_executor.h"
#include "../src/operator/tensor/elemwise_binary_op.h"
#include "../src/ndarray/ndarray.cc"

using namespace mxnet;

void CheckDataRegion(const TBlob &src, const TBlob &dst) {
  auto size = src.shape_.Size() * mshadow::mshadow_sizeof(src.type_flag_);
  auto equals = memcmp(src.dptr_, dst.dptr_, size);
  EXPECT_EQ(equals, 0);
}

NDArray GetIndexND(TShape shape, Context ctx, std::vector<ROW_SPARSE_TYPE> values) {
  NDArray nd(shape, ctx, false, DEFAULT_AUX_TYPE);
  size_t num_vals = values.size();
  for (size_t i = 0; i < num_vals; i++) {
      nd.data().FlatTo1D<cpu, ROW_SPARSE_TYPE>()[i] = values[i];
  }
  return nd;
}

NDArray GetDenseND(TShape shape, Context ctx, std::vector<real_t> values) {
  NDArray nd(shape, ctx, false);
  size_t num_vals = values.size();
  for (size_t i = 0; i < num_vals; i++) {
      nd.data().FlatTo1D<cpu, real_t>()[i] = values[i];
  }
  return nd;
}


void BasicTest() {
  Context ctx;
  TShape shape({1, 2});
  NDArray nd(shape, ctx, false);
  EXPECT_NE(nd.data().dptr_, nullptr);
  Engine::Get()->WaitForAll();
}

void BinaryDenseSparseTest() {
  Context ctx = Context::CPU();

  TShape index_shape({2});
  NDArray index0 = GetIndexND(index_shape, ctx, {0, 1});

  TShape data_shape({2, 2});
  NDArray raw_data0 = GetDenseND(data_shape, ctx, {10, 10, 10, 10});

  TShape output_shape({3, 2});
  NDArray input_nd0(raw_data0, {index0}, ctx, kRowSparseChunk, data_shape);
  NDArray input_nd1 = GetDenseND(output_shape, ctx, {1, 2, 3, 4, 5, 6});
  Engine::Get()->WaitForAll();

  NDArray output(kRowSparseChunk, output_shape, ctx);
  // Push the right vars! FIXME
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(raw_data0.var());
  const_vars.push_back(index0.var());
  // TODO Add switch stmt
      Engine::Get()->PushSync([input_nd0, input_nd1, output](RunContext ctx) {
          nnvm::NodeAttrs attrs;
          OpContext op_ctx;
          std::vector<NDArray> inputs, outputs;
          std::vector<OpReqType> req;
          inputs.push_back(input_nd0);
          inputs.push_back(input_nd1);
          outputs.push_back(output);
          op::BinaryComputeND<cpu, mshadow::op::plus>(attrs, op_ctx, inputs, req, outputs);
        }, input_nd0.ctx(), const_vars, {output.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  std::vector<real_t> output_vals({11, 12, 3, 4, 15, 16});
  NDArray out_data = GetDenseND(output_shape, ctx, output_vals);
  Engine::Get()->WaitForAll();
  CheckDataRegion(out_data.data(), output.data());
  // TODO also check with zeros..
}
void BinarySpSpTest() {
  Context ctx = Context::CPU();

  TShape index_shape({2});
  NDArray index0 = GetIndexND(index_shape, ctx, {0, 1});
  NDArray index1 = GetIndexND(index_shape, ctx, {0, 2});

  TShape data_shape({2, 2});
  NDArray raw_data0 = GetDenseND(data_shape, ctx, {10, 10, 10, 10});
  NDArray raw_data1 = GetDenseND(data_shape, ctx, {5, 5, 5, 5});
  Engine::Get()->WaitForAll();

  NDArray input_nd0(raw_data0, {index0}, ctx, kRowSparseChunk, data_shape);
  NDArray input_nd1(raw_data1, {index1}, ctx, kRowSparseChunk, data_shape);
  CheckDataRegion(input_nd0.data(), raw_data0.data());
  CheckDataRegion(input_nd1.data(), raw_data1.data());

  TShape output_shape({3, 2});
  NDArray output(kRowSparseChunk, output_shape, ctx);
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(raw_data0.var());
  const_vars.push_back(raw_data1.var());
  const_vars.push_back(index0.var());
  const_vars.push_back(index1.var());

      Engine::Get()->PushSync([input_nd0, input_nd1, output](RunContext ctx) {
          nnvm::NodeAttrs attrs;
          OpContext op_ctx;
          std::vector<NDArray> inputs, outputs;
          std::vector<OpReqType> req;
          inputs.push_back(input_nd0);
          inputs.push_back(input_nd1);
          outputs.push_back(output);
          op::BinaryComputeNDSpSp<cpu, cpu>(attrs, op_ctx, inputs, req, outputs);
        }, input_nd0.ctx(), const_vars, {output.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);

  std::vector<real_t> output_vals({15, 15, 10, 10, 5, 5});
  NDArray out_data = GetDenseND(output_shape, ctx, output_vals);
  Engine::Get()->WaitForAll();
  CheckDataRegion(out_data.data(), output.data());
  // TODO also check with zeros..
}

void InferElemwiseChunkTest() {
  nnvm::NodeAttrs attrs;
  attrs.name = "Test op";
  std::vector<int> in_attrs({kRowSparseChunk, kDefaultChunk});
  std::vector<int> out_attrs({-1});

  op::ElemwiseChunkType<2, 1>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultChunk);
  in_attrs = {kDefaultChunk, kRowSparseChunk};
  out_attrs = {-1};
  op::ElemwiseChunkType<2, 1>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultChunk);
}

TEST(NDArray, basics) {
  BasicTest();
  BinarySpSpTest();
  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  InferElemwiseChunkTest();
}

// dense to dense conversion
void TestDenseToDenseConversion() {
  Context ctx;
  const real_t val = 1;
  TShape shape({2, 2});
  NDArray nd(shape, ctx, false);
  EXPECT_NE(nd.data().dptr_, nullptr);
  nd = val;
  // TODO remove WaitFroAll
  Engine::Get()->WaitForAll();
  auto nd_copy = nd.ConvertTo<cpu>(kDefaultChunk, nullptr);
  CheckDataRegion(nd_copy.data(), nd.data());
}
// TODO refactor: GetDense, GetSparse
// sparse to dense conversion
void TestSparseToDenseConversion() {
  Context ctx;
  const real_t val = 0;
  // Raw Data
  NDArray raw_data0(TShape({1, 2}), ctx, false);
  raw_data0 = 0;
  
  // Index
  NDArray index0(TShape({1}), ctx, false, DEFAULT_AUX_TYPE);
  index0 = 0;

  TShape shape({2, 2});
  NDArray nd(raw_data0, {index0}, ctx, kRowSparseChunk, shape);

  // Dense ndarray
  NDArray dense_nd(shape, ctx, false);
  dense_nd = 0;
  dense_nd.data().FlatTo2D<cpu, real_t>()[0][0] = 1;
  dense_nd.data().FlatTo2D<cpu, real_t>()[0][1] = 1;
  Engine::Get()->WaitForAll(); 

  auto converted_nd = nd.ConvertTo<cpu>(kDefaultChunk, nullptr);
  auto converted_data = converted_nd.data();
  CheckDataRegion(converted_data, dense_nd.data());
}

TEST(NDArray, conversion) {
  TestDenseToDenseConversion();
  TestSparseToDenseConversion();
  Engine::Get()->WaitForAll();
  LOG(INFO) << "All pass";
}
