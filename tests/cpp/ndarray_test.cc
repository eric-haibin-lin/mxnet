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

void BasicTest() {
  Context ctx;
  TShape shape({1, 2});
  NDArray nd(shape, ctx, false);
  EXPECT_NE(nd.data().dptr_, nullptr);
  Engine::Get()->WaitForAll();
}

void BinarySparseTest() {
  Context ctx;
  const real_t val = 0;
  const real_t val1 = 1;
  const real_t val2 = 2;

  TShape index_shape({2});
  NDArray index0(index_shape, ctx, false, DEFAULT_AUX_TYPE);
  index0.data().FlatTo1D<cpu, ROW_SPARSE_TYPE>()[0] = 0;
  index0.data().FlatTo1D<cpu, ROW_SPARSE_TYPE>()[1] = 1;

  NDArray index1(index_shape, ctx, false, DEFAULT_AUX_TYPE);
  index1.data().FlatTo1D<cpu, ROW_SPARSE_TYPE>()[0] = 0;
  index1.data().FlatTo1D<cpu, ROW_SPARSE_TYPE>()[1] = 2;

  int dev_id = 0;
  TShape data_shape({2, 2});
  NDArray raw_data0(data_shape, ctx, false);
  raw_data0 = 10;
  
  NDArray raw_data1(data_shape, ctx, false);
  raw_data1 = 5;
  Engine::Get()->WaitForAll();

  NDArray input_nd0(raw_data0, {index0}, dev_id, kRowSparseChunk, data_shape);
  NDArray input_nd1(raw_data1, {index1}, dev_id, kRowSparseChunk, data_shape);
  CheckDataRegion(input_nd0.data(), raw_data0.data());
  CheckDataRegion(input_nd1.data(), raw_data1.data());

  TShape output_shape({3, 2});
  NDArray output(kRowSparseChunk, output_shape, ctx);
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(raw_data0.var());
  const_vars.push_back(raw_data1.var());
  const_vars.push_back(index0.var());
  const_vars.push_back(index1.var());

  // redirect everything to mshadow operations
  switch (input_nd0.ctx().dev_mask()) {
    case cpu::kDevMask: {
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
      break;
    }
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
  NDArray out_data(output_shape, ctx);
  out_data.CheckAndAlloc();
  out_data.data().FlatTo2D<cpu, real_t>()[0][0] = 15;
  out_data.data().FlatTo2D<cpu, real_t>()[0][1] = 15;
  out_data.data().FlatTo2D<cpu, real_t>()[1][0] = 10;
  out_data.data().FlatTo2D<cpu, real_t>()[1][1] = 10;
  out_data.data().FlatTo2D<cpu, real_t>()[2][0] = 5;
  out_data.data().FlatTo2D<cpu, real_t>()[2][1] = 5;
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
  BinarySparseTest();
  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  InferElemwiseChunkTest();
}

TEST(NDArray, conversion) {
  Context ctx;
  const real_t val = 0;
  // dense to dense conversion
  {
    TShape shape({2, 2});
    NDArray nd(shape, ctx, false);
    EXPECT_NE(nd.data().dptr_, nullptr);
    nd = val;
    Engine::Get()->WaitForAll();
    auto nd_copy = nd.ConvertTo<cpu>(kDefaultChunk);
    CheckDataRegion(nd_copy.data(), nd.data());
  }

  // sparse to dense conversion
  {
  size_t dev_id = 0;
  // Raw Data
  NDArray raw_data0(TShape({1, 2}), ctx, false);
  raw_data0 = 0;
  
  // Index
  NDArray index0(TShape({1}), ctx, false, DEFAULT_AUX_TYPE);
  index0 = 0;

  TShape shape({2, 2});
  NDArray nd(raw_data0, {index0}, dev_id, kRowSparseChunk, shape);

  // Dense ndarray
  NDArray dense_nd(shape, ctx, false);
  dense_nd = 0;
  dense_nd.data().FlatTo2D<cpu, real_t>()[0][0] = 1;
  dense_nd.data().FlatTo2D<cpu, real_t>()[0][1] = 1;
  Engine::Get()->WaitForAll(); 

  auto converted_nd = nd.ConvertTo<cpu>(kDefaultChunk);
  auto converted_data = converted_nd.data();
  CheckDataRegion(converted_data, dense_nd.data());
  }

  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  LOG(INFO) << "All pass";
}
