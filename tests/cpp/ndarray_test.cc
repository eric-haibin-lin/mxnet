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
#include "../src/operator/tensor/elemwise_unary_op.h"

#define TEST_DTYPE float
#define TEST_AUX_TYPE int32_t
using namespace mxnet;
void CheckDataRegion(const TBlob &src, const TBlob &dst) {
  auto size = src.shape_.Size() * mshadow::mshadow_sizeof(src.type_flag_);
  auto equals = memcmp(src.dptr_, dst.dptr_, size);
  EXPECT_EQ(equals, 0);
}

NDArray GetIndexND(const TShape shape, const Context ctx, const std::vector<TEST_AUX_TYPE> &values) {
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

NDArray GetDenseND(const TShape shape, const Context ctx, const std::vector<TEST_DTYPE> &values) {
  NDArray nd(shape, ctx, false);
  size_t num_val = values.size();
  CHECK_EQ(num_val, nd.shape().ProdShape(0, nd.shape().ndim()));
  MSHADOW_TYPE_SWITCH(nd.dtype(), DType, {
    auto tensor = nd.data().FlatTo1D<cpu, DType>();
    for (size_t i = 0; i < num_val; i++) {
      tensor[i] = values[i];
    }
  });
  return nd;
}

NDArray Convert(NDArrayStorageType type, NDArray src) {
  CHECK_EQ(type, kDefaultStorage);
  NDArray converted(src.shape(), src.ctx(), false);
  Engine::Get()->PushSync([src, converted](RunContext ctx) {
      // TODO provide type in attrs, which is empty now
      OpContext op_ctx;
      op_ctx.run_ctx = ctx;
      if (src.storage_type() == kRowSparseStorage) {
        std::vector<NDArray> inputs({src}), outputs({converted});
        op::CastStorageComputeEx<cpu>({}, op_ctx, inputs, {}, outputs);
      } else if (src.storage_type() == kDefaultStorage) {
        std::vector<TBlob> inputs({src.data()}), outputs({converted.data()});
        op::IdentityCompute<cpu>({}, op_ctx, inputs, {kWriteTo}, outputs);
      } else {
        LOG(FATAL) << "unsupported storage type";
      }
    }, src.ctx(), {src.var()}, {converted.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  converted.WaitToRead();
  return converted;
}

void BinaryDenseSparseTest() {
  Context ctx = Context::CPU();

  TShape index_shape({2});
  NDArray index0 = GetIndexND(index_shape, ctx, {0, 1});

  TShape data_shape({2, 2});
  NDArray raw_data0 = GetDenseND(data_shape, ctx, {10, 10, 10, 10});

  TShape output_shape({3, 2});
  NDArray input_nd0(raw_data0, {index0}, ctx, kRowSparseStorage, data_shape);
  NDArray input_nd1 = GetDenseND(output_shape, ctx, {1, 2, 3, 4, 5, 6});
  Engine::Get()->WaitForAll();

  NDArray output(kRowSparseStorage, output_shape, ctx);
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(raw_data0.var());
  const_vars.push_back(index0.var());
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
  NDArray out_data = GetDenseND(output_shape, ctx, output_vals);
  Engine::Get()->WaitForAll();
  CheckDataRegion(out_data.data(), output.data());
  // TODO(haibin) also check with zeros..
}

void SetValueTest() {
  Context ctx = Context::CPU();
  TShape data_shape({2, 2});
  NDArray nd0 = GetDenseND(data_shape, ctx, {10, 10, 10, 10});
  NDArray nd1(data_shape, ctx, false);
  nd1 = 10;
  nd1.WaitToRead();
  CheckDataRegion(nd0.data(), nd1.data());
}

void BinaryRsRsTest() {
  Context ctx = Context::CPU();

  TShape index_shape({2});
  NDArray index0 = GetIndexND(index_shape, ctx, {0, 1});
  NDArray index1 = GetIndexND(index_shape, ctx, {0, 2});

  TShape data_shape({2, 2});
  NDArray raw_data0 = GetDenseND(data_shape, ctx, {10, 10, 10, 10});
  NDArray raw_data1 = GetDenseND(data_shape, ctx, {5, 5, 5, 5});

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
      op::BinaryComputeRsRs<cpu, cpu>({}, op_ctx, inputs, req, outputs);
    }, input_nd0.ctx(), const_vars, {output.var()},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);


  // Check the data region of output ndarray
  NDArray dense_output = GetDenseND(output_shape, ctx, {15, 15, 10, 10, 5, 5, 0, 0});
  NDArray copy = Convert(kDefaultStorage, output);
  CheckDataRegion(input_nd0.data(), raw_data0.data());
  CheckDataRegion(input_nd1.data(), raw_data1.data());
  CheckDataRegion(dense_output.data(), copy.data());
}

void InferElemwiseStorageTest() {
  nnvm::NodeAttrs attrs;
  attrs.name = "Test op";
  std::vector<int> in_attrs({kRowSparseStorage, kDefaultStorage});
  std::vector<int> out_attrs({-1});

  op::ElemwiseStorageType<2, 1>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultStorage);
  in_attrs = {kDefaultStorage, kRowSparseStorage};
  out_attrs = {-1};
  op::ElemwiseStorageType<2, 1>(attrs, &in_attrs, &out_attrs);
  EXPECT_EQ(out_attrs[0], kDefaultStorage);
}

TEST(NDArray, basics) {
  BinaryRsRsTest();
  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  InferElemwiseStorageTest();
}

// dense to dense conversion
void TestDenseToDenseConversion() {
  Context ctx;
  TShape shape({2, 2});
  NDArray nd = GetDenseND(shape, ctx, {1, 2, 3, 10});
  auto nd_copy = Convert(kDefaultStorage, nd);
  CheckDataRegion(nd_copy.data(), nd.data());
}

// sparse to dense conversion
void TestSparseToDenseConversion() {
  Context ctx;
  // Raw Data
  NDArray raw_data0 = GetDenseND(TShape({1, 2}), ctx, {1, 1});
  // Index
  NDArray index0 = GetIndexND(TShape({1}), ctx, {0});
  // Sparse ndarray
  TShape shape({2, 2});
  NDArray nd(raw_data0, {index0}, ctx, kRowSparseStorage, shape);

  // Dense ndarray
  NDArray dense_nd = GetDenseND(shape, ctx, {1, 1, 0, 0});
  NDArray converted = Convert(kDefaultStorage, nd);
  CheckDataRegion(converted.data(), dense_nd.data());
}

TEST(NDArray, conversion) {
  TestDenseToDenseConversion();
  TestSparseToDenseConversion();
  Engine::Get()->WaitForAll();
}

TEST(NDArray, setvalue) {
  SetValueTest();
}
