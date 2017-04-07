//#include <time.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
using namespace mxnet;

void CheckDataRegion(const TBlob &src, const TBlob &dst) {
  auto size = src.shape_.Size() * mshadow::mshadow_sizeof(src.type_flag_);
  auto equals = memcmp(src.dptr_, dst.dptr_, size);
  EXPECT_EQ(equals, 0);
}

void VeryBasicTest() {
  Context ctx;
  const real_t val = 0;
  TShape shape2({1, 2});
  NDArray nd2(shape2, ctx);
  // alloc one row
  nd2.CheckAndAlloc();
  EXPECT_NE(nd2.data().dptr_, nullptr);
  nd2 = val;
  Engine::Get()->WaitForAll();
}

void BasicTest() {
  Context ctx;
  const real_t val = 0;
  const real_t val1 = 1;
  const real_t val2 = 2;

  VeryBasicTest();

  TShape index_shape({2});
  NDArray index0(index_shape, ctx);
  index0.CheckAndAlloc();
  index0.data().FlatTo1D<cpu, real_t>()[0] = 0;
  index0.data().FlatTo1D<cpu, real_t>()[1] = 1;

  NDArray index1(index_shape, ctx);
  index1.CheckAndAlloc();
  index1.data().FlatTo1D<cpu, real_t>()[0] = 0;
  index1.data().FlatTo1D<cpu, real_t>()[1] = 2;

  int dev_id = 0;
  TShape data_shape({2, 2});
  NDArray raw_data0(data_shape, ctx);
  raw_data0.CheckAndAlloc();
  raw_data0 = 10;
  
  NDArray raw_data1(data_shape, ctx);
  raw_data1.CheckAndAlloc();
  raw_data1 = 5;
  Engine::Get()->WaitForAll();

  NDArray input_nd0(raw_data0.data(), index0.data(), dev_id, RowSparseChunk, data_shape);
  NDArray input_nd1(raw_data1.data(), index1.data(), dev_id, RowSparseChunk, data_shape);
  CheckDataRegion(input_nd0.data(), raw_data0.data());
  CheckDataRegion(input_nd1.data(), raw_data1.data());

  TShape output_shape({3, 2});
  NDArray output(RowSparseChunk, output_shape, ctx);
  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(raw_data0.var());
  const_vars.push_back(raw_data1.var());
  const_vars.push_back(index0.var());
  const_vars.push_back(index1.var());

  // redirect everything to mshadow operations
  switch (input_nd0.ctx().dev_mask()) {
    case cpu::kDevMask: {
      Engine::Get()->PushSync([input_nd0, input_nd1, output](RunContext ctx) {
          TShape aux_shape({3});
          output.CheckAndAlloc(aux_shape);
          mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
          // Indices
          // NO AUX_SHAPE AVAILABLE
          auto in_idx_0 = input_nd0.aux_data().FlatTo1D<cpu, real_t>(s);
          auto in_idx_1 = input_nd1.aux_data().FlatTo1D<cpu, real_t>(s);
          TShape idx_shape0 = input_nd0.aux_shape();
          TShape idx_shape1 = input_nd1.aux_shape();

          auto in_data0 = input_nd0.data().FlatTo2D<cpu, real_t>(s);
          auto in_data1 = input_nd1.data().FlatTo2D<cpu, real_t>(s);
          auto out_data = output.data().FlatTo2D<cpu, real_t>(s);
          auto in_shape0 = input_nd0.chunk_shape();
          auto in_shape1 = input_nd1.chunk_shape();
          size_t num_rows_left = idx_shape0[0];
          size_t num_rows_right = idx_shape1[0];
          size_t i_left = 0;
          size_t i_right = 0;
          size_t i_out = 0;
          while (i_left < num_rows_left && i_right < num_rows_right) {
            size_t row_idx_left = in_idx_0[i_left];
            size_t row_idx_right = in_idx_1[i_right];
            if (row_idx_left == row_idx_right) {
              mshadow::Copy(out_data[i_out], in_data0[i_left++], s);
              out_data[i_out] += in_data1[i_right++];
            } else if (row_idx_left < row_idx_right) {
              mshadow::Copy(out_data[i_out], in_data0[i_left], s);
              i_left++;
            } else {
              mshadow::Copy(out_data[i_out], in_data1[i_right], s);
              i_right++;
            }
            i_out++;
          }

          while (i_left < num_rows_left) {
            mshadow::Copy(out_data[i_out++], in_data0[i_left++], s);
          }
          while (i_right < num_rows_right) {
            mshadow::Copy(out_data[i_out++], in_data1[i_right++], s);
          }
        }, input_nd0.ctx(), const_vars, {output.var()},
        FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
      break;
    }
    default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
  }
  //TODO Compare with dense matrix
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

TEST(NDArray, basics) {
  BasicTest();
  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
}

TEST(NDArray, conversion) {
  Context ctx;
  const real_t val = 0;
  // dense to dense conversion
  {
    TShape shape({2, 2});
    NDArray nd(shape, ctx);
    // alloc one row
    nd.CheckAndAlloc();
    EXPECT_NE(nd.data().dptr_, nullptr);
    nd = val;
    Engine::Get()->WaitForAll();
    auto nd_copy = nd.ConvertTo(DefaultChunk);
    CheckDataRegion(nd_copy.data(), nd.data());
  }

  // sparse to dense conversion
  {
  size_t dev_id = 0;
  // Raw Data
  NDArray raw_data0(TShape({1, 2}), ctx);
  raw_data0.CheckAndAlloc();
  raw_data0 = 0;
  
  // Index index_shape(1,)
  NDArray index0(TShape({1}), ctx);
  index0.CheckAndAlloc();
  index0 = 0; // idx = [0]

  TShape shape({2, 2});
  NDArray nd(raw_data0.data(), index0.data(), dev_id, RowSparseChunk, shape);

  // Dense ndarray
  NDArray dense_nd(shape, ctx);
  dense_nd.CheckAndAlloc();
  dense_nd = 0;
  dense_nd.data().FlatTo2D<cpu, real_t>()[0][0] = 1;
  dense_nd.data().FlatTo2D<cpu, real_t>()[0][1] = 1;
  Engine::Get()->WaitForAll(); 

  std::vector<Engine::VarHandle> const_vars;
  const_vars.push_back(nd.var());
  const_vars.push_back(raw_data0.var());
  const_vars.push_back(index0.var());
  const_vars.push_back(dense_nd.var());
  Engine::Get()->PushSync([nd, dense_nd](RunContext ctx) {
      mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
      auto nd_copy = nd.ConvertTo(DefaultChunk);
      CheckDataRegion(nd_copy.data(), dense_nd.data());
    }, nd.ctx(), const_vars, {},
    FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
  auto converted_nd = nd.ConvertTo(DefaultChunk);
  auto converted_data = converted_nd.data();
  CheckDataRegion(converted_data, dense_nd.data());
  }

  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  LOG(INFO) << "All pass";
}
