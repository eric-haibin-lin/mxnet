#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
#include <cstdlib>

#include "../src/operator/tensor/elemwise_binary_op.h"
#include "../src/operator/tensor/elemwise_unary_op.h"
#include "../src/operator/optimizer_op-inl.h"
#include "../src/operator/tensor/init_op.h"

using namespace mxnet;
#define TEST_DTYPE float
#define TEST_ITYPE int32_t

void CheckDataRegion(const TBlob &src, const TBlob &dst) {
  auto size = src.shape_.Size() * mshadow::mshadow_sizeof(src.type_flag_);
  auto equals = memcmp(src.dptr_, dst.dptr_, size);
  EXPECT_EQ(equals, 0);
}

float RandFloat() {
  float v = rand() * 1.0 / RAND_MAX;
  return v;
}

NDArray RspIdxND(const TShape shape, const Context ctx, const std::vector<TEST_ITYPE> &values) {
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

NDArray DnsND(const TShape shape, const Context ctx, std::vector<TEST_DTYPE> vs) {
  NDArray nd(shape, ctx, false);
  size_t num_val = shape.Size();
  while (vs.size() < num_val) {
    auto v = RandFloat();
    vs.push_back(v);
  }
  CHECK_EQ(vs.size(), nd.shape().Size());
  MSHADOW_TYPE_SWITCH(nd.dtype(), DType, {
    auto tensor = nd.data().FlatTo1D<cpu, DType>();
    for (size_t i = 0; i < num_val; i++) {
      tensor[i] = vs[i];
    }
  });
  return nd;
}

NDArray RspND(const TShape shape, const Context ctx, const std::vector<TEST_ITYPE> idx,
              std::vector<TEST_DTYPE> vals) {
  index_t num_rows = idx.size();
  index_t num_cols = vals.size() / idx.size();
  NDArray index = RspIdxND(TShape({num_rows}), ctx, idx);
  CHECK_EQ(vals.size() % idx.size(), 0);
  NDArray raw_data = DnsND(TShape({num_rows, num_cols}), ctx, vals);
  NDArray nd(raw_data, {index}, ctx, kRowSparseStorage, shape);
  return nd;
}

// TODO(haibin) support other types
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
