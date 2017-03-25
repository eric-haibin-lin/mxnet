//#include <time.h>
#include <unistd.h>
#include <dmlc/logging.h>
#include <cstdio>
#include <gtest/gtest.h>
//#include <thread>
//#include <chrono>
#include <vector>

#include <mxnet/engine.h>
#include <mxnet/ndarray.h>
//#include "../src/engine/engine_impl.h"
//#include <dmlc/timer.h>
using namespace mxnet;

void BasicTest() {

  Context ctx;
  const real_t val = 9;

  TShape shape2(2);
  shape2[0] = 1;
  shape2[1] = 2;
  NDArray nd2(shape2, ctx);
  // alloc one row
  nd2.CheckAndAlloc();
  EXPECT_NE(nd2.data().dptr_, nullptr);
  nd2 = val;
  Engine::Get()->WaitForAll();



  TShape shape(2);
  shape[0] = 1;
  shape[1] = 2;
  //What to do when creating a copy/reference?
  NDArray nd1(RowSparseChunk, shape, ctx);
  nd1.CheckAndAlloc(1);
  EXPECT_NE(nd1.data().dptr_, nullptr);
  nd1 = val;
  Engine::Get()->WaitForAll();                                                              
  
  auto size = nd1.data().shape_.Size() * mshadow::mshadow_sizeof(nd1.data().type_flag_);
  auto n = memcmp(nd1.data().dptr_, nd2.data().dptr_, size);
  EXPECT_EQ(n, 0);
}

TEST(NDArray, basics) {
  BasicTest();
  //Wait for all operations to finish
  Engine::Get()->WaitForAll();
  LOG(INFO) << "All pass";
}
