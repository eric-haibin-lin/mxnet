# TODO param c == 0.0, const, test case.. 

# A Guide to Implementing Sparse Operators in MXNet Backend

## Prerequisites
- Basic knowledge of [how to implement a dense operator in MXNet backend](https://mxnet.incubator.apache.org/versions/master/how_to/add_op_in_backend.html)
- Basic knowledge of [CSRNDArray](http://mxnet.incubator.apache.org/tutorials/sparse/csr.html) and [RowSparseNDArray](http://mxnet.incubator.apache.org/tutorials/sparse/row_sparse.html) in MXNet

## Introduction
In the [previous tutorial](https://mxnet.incubator.apache.org/versions/master/how_to/add_op_in_backend.html),
we went through the steps to implementing an operator using C++ in the MXNet backend.
In this tutorial, we will cover how sparse operators are implemented
in the backend. Specifically, we will practice adding CSRNDArray support to the forward function of the `quadratic` operator.

## Implementation
### A Sparse Operator Example

Let's consider the quadratic function `f(x) = ax^2+bx+c` when x is a CSRNDArray. 
Notice that if the input x is sparse and c is 0.0, the output is also sparse.
If c is non-zero, the output is dense. In MXNet frontend, the operator works like this:

```python
>>> x = mx.nd.array([[0,1],[2,0]).tostype('csr')
>>> x
<CSRNDArray 2x2 @cpu(0)>
>>> y = mx.nd.sparse.quadratic(x, a=1, b=2, c=0)
>>> y
<CSRNDArray 2x2 @cpu(0)>
>>> z = mx.nd.quadratic(x, a=1, b=2, c=3)
>>> z
[[  3.   6.]
 [ 11.   3.]]
<NDArray 2x2 @cpu(0)>
```

The statement `z = mx.nd.quadratic(x, a=1, b=2, c=3)` generates a warning message which says
the sparse input is converted to dense storage, and the dense operator is used to compute the dense output.
This is the "storage fallback" mechanism in MXNet, where a dense operator is automatically used for
inputs that a sparse operator doesn't have special kernels for.

In this tutorial, we will implement the forward function of the sparse quadratic operator.
The storage type of the output depends on the inputs:
- quadratic('csr', a, b, 0.0) outputs 'csr'
- otherwise, outputs 'default'

To implement this, we first register the storage type inference property of the operator, from which the operator
infers the output storage type based on operator arguments and inputs types. Then we implement the forward
function for the case where c is 0.0 and x is a CSRNDArray.

Next, we are going to

- Understand the FComputeEx and relevant NDArray interfaces in backend.
- Define storage type inference functions in quadratic_op-inl.h.
- Define the forward function in quadratic_op-inl.h.
- Register the sparse operator using nnvm in quadratic_op.cc and quadratic_op.cu for CPU and GPU computing, respectively.

Now let's walk through the process step by step.

### The FComputeEx and Relevant NDArray Interfaces in Backend

Before we dive into the details of relevant interfaces, here are two differences between
dense and sparse operators:
- Dense operators only handle dense inputs and outputs. Sparse operators support various combinations of
storage types.
- Memories of inputs and outputs are pre-allocated based their shapes for dense operators. However, with sparse representations, memories for sparse inputs and outputs depend on the number of non-zero elements they have,
which is only known at runtime.

With these differences in mind, let's review the `FCompute` interface introduced in the previous operator tutorial:
```cpp
void (const nnvm::NodeAttrs& attrs,
      const OpContext& ctx,
      const std::vector<TBlob>& inputs,
      const std::vector<OpReqType>& req,
      const std::vector<TBlob>& outputs);
```
Notice the `FCompute` interface doesn't include data structures that could be used to query storage
types of inputs, nor manipulate auxiliary arrays like `indices` and `indptr`. 
Therefore, instead of the `FCompute` interface, sparse operators are registered with the following `FComputeEx` interface:
```cpp
void (const nnvm::NodeAttrs& attrs,
      const OpContext& ctx,
      const std::vector<NDArray>& inputs,
      const std::vector<OpReqType>& req,
      const std::vector<NDArray>& outputs);
```
where the vectors of TBlobs are replaced with vectors of NDArrays. Now, let's go through a few important methods in the NDArray class.

In the python frontend, there are three types of NDArrays, namely `mx.nd.NDArray`, `mx.nd.sparse.RowSparseNDArray` and `mx.nd.sparse.CSRNDArray`. In the C++ backend, however, all of them are represented by the `mxnet::NDArray` class.
The `storage_type()` method indicates the storage type of the NDArray:
```cpp
enum NDArrayStorageType {
  kUndefinedStorage = -1,  // undefined storage
  kDefaultStorage,         // dense
  kRowSparseStorage,       // row sparse
  kCSRStorage,             // csr
};

// return the type of storage format
inline NDArrayStorageType storage_type() const;
```

On the other hand, from python one could inspect the auxiliary array of a sparse ndarray via
`RowSparseNDArray.indices`, `CSRNDArray.indices` and `CSRNDArray.indptr`, and the actual data array
via `RowSparseNDArray.data` and `CSRNDArray.data`.

In the backend, auxliary arrays such as `indices` and `indptr` are retrieved by
the `aux_data` method, while the actual data array is retrived by the 
`data` method.

```cpp
namespace csr {
enum CSRAuxType {kIndPtr, kIdx};
}

namespace rowsparse {
enum RowSparseAuxType {kIdx};
}
  
// return the i-th aux data TBlob
inline TBlob aux_data(size_t i) const;
  
// return the data TBlob
inline const TBlob& data() const;
```

Finally, the `CheckAndAlloc` method comes in handy when memory allocations for
the data and auxiliary arrays are needed for sparse NDArrays at run time.
```cpp
// allocate memory for non-default storage ndarrays based on auxliary array shapes
inline void CheckAndAlloc(const std::vector<TShape> &aux_shapes)
```

### Storage Type Inference
Storage type inference is the process of deducing storage types of `NDArray`s
in neural networks from operator arguments, and deciding whether to dispatch to
the `FCompute` or `FComputeEx` interface.
Let's take a look at the following example.
Given an input `CSRNDArray` called `x`, you invoke the `quadratic` operator
like this: `output = mx.nd.sparse.quadratic(x, a=1, b=2, c=0)`. Before calculating
the `output` values, MXNet infers the storage type of `output` to be `default`(dense),
and dispatch to `FComputeEx` operator implementation following the
the storage type inference rules you defined.

For our `quadratic` operator, the storage type inference function is the following.
Let's go through it line by line.
```cpp
inline bool QuadraticOpStorageType(const nnvm::NodeAttrs& attrs,                // 1
                                   const int dev_mask,                          // 2
                                   DispatchMode* dispatch_mode,                 // 3
                                   std::vector<int>* in_attrs,                  // 4
                                   std::vector<int>* out_attrs) {               // 5
  CHECK_EQ(in_attrs->size(), 1U);                                               // 6
  CHECK_EQ(out_attrs->size(), 1U);                                              // 7
  const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);        // 8
  const int in_stype = in_attrs->at(0);                                         // 9
  int& out_stype = out_attrs->at(0);                                            // 10
  bool dispatched = false;                                                      // 11
  if (!dispatched && in_stype == kDefaultStorage) {                             // 12
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,               // 13
                                     dispatch_mode, DispatchMode::kFCompute);   // 14
  }                                                                             // 15
  if (!dispatched && in_stype == kCSRStorage && param.c == 0.0) {               // 16
    dispatched = storage_type_assign(&out_stype, kCSRStorage,                   // 17
                                     dispatch_mode, DispatchMode::kFComputeEx); // 18
  }                                                                             // 19
  if (!dispatched) {                                                            // 20
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);                   // 21
  }                                                                             // 22
  return dispatched;                                                            // 23
}                                                                               // 24
```
- Line 1: `attrs` contains the parameters of the operator `a`, `b` and `c`.
- Line 2: `dev_mask` is the enum of device information of the operator such as
`Context::kCPU` and `Context::kGPU`. It is not used here since both contexts are supported.
- Line 3: `dispatch_mode` is the output dispatch mode for the operator. 
The initial value of `dispatch_mode` is `kUndefined`.
The types of dispatch mode include the following: 
```cpp
enum class DispatchMode {
  kUndefined = -1,
  // dispatch on FCompute interface
  kFCompute,
  // dispatch on FComputeEx interface
  kFComputeEx,
  // dispatch on FCompute interface with inputs / outputs converted to dense NDArrays
  kFComputeFallback,
  // special dispatch mode reserved for variables
  kVariable,
};
```
- Lines 4-5: `in_attrs` is a vector containing all input storage types.
`out_attrs` is a vector containing all output storage types.
- Lines 6-7: We check the number of inputs and that of outputs. Both should be equal to 1.
- Line 8: We get `QuadraticParam` from `attrs`. It contains the argument `c`, whose
value is used later to decide if the output is sparse.
- Lines 9-10: The storage type of the input is stored in the local varible `in_stype`.
The reference to output storage type is stored in the local varible `out_stype`.
- Line 11: The initialize the return value `dispatched` to `false`. 
- Lines 12-15: If the input is dense, try to assign dense storage to the output storage
type and assign `kFCompute` to `dispatch_mode`. 
The function `storage_type_assign()` first **attempts** to assign `kDefaultStorageType`
to `out_stype`. If the assignment to `out_stype` is successful
(i.e. `out_stype` was either not defined, or was already assigned with 
`kDefaultStorageType` previously), `storage_type_assign()` assigns `dispatch_mode`
to `kFCompute` and returns true; If the assignment to `out_stype` is not successful,
`dispatch_mode` keeps its old value and false is returned.
- Lines 16-19: If `dispatch_mode` is not defined, the input storage type is "csr"
and `c` is 0.0, try to assign csr storage to the output storage type and
assign `kFComputeEx` to `dispatch_mode`.
- Line 20-22: If `dispatch_mode` is still not defined, infer dense storage for the output
and dispatch to storage fallback mode. The `dispatch_fallback()` functions first attempts to
assign `kDefaultStorage` to all `out_attrs`. If the assignment is successful, return true;
otherwise, return false.
- Line 23: return the value of `dispatched`. If `dispatched` is false,
an exception will be thrown by MXNet.

### Forward Function
Let's go through the Forward function implementation line by line.

```cpp
template<typename xpu>                                                          // 1
void QuadraticOpForwardEx(const nnvm::NodeAttrs& attrs,                         // 2
                          const OpContext& ctx,                                 // 3
                          const std::vector<NDArray>& inputs,                   // 4
                          const std::vector<OpReqType>& req,                    // 5
                          const std::vector<NDArray>& outputs) {                // 6
  CHECK_EQ(inputs.size(), 1U);                                                  // 7
  CHECK_EQ(outputs.size(), 1U);                                                 // 8
  CHECK_EQ(req.size(), 1U);                                                     // 9
  const QuadraticParam& param = nnvm::get<QuadraticParam>(attrs.parsed);        // 10
  const auto in_stype = inputs[0].storage_type();                               // 11
  const auto out_stype = outputs[0].storage_type();                             // 12
  if (in_stype == kCSRStorage && out_stype == kCSRStorage && param.c == 0.0) {  // 13
    QuadraticOpForwardCsrImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0]);  // 14
  } else {                                                                      // 15
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);                       // 16
  }                                                                             // 17
}                                                                               // 18
                                                                                // 19
template<typename xpu>                                                          // 20
void QuadraticOpForwardCsrImpl(const QuadraticParam& param,                     // 21
                               const OpContext& ctx,                            // 22
                               const NDArray& input,                            // 23
                               const OpReqType req,                             // 24
                               const NDArray& output) {                         // 25
  using namespace mshadow;                                                      // 26
  using namespace mxnet_op;                                                     // 27
  using namespace csr;                                                          // 28
  if (req == kNullOp) return;                                                   // 29
  CHECK_EQ(req, kWriteTo) << "QuadraticOp with CSR only supports kWriteTo";     // 30
  Stream<xpu> *s = ctx.get_stream<xpu>();                                       // 31
  if (!input.storage_initialized()) {                                           // 32
    FillZerosCsrImpl(s, output);                                                // 33
    return;                                                                     // 34
  }                                                                             // 35
  const nnvm::dim_t nnz = input.storage_shape()[0];                             // 36
  const nnvm::dim_t num_rows = output.shape()[0];                               // 37
  output.CheckAndAlloc({Shape1(num_rows + 1), Shape1(nnz)});                    // 38
  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {                                  // 39
    MSHADOW_TYPE_SWITCH(output.aux_type(kIdx), CType, {                         // 40
      MSHADOW_TYPE_SWITCH(output.aux_type(kIndPtr), RType, {                    // 41
        MXNET_ASSIGN_REQ_SWITCH(req, req_type, {                                // 42
          Kernel<quadratic_forward<req_type>, xpu>::Launch(                     // 43
              s, nnz, output.data().dptr<DType>(), input.data().dptr<DType>(),  // 44
              param.a, param.b, param.c);                                       // 45
          Copy(output.aux_data(kIdx).FlatTo1D<xpu, CType>(),                    // 46
               input.aux_data(kIdx).FlatTo1D<xpu, CType>());                    // 47
          Copy(output.aux_data(kIndPtr).FlatTo1D<xpu, RType>(),                 // 48
               input.aux_data(kIndPtr).FlatTo1D<xpu, RType>());                 // 49
        });                                                                     // 50
      });                                                                       // 51
    });                                                                         // 52
  });                                                                           // 53
}                                                                               // 54

```

- Line 1-6: `inputs` is a vector of input NDArrays (only one input tensor
for the `quadratic` operator). `outputs` is a vector of output NDArrays
(only one for the `quadratic` operator). `xpu`, `attrs`, `ctx` and `req`
each holds the same thing introduced in the dense operator tutorial.
- Lines 7-9: Verify that the size of each vector is expected.
Otherwise, stop moving forward and print error message.
- Line 10: Get operator parameters, the input storage type and the output
storage type respectively. 
- Lines 13-18: If both the input storage type and the output storage type
are "csr" and c is 0.0, invoke the "csr" implementation. Otherwise,
an exception will be thrown with detailed information about the unimplemented
operator arguments.
- Lines 20-25: Function definition for the "csr" implementation of the `quadratic`
operator.
- Lines 26-28: Declare a few namespaces used in the current function scope.
Note that the `csr::kIdx` is for the access to the `indices` array of
all auxiliary arrays, while `csr::kIndPtr` is for the access to the `indptr`
array. 
- Line 29-30: Check the provided `req` of the operator. If `req` is `kNullOp`,
no work is required. Since the output of this operator is a "csr" NDArray,
whose memory has to be allocated at runtime, only `kWriteTo` is allowed.
Both `kAddTo` and `kWriteInplace` usually are not supported when the
output is sparse.
- Line 31: Get the `stream` of the context for serializing asynchronous executions.
- Lines 32-35: Before we access the `data`, `indices` and `indptr` arrays
to compute the result, we first check if these arrays are empty. If so,
we set the output to be zeros.
The `storage_initialized()` method returns true if a sparse NDArray
contains at least one element in its data and indices array; it returns false
otherwise.
- Line 36: Get the number of elements stored in the input and store
it in variable `nnz`. The `storage_shape()` method returns the shape of the
`data` array of a sparse NDArray.
- Line 37: Get the number of rows of the output and store it in variable `num_rows`.
- Line 38: Allocate memory for the data array and auxiliary arrays. For a CSRNDArray
of shape (M, N) storing K elements, it has a `data` array of length K, an `indices` array 
of length K and an `indptr` array of length (M + 1).
The `CheckAndAlloc` method takes the shape of auxiliary arrays as the input, and allocates
the memory for the data array and auxiliary arrays. It is not necessary to provie
the shape of the data array, as it can be inferred from shapes of auxilary arrays.
- Line 39-54: This is the place where the values of output data array and auxiliary arrays
are computed. 
The macros `MSHADOW_TYPE_SWITCH` and `MXNET_ASSIGN_REQ_SWITCH` enable
the code block to work for all the supported data types and `req` types in MXNet.
For this operator, since the transformation only happens on the data array,
we simply invoke the quadratic operator kernel via `Kernel::Launch`.
For the `indices` and `indptr` arrays, we just copy the values from the inputs.
This way, a complete output CSRNDArray is computed. 

### Operator Registration
Finally let's extend the operator registration logic to expose `sparse.quadratic`
to frontend. Below is the extended registration code in `quadratic_op.cc`:
```cpp
NNVM_REGISTER_OP(quadratic)
MXNET_ADD_SPARSE_OP_ALIAS(quadratic)
.describe(R"code(This operators implements the quadratic function:
.. math::
    f(x) = ax^2+bx+c
where :math:`x` is an input tensor and all operations
in the function are element-wise.
The storage type of ``quadratic`` output depends on storage types of inputs
  - quadratic(csr, a, b, 0) = csr
  - quadratic(default, a, b, c) = default
Example::
  x = [[1, 2], [3, 4]]
  y = quadratic(data=x, a=1, b=2, c=3)
  y = [[6, 11], [18, 27]]
)code" ADD_FILELINE)
.set_attr_parser(ParamParser<QuadraticParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuadraticOpShape)
.set_attr<nnvm::FInferType>("FInferType", QuadraticOpType)
.set_attr<FInferStorageType>("FInferStorageType", QuadraticOpStorageType)
.set_attr<FCompute>("FCompute<cpu>", QuadraticOpForward<cpu>)
.set_attr<FComputeEx>("FComputeEx<cpu>", QuadraticOpForwardEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_quadratic"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(QuadraticParam::__FIELDS__());
```

If you compare it with the original registration code,
only three lines of code are added to the above code block:
- MXNET_ADD_SPARSE_OP_ALIAS(quadratic)
This line adds an alias for the quadratic function in
python frontend so that `quadratic` is accessible from both `mx.symbol.sparse`
and `mx.ndarray.sparse`.
- .set_attr<FInferStorageType>("FInferStorageType", QuadraticOpStorageType)
This line register the storage type inference attribute of the operator.
- .set_attr<FComputeEx>("FComputeEx<cpu>", QuadraticOpForwardEx<cpu>)
This line register the `FComputeEx` attribute of the operator.

To register this sparse operator on GPU, `quadratic_op.cu` is extended
as below:
```cpp
NNVM_REGISTER_OP(quadratic)
.set_attr<FCompute>("FCompute<gpu>", QuadraticOpForward<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", QuadraticOpForwardEx<gpu>);
```

### Unit Test
To unit test the sparse operator in frontend, we need to add the following code
to the python file `test_sparse_ndarray.py`. 
```python
@with_seed()
def test_sparse_quadratic_function():
    def f(x, a, b, c):
        return a * x**2 + b * x + c

    def check_sparse_quadratic_function(c):
      # check forward and compare the result with dense op
      ndim = 2
      shape = rand_shape_nd(ndim, 5)
      data = rand_ndarray(shape=shape, stype='csr')
      data_np = data.asnumpy()
      expected = f(data_np, a, b, c)
      output = mx.nd.sparse.quadratic(data, a=a, b=b, c=c)
      assert(output.stype == expected_stype)
      assert_almost_equal(output.asnumpy(), expected)

    a = np.random.random_sample()
    b = np.random.random_sample()
    check_sparse_quadratic_function(0.0, 'csr')
    check_sparse_quadratic_function(1.0, 'default')

```

In this test, we are testing the result of the `sparse.quadratic` operator
on two cases:
- CSRNDArray input with c = 0.0, which outputs a CSRNDArray
- CSRNDArray input with c = 1.0, which outputs a NDArray

## Backward Function
So far, only the forward operator supports sparse inputs. To add sparse support to the
backward operator, you also need to register these two attributes to `_backward_quadratic`:
- `FComputeEx` for sparse operator implementation
- `FInferStorage` for storage tyep inference in backward
Due to length constraint, this is left as an exercise for readers.

## Summary
In this tutorial, we practiced adding sparse support to the operator `quadratic` in MXNet backend
and unit testing the implementation in frontend. More specifically, we went through a few
important interfaces, added the storage type inference function,
implemented the forward function, and registered the sparse operator
using nnvm. Congratulations! You now know how to add sparse operators.
We welcome your contributions to MXNet.
