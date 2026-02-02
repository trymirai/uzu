#ifndef definitions_metal
#define definitions_metal

#include <metal_stdlib>

using namespace metal;

// MARK: - Tensor Access Templates

template <typename T>
struct TensorView2D {
  device T* data;
  uint dim0, dim1;
  uint stride0, stride1;

  TensorView2D(device T* ptr) : data(ptr) {}

  TensorView2D(device T* ptr, int d0, int d1) : data(ptr), dim0(d0), dim1(d1) {
    stride0 = d1;
    stride1 = 1;
  }

  // Custom stride constructor
  TensorView2D(device T* ptr, int d0, int d1, int s0, int s1)
      : data(ptr), dim0(d0), dim1(d1), stride0(s0), stride1(s1) {}

  thread TensorView2D& shaped(int d0, int d1) {
    dim0 = d0;
    dim1 = d1;
    stride0 = d1;
    stride1 = 1;
    return *this;
  }

  device T& at(uint i, uint j) const { return data[i * stride0 + j * stride1]; }

  device T& operator()(uint i, uint j) const { return at(i, j); }
};

template <typename T>
struct TensorView3D {
  device T* data;
  uint dim0, dim1, dim2;
  uint stride0, stride1, stride2;

  TensorView3D(device T* ptr) : data(ptr) {}

  TensorView3D(device T* ptr, int d0, int d1, int d2)
      : data(ptr), dim0(d0), dim1(d1), dim2(d2) {
    stride0 = d1 * d2;
    stride1 = d2;
    stride2 = 1;
  }

  // Custom stride constructor
  TensorView3D(device T* ptr, int d0, int d1, int d2, int s0, int s1, int s2)
      : data(ptr), dim0(d0), dim1(d1), dim2(d2), stride0(s0), stride1(s1),
        stride2(s2) {}

  thread TensorView3D& shaped(int d0, int d1, int d2) {
    dim0 = d0;
    dim1 = d1;
    dim2 = d2;
    stride0 = d1 * d2;
    stride1 = d2;
    stride2 = 1;
    return *this;
  }

  device T& at(uint i, uint j, uint k) const {
    return data[i * stride0 + j * stride1 + k * stride2];
  }

  device T& operator()(uint i, uint j, uint k) const { return at(i, j, k); }
};

template <typename T>
struct TensorView4D {
  device T* data;
  uint dim0, dim1, dim2, dim3;
  uint stride0, stride1, stride2, stride3;

  TensorView4D(device T* ptr) : data(ptr) {}

  TensorView4D(device T* ptr, int d0, int d1, int d2, int d3)
      : data(ptr), dim0(d0), dim1(d1), dim2(d2), dim3(d3) {
    stride0 = d1 * d2 * d3;
    stride1 = d2 * d3;
    stride2 = d3;
    stride3 = 1;
  }

  // Custom stride constructor
  TensorView4D(
      device T* ptr,
      int d0,
      int d1,
      int d2,
      int d3,
      int s0,
      int s1,
      int s2,
      int s3
  )
      : data(ptr), dim0(d0), dim1(d1), dim2(d2), dim3(d3), stride0(s0),
        stride1(s1), stride2(s2), stride3(s3) {}

  thread TensorView4D& shaped(int d0, int d1, int d2, int d3) {
    dim0 = d0;
    dim1 = d1;
    dim2 = d2;
    dim3 = d3;
    stride0 = d1 * d2 * d3;
    stride1 = d2 * d3;
    stride2 = d3;
    stride3 = 1;
    return *this;
  }

  device T& at(uint i, uint j, uint k, uint l) const {
    return data[i * stride0 + j * stride1 + k * stride2 + l * stride3];
  }

  device T& operator()(uint i, uint j, uint k, uint l) const {
    return at(i, j, k, l);
  }
};

///////////////////////////////////////////////////////////////////////////////
//  MARK: - Thread Functions
///////////////////////////////////////////////////////////////////////////////

template <ushort LENGTH, typename T>
static inline T thread_prefix_inclusive_sum(thread T (&values)[LENGTH]) {
  for (ushort i = 1; i < LENGTH; i++) {
    values[i] += values[i - 1];
  }
  return values[LENGTH - 1];
}

template <ushort LENGTH, typename T>
static inline T thread_prefix_inclusive_sum(threadgroup T* values) {
  for (ushort i = 1; i < LENGTH; i++) {
    values[i] += values[i - 1];
  }
  return values[LENGTH - 1];
}

template <ushort LENGTH, typename T>
static inline T thread_prefix_exclusive_sum(thread T (&values)[LENGTH]) {
  //  do as an inclusive scan first
  T inclusive_prefix = thread_prefix_inclusive_sum<LENGTH>(values);
  //  convert to an exclusive scan in the reverse direction
  for (ushort i = LENGTH - 1; i > 0; i--) {
    values[i] = values[i - 1];
  }
  values[0] = 0;
  return inclusive_prefix;
}

template <ushort LENGTH, typename T>
static inline T thread_prefix_exclusive_sum(threadgroup T* values) {
  //  do as an inclusive scan first
  T inclusive_prefix = thread_prefix_inclusive_sum<LENGTH>(values);
  //  convert to an exclusive scan in the reverse direction
  for (ushort i = LENGTH - 1; i > 0; i--) {
    values[i] = values[i - 1];
  }
  values[0] = 0;
  return inclusive_prefix;
}

template <ushort LENGTH, typename T>
static inline void thread_uniform_add(thread T (&values)[LENGTH], T uni) {
  for (ushort i = 0; i < LENGTH; i++) {
    values[i] += uni;
  }
}

template <ushort LENGTH, typename T>
static inline void thread_uniform_add(threadgroup T* values, T uni) {
  for (ushort i = 0; i < LENGTH; i++) {
    values[i] += uni;
  }
}

///////////////////////////////////////////////////////////////////////////////
//  MARK: - Threadgroup Functions
///////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------------------------------//
//  Raking threadgroup scan
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_prefix_exclusive_sum(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  // load values into shared memory
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  //  only the first 32 threads form the rake
  if (lid < 32) {
    //  scan by thread in shared mem
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      shared[i] += shared[i - 1];
    }
    T partial_sum = shared[first_index + values_per_thread - 1];
    for (short i = first_index + values_per_thread - 1; i > first_index; i--) {
      shared[i] = shared[i - 1];
    }
    shared[first_index] = 0;

    //  scan the partial sums
    T prefix = simd_prefix_exclusive_sum(partial_sum);

    // add back the prefix
    for (short i = first_index; i < first_index + values_per_thread; i++) {
      shared[i] += prefix;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[lid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

//------------------------------------------------------------------------------------------------//
//  Raking threadgroup sum reduction
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_reduce_sum(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  // load values into shared memory
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  //  only the first 32 threads form the rake
  if (lid < 32) {
    //  reduce by thread in shared mem
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;

    T thread_sum = shared[first_index];
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_sum += shared[i];
    }

    //  reduce the partial sums using SIMD
    T total_sum = simd_sum(thread_sum);

    // broadcast result back to shared memory
    if (lid == 0) {
      shared[0] = total_sum;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

//------------------------------------------------------------------------------------------------//
//  Raking threadgroup max reduction
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_reduce_max(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  // load values into shared memory
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  //  only the first 32 threads form the rake
  if (lid < 32) {
    //  reduce by thread in shared mem
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;

    T thread_max = shared[first_index];
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_max = max(thread_max, shared[i]);
    }

    //  reduce the partial maxes using SIMD
    T total_max = simd_max(thread_max);

    // broadcast result back to shared memory
    if (lid == 0) {
      shared[0] = total_max;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

//------------------------------------------------------------------------------------------------//
//  Raking threadgroup min reduction
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_raking_reduce_min(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  // load values into shared memory
  shared[lid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  //  only the first 32 threads form the rake
  if (lid < 32) {
    //  reduce by thread in shared mem
    const short values_per_thread = BLOCK_SIZE / 32;
    const short first_index = lid * values_per_thread;

    T thread_min = shared[first_index];
    for (short i = first_index + 1; i < first_index + values_per_thread; i++) {
      thread_min = min(thread_min, shared[i]);
    }

    //  reduce the partial mins using SIMD
    T total_min = simd_min(thread_min);

    // broadcast result back to shared memory
    if (lid == 0) {
      shared[0] = total_min;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

//------------------------------------------------------------------------------------------------//
//  Cooperative threadgroup sum reduction (2-level hierarchical)
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_cooperative_reduce_sum(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  const ushort simd_group_id = lid / 32;
  const ushort simd_lane_id = lid % 32;

  // Reduce within simdgroup
  T local_sum = simd_sum(value);

  // First thread in each simdgroup writes to shared memory
  if (simd_lane_id == 0) {
    shared[simd_group_id] = local_sum;
  }

  // Synchronize across the threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce across simdgroups
  T total_sum = T(0);
  const ushort num_simd_groups = (BLOCK_SIZE + 31) / 32;
  if (lid < num_simd_groups) {
    total_sum = shared[lid];
  }
  total_sum = simd_sum(total_sum);

  // Broadcast the result to all threads
  if (lid == 0) {
    shared[0] = total_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

//------------------------------------------------------------------------------------------------//
//  Cooperative threadgroup max reduction (2-level hierarchical)
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_cooperative_reduce_max(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  const ushort simd_group_id = lid / 32;
  const ushort simd_lane_id = lid % 32;

  // Reduce within simdgroup
  T local_max = simd_max(value);

  // First thread in each simdgroup writes to shared memory
  if (simd_lane_id == 0) {
    shared[simd_group_id] = local_max;
  }

  // Synchronize across the threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce across simdgroups
  T total_max = (lid < ((BLOCK_SIZE + 31) / 32)) ? shared[lid] : T(-INFINITY);
  total_max = simd_max(total_max);

  // Broadcast the result to all threads
  if (lid == 0) {
    shared[0] = total_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

//------------------------------------------------------------------------------------------------//
//  Cooperative threadgroup min reduction (2-level hierarchical)
template <ushort BLOCK_SIZE, typename T>
static T threadgroup_cooperative_reduce_min(
    T value,
    threadgroup T* shared,
    const ushort lid
) {
  const ushort simd_group_id = lid / 32;
  const ushort simd_lane_id = lid % 32;

  // Reduce within simdgroup
  T local_min = simd_min(value);

  // First thread in each simdgroup writes to shared memory
  if (simd_lane_id == 0) {
    shared[simd_group_id] = local_min;
  }

  // Synchronize across the threadgroup
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce across simdgroups
  T total_min = (lid < ((BLOCK_SIZE + 31) / 32)) ? shared[lid] : T(INFINITY);
  total_min = simd_min(total_min);

  // Broadcast the result to all threads
  if (lid == 0) {
    shared[0] = total_min;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const T result = shared[0];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}

// warning: constexpr if is a C++17 extension [-Wc++17-extensions]
#if defined(__cpp_if_constexpr)
  #define IF_CONSTEXPR(cond) if constexpr (cond)
#else
  #define IF_CONSTEXPR(cond) if (cond)
#endif

// MARK: - DSL Annotation Helpers

#ifdef DSL_ANALYZE
#define DSL_STRUCT struct [[clang::annotate("dsl.struct")]]
#else
#define DSL_STRUCT struct
#endif

#ifdef DSL_ANALYZE
#define DSL_META(...) [[clang::annotate("", __VA_ARGS__)]]
#else
#define DSL_META(...)
#endif

#define DSL_STR(X) #X
#define DSL_XSTR(X) DSL_STR(X)

#define VARIANTS(TYPENAME, ...) DSL_META("dsl.variants", #TYPENAME, #__VA_ARGS__)
#define KERNEL(NAME) DSL_META("dsl.kernel") void NAME

#define SPECIALIZE DSL_META("dsl.specialize")

#define AXIS(TDS, TPG) DSL_META("dsl.axis", DSL_XSTR(TDS), DSL_XSTR(TPG))
#define GROUPS(EXPR) DSL_META("dsl.groups", DSL_XSTR(EXPR))
#define THREADS(EXPR) DSL_META("dsl.threads", DSL_XSTR(EXPR))

// MARK: - Generate Template Kernels

#define generateKernel(max_threads, functionName, scalarType, outerArgs, innerArgs) \
  [[max_total_threads_per_threadgroup(max_threads)]]                           \
  kernel void functionName##_##scalarType outerArgs { functionName innerArgs; }

#define generateKernels(max_threads, functionName)                             \
  generateKernel(max_threads, functionName, float, outerArguments(float), innerArguments);  \
  generateKernel(                                                              \
      max_threads,                                                             \
      functionName,                                                            \
      bfloat,                                                                  \
      outerArguments(bfloat),                                                  \
      innerArguments                                                           \
  );                                                                           \
  generateKernel(max_threads, functionName, half, outerArguments(half), innerArguments);

#endif /* definitions_metal */
