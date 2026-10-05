// Atomics of the Metal Shading Language.
// C++ for OpenCL predeclares atomic_int / atomic_uint / ... (even with -cl-no-stdinc), so the MSL
// functions are overloads on those types implemented with __opencl_atomic_* builtins. Object pointers are generic, so device and threadgroup atomics share one overload.
#pragma once

namespace metal {

enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST,
};

enum thread_scope {
  thread_scope_thread,
  thread_scope_simdgroup,
  thread_scope_threadgroup,
  thread_scope_device,
};

#define METAL_COMPAT_ATOMIC_SCOPE __OPENCL_MEMORY_SCOPE_DEVICE

#define METAL_COMPAT_ATOMIC_OPS(A, T)                                                                                 \
  METAL_FUNC void atomic_store_explicit(volatile A* object, T desired, memory_order order) {                         \
    __opencl_atomic_store(object, desired, order, METAL_COMPAT_ATOMIC_SCOPE);                                         \
  }                                                                                                                   \
  METAL_FUNC T atomic_load_explicit(volatile A* object, memory_order order) {                                        \
    return __opencl_atomic_load(object, order, METAL_COMPAT_ATOMIC_SCOPE);                                            \
  }                                                                                                                   \
  METAL_FUNC T atomic_exchange_explicit(volatile A* object, T desired, memory_order order) {                         \
    return __opencl_atomic_exchange(object, desired, order, METAL_COMPAT_ATOMIC_SCOPE);                               \
  }                                                                                                                   \
  METAL_FUNC bool atomic_compare_exchange_weak_explicit(volatile A* object, thread T* expected, T desired,           \
                                                        memory_order success, memory_order failure) {                 \
    return __opencl_atomic_compare_exchange_weak(object, expected, desired, success, failure,                        \
                                                 METAL_COMPAT_ATOMIC_SCOPE);                                          \
  }                                                                                                                   \
  METAL_FUNC bool atomic_compare_exchange_strong_explicit(volatile A* object, thread T* expected, T desired,         \
                                                          memory_order success, memory_order failure) {               \
    return __opencl_atomic_compare_exchange_strong(object, expected, desired, success, failure,                      \
                                                   METAL_COMPAT_ATOMIC_SCOPE);                                        \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_add_explicit(volatile A* object, T operand, memory_order order) {                        \
    return __opencl_atomic_fetch_add(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                              \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_sub_explicit(volatile A* object, T operand, memory_order order) {                        \
    return __opencl_atomic_fetch_sub(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                              \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_and_explicit(volatile A* object, T operand, memory_order order) {                        \
    return __opencl_atomic_fetch_and(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                              \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_or_explicit(volatile A* object, T operand, memory_order order) {                         \
    return __opencl_atomic_fetch_or(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                               \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_xor_explicit(volatile A* object, T operand, memory_order order) {                        \
    return __opencl_atomic_fetch_xor(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                              \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_min_explicit(volatile A* object, T operand, memory_order order) {                        \
    return __opencl_atomic_fetch_min(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                              \
  }                                                                                                                   \
  METAL_FUNC T atomic_fetch_max_explicit(volatile A* object, T operand, memory_order order) {                        \
    return __opencl_atomic_fetch_max(object, operand, order, METAL_COMPAT_ATOMIC_SCOPE);                              \
  }

METAL_COMPAT_ATOMIC_OPS(atomic_uint, uint)
METAL_COMPAT_ATOMIC_OPS(atomic_int, int)
#undef METAL_COMPAT_ATOMIC_OPS
METAL_FUNC void atomic_thread_fence(mem_flags flags, memory_order order, thread_scope scope = thread_scope_device) {
  (void)flags;
  (void)order;
  switch (scope) {
    case thread_scope_thread:
    case thread_scope_simdgroup:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
      break;
    case thread_scope_threadgroup:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
      break;
    case thread_scope_device:
      __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
      break;
  }
}

} // namespace metal
