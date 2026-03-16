#pragma once

#define METAL_CONST static constant constexpr
#define METAL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define METAL_SIMD_SIZE 32

METAL_CONST int MAX_REDUCE_SPECIALIZED_DIMS = 4;
METAL_CONST int REDUCE_N_READS = 4;
METAL_CONST int REDUCE_N_WRITES = 4;
METAL_CONST int SOFTMAX_N_READS = 4;
METAL_CONST int RMS_N_READS = 4;
METAL_CONST int RMS_LOOPED_LIMIT = 4096;

#define instantiate_kernel(name, func, ...)                                    \
  template [[host_name(                                                        \
      name                                                                     \
  )]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
