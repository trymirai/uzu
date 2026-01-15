#ifndef MLX_RS_CXX_UTILS_MATMUL_H_
#define MLX_RS_CXX_UTILS_MATMUL_H_

#include <chrono>
#include <cstring>
#include <vector>
#include "mlx/mlx.h"

namespace cxx_utils {

namespace mx = mlx::core;

struct BenchmarkResult {
  double avg_time_ms;
  double gflops;
};

// Compute matmul and copy result to output buffer for accuracy testing
// Input: A[M,K] row-major, B[K,N] row-major (but we'll transpose B for
// consistency with UZU) UZU expects: A[M,K], B^T[N,K] (transpose_b=true means B
// is stored as [N,K]) So we need to pass B as [N,K] to MLX too, then transpose
// it
inline void matmul_bf16_with_output(
    const uint16_t* a_data, // [M, K] row-major bf16 as raw bytes
    const uint16_t* b_data, // [N, K] row-major bf16 (transposed storage)
    uint16_t* output_data,  // [M, N] output
    int M,
    int K,
    int N
) {
  // Create MLX arrays from raw data
  // A is [M, K], B is stored as [N, K] (transposed)
  mx::Shape a_shape = {M, K};
  mx::Shape b_shape = {N, K};

  auto a = mx::array(
      reinterpret_cast<const mx::bfloat16_t*>(a_data),
      a_shape,
      mx::bfloat16
  );

  auto b_transposed = mx::array(
      reinterpret_cast<const mx::bfloat16_t*>(b_data),
      b_shape,
      mx::bfloat16
  );

  // Transpose B back to [K, N] for standard matmul
  auto b = mx::transpose(b_transposed);
  mx::eval(a, b);

  // Compute matmul: [M, K] @ [K, N] -> [M, N]
  auto result = mx::matmul(a, b);
  mx::eval(result);

  // Copy result to output buffer
  const auto* result_ptr = result.data<mx::bfloat16_t>();
  std::memcpy(output_data, result_ptr, M * N * sizeof(uint16_t));
}

inline void matmul_f16_with_output(
    const uint16_t* a_data,
    const uint16_t* b_data,
    uint16_t* output_data,
    int M,
    int K,
    int N
) {
  mx::Shape a_shape = {M, K};
  mx::Shape b_shape = {N, K};

  auto a = mx::array(
      reinterpret_cast<const mx::float16_t*>(a_data),
      a_shape,
      mx::float16
  );

  auto b_transposed = mx::array(
      reinterpret_cast<const mx::float16_t*>(b_data),
      b_shape,
      mx::float16
  );

  auto b = mx::transpose(b_transposed);
  mx::eval(a, b);

  auto result = mx::matmul(a, b);
  mx::eval(result);

  const auto* result_ptr = result.data<mx::float16_t>();
  std::memcpy(output_data, result_ptr, M * N * sizeof(uint16_t));
}

inline void matmul_f32_with_output(
    const float* a_data,
    const float* b_data,
    float* output_data,
    int M,
    int K,
    int N
) {
  mx::Shape a_shape = {M, K};
  mx::Shape b_shape = {N, K};

  auto a = mx::array(a_data, a_shape, mx::float32);
  auto b_transposed = mx::array(b_data, b_shape, mx::float32);

  auto b = mx::transpose(b_transposed);
  mx::eval(a, b);

  auto result = mx::matmul(a, b);
  mx::eval(result);

  const auto* result_ptr = result.data<float>();
  std::memcpy(output_data, result_ptr, M * N * sizeof(float));
}

inline BenchmarkResult benchmark_matmul_f16(
    int M,
    int N,
    int K,
    int warmup_iters,
    int bench_iters
) {
  auto a = mx::random::uniform({M, K}, mx::float16);
  // Match UZU weight layout: B is stored as [N, K] and viewed as [K, N] via
  // transpose
  auto b_transposed = mx::random::uniform({N, K}, mx::float16);
  auto b = mx::transpose(b_transposed);
  mx::eval(a, b);

  for (int i = 0; i < warmup_iters; ++i) {
    mx::eval(mx::matmul(a, b));
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < bench_iters; ++i) {
    mx::eval(mx::matmul(a, b));
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = total_ms / bench_iters;
  double flops = 2.0 * M * N * K;
  double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

  return BenchmarkResult{avg_ms, gflops};
}

inline BenchmarkResult benchmark_matmul_f32(
    int M,
    int N,
    int K,
    int warmup_iters,
    int bench_iters
) {
  auto a = mx::random::uniform({M, K}, mx::float32);
  // Match UZU weight layout: B is stored as [N, K] and viewed as [K, N] via
  // transpose
  auto b_transposed = mx::random::uniform({N, K}, mx::float32);
  auto b = mx::transpose(b_transposed);
  mx::eval(a, b);

  for (int i = 0; i < warmup_iters; ++i) {
    mx::eval(mx::matmul(a, b));
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < bench_iters; ++i) {
    mx::eval(mx::matmul(a, b));
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = total_ms / bench_iters;
  double flops = 2.0 * M * N * K;
  double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

  return BenchmarkResult{avg_ms, gflops};
}

inline BenchmarkResult benchmark_matmul_bf16(
    int M,
    int N,
    int K,
    int warmup_iters,
    int bench_iters
) {
  auto a = mx::random::uniform({M, K}, mx::bfloat16);
  // Match UZU weight layout: B is stored as [N, K] and viewed as [K, N] via
  // transpose
  auto b_transposed = mx::random::uniform({N, K}, mx::bfloat16);
  auto b = mx::transpose(b_transposed);
  mx::eval(a, b);

  for (int i = 0; i < warmup_iters; ++i) {
    mx::eval(mx::matmul(a, b));
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < bench_iters; ++i) {
    mx::eval(mx::matmul(a, b));
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double avg_ms = total_ms / bench_iters;
  double flops = 2.0 * M * N * K;
  double gflops = (flops / (avg_ms / 1000.0)) / 1e9;

  return BenchmarkResult{avg_ms, gflops};
}

} // namespace cxx_utils

#endif // MLX_RS_CXX_UTILS_MATMUL_H_
