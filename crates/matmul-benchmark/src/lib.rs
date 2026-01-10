#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unused_imports)]

use autocxx::prelude::*;

include_cpp! {
    #include "mlx/mlx.h"
    #include "cxx_utils.hpp"
    safety!(unsafe_ffi)

    // Benchmark utilities
    generate_pod!("cxx_utils::BenchmarkResult")
    generate!("cxx_utils::benchmark_matmul_f16")
    generate!("cxx_utils::benchmark_matmul_f32")
    generate!("cxx_utils::benchmark_matmul_bf16")
    generate!("cxx_utils::matmul_f16_with_output")
    generate!("cxx_utils::matmul_f32_with_output")
    generate!("cxx_utils::matmul_bf16_with_output")
}

pub use ffi::cxx_utils::BenchmarkResult;

pub fn benchmark_matmul_f16(
    m: i32,
    n: i32,
    k: i32,
    warmup: i32,
    iters: i32,
) -> BenchmarkResult {
    ffi::cxx_utils::benchmark_matmul_f16(
        c_int(m),
        c_int(n),
        c_int(k),
        c_int(warmup),
        c_int(iters),
    )
}

pub fn benchmark_matmul_f32(
    m: i32,
    n: i32,
    k: i32,
    warmup: i32,
    iters: i32,
) -> BenchmarkResult {
    ffi::cxx_utils::benchmark_matmul_f32(
        c_int(m),
        c_int(n),
        c_int(k),
        c_int(warmup),
        c_int(iters),
    )
}

pub fn benchmark_matmul_bf16(
    m: i32,
    n: i32,
    k: i32,
    warmup: i32,
    iters: i32,
) -> BenchmarkResult {
    ffi::cxx_utils::benchmark_matmul_bf16(
        c_int(m),
        c_int(n),
        c_int(k),
        c_int(warmup),
        c_int(iters),
    )
}

/// Compute matmul using MLX and return result for accuracy comparison
/// A: [m, k] row-major, B: [n, k] row-major (transposed storage)
/// Output: [m, n]
pub fn matmul_f16_with_output(
    a: &[u16],
    b: &[u16],
    output: &mut [u16],
    m: i32,
    k: i32,
    n: i32,
) {
    unsafe {
        ffi::cxx_utils::matmul_f16_with_output(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            c_int(m),
            c_int(k),
            c_int(n),
        );
    }
}

pub fn matmul_f32_with_output(
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
    m: i32,
    k: i32,
    n: i32,
) {
    unsafe {
        ffi::cxx_utils::matmul_f32_with_output(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            c_int(m),
            c_int(k),
            c_int(n),
        );
    }
}

pub fn matmul_bf16_with_output(
    a: &[u16],
    b: &[u16],
    output: &mut [u16],
    m: i32,
    k: i32,
    n: i32,
) {
    unsafe {
        ffi::cxx_utils::matmul_bf16_with_output(
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            c_int(m),
            c_int(k),
            c_int(n),
        );
    }
}
