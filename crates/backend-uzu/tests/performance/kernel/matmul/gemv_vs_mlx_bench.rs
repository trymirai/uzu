//! Head-to-head GEMV perf: uzu Metal kernels vs Apple MLX (`pmetal-mlx-rs`).
//! Same shapes through both engines for FP bf16 and 4-bit affine quant, under a
//! symmetric amortized measurement (CHUNK GEMVs per GPU submission, so per-op
//! kernel throughput dominates submission overhead). macOS-only (MLX dep).
//! Methodology: docs/gemv-quant-simdgroups.md.
#![cfg(all(metal_backend, target_os = "macos"))]

use std::time::{Duration, Instant};

use backend_uzu::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Encoder,
            gpu_types::QuantizationMethod,
            kernel::{
                Kernels,
                matmul::{MatmulArguments, MatmulB, MatmulDOps, MatmulKernel},
            },
        },
        metal::{Metal, MetalContext},
    },
};
use criterion::{Bencher, BenchmarkId, Criterion, Throughput};
use half::bf16;
use pmetal_mlx_rs as mlx;
use rand::{RngExt, SeedableRng, rngs::SmallRng};

use crate::{
    common::{
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::{QuantBuffers, QuantInput, Shape, bench_quant_gemv_shapes, quant_arguments},
        shared_metal_context,
    },
    uzu_bench,
};

const QUANT_BITS: u32 = 4;
const QUANT_GROUP_SIZE: u32 = 64;
/// GEMVs per GPU submission. Amortizes the fixed submit/sync cost so the
/// kernel dominates; bounded so command buffers and the MLX graph stay small.
const CHUNK: u64 = 64;

fn random_bf16(
    len: usize,
    seed: u64,
) -> Vec<bf16> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..len).map(|_| bf16::from_f32(rng.random_range(-0.3f32..0.3f32))).collect()
}

/// Drives criterion's `iter_custom` by issuing `submit_n(n)` calls, each of
/// which enqueues `n` GEMVs in one submission and blocks until done. The work
/// is chunked at [`CHUNK`] so every submission carries the same amortizable
/// overhead, identically for uzu and MLX. `submit_n` is warmed once first so
/// kernel/graph compilation is excluded from the timed region.
fn bench_chunked<F: FnMut(u64)>(
    b: &mut Bencher,
    mut submit_n: F,
) {
    submit_n(1);
    b.iter_custom(|iters| {
        let start = Instant::now();
        let mut done = 0u64;
        while done < iters {
            let n = (iters - done).min(CHUNK);
            submit_n(n);
            done += n;
        }
        start.elapsed().max(Duration::from_nanos(1))
    });
}

// ---- Full-precision bf16 GEMV ------------------------------------------------

fn bench_fp(
    c: &mut Criterion,
    context: &MetalContext,
) {
    let mut group = c.benchmark_group("Mlx/Kernel/Gemv/FP_BF16");

    for shape in bench_quant_gemv_shapes(QUANT_BITS) {
        let Shape {
            m,
            k,
            n,
        } = shape;
        group.throughput(Throughput::Elements((m * n * k) as u64));

        let x_data = random_bf16(m * k, 1);
        let w_data = random_bf16(n * k, 2);

        // --- uzu ---
        let x = alloc_allocation_with_data::<Metal, bf16>(context, &x_data);
        let w = alloc_allocation_with_data::<Metal, bf16>(context, &w_data);
        let mut y = alloc_allocation::<Metal, bf16>(context, m * n);
        let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            bf16::data_type(),
        )
        .expect("MatmulKernel");

        group.bench_function(BenchmarkId::new("uzu", shape), |b| {
            bench_chunked(b, |reps| {
                let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
                for _ in 0..reps {
                    kernel
                        .encode(
                            MatmulArguments {
                                a: &x,
                                a_offset: 0,
                                b: MatmulB::FullPrecision {
                                    b: &w,
                                },
                                b_offset: 0,
                                b_leading_dimension: None,
                                b_transpose: true,
                                d: &mut y,
                                d_transform: MatmulDOps::none(),
                                m: m as u32,
                                n: n as u32,
                                k: k as u32,
                            },
                            &mut encoder,
                        )
                        .expect("uzu fp gemv encode");
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
        });

        // --- mlx ---
        let mx = mlx::Array::from_slice(&x_data, &[m as i32, k as i32]);
        let mw = mlx::Array::from_slice(&w_data, &[n as i32, k as i32]);
        mlx::transforms::eval([&mx, &mw]).expect("mlx materialize inputs");

        group.bench_function(BenchmarkId::new("mlx", shape), |b| {
            bench_chunked(b, |reps| {
                // x[m,k] @ w[n,k]^T = [m,n], matching uzu's b_transpose=true.
                let ys: Vec<mlx::Array> =
                    (0..reps).map(|_| mlx::ops::matmul(&mx, mw.t()).expect("mlx matmul")).collect();
                mlx::transforms::eval(ys.iter()).expect("mlx eval");
            });
        });
    }

    group.finish();
}

// ---- 4-bit affine (scale+bias) GEMV -----------------------------------------

fn bench_quant(
    c: &mut Criterion,
    context: &MetalContext,
) {
    let mut group = c.benchmark_group("Mlx/Kernel/Gemv/Quant4_BF16_gs64");

    for shape in bench_quant_gemv_shapes(QUANT_BITS) {
        let Shape {
            m,
            k,
            n,
        } = shape;
        group.throughput(Throughput::Elements((m * n * k) as u64));

        // --- uzu ---
        let input = QuantInput::<bf16>::new(m, k, n, QUANT_GROUP_SIZE, QUANT_BITS, QuantizationMethod::ScaleBias, 42);
        let mut buffers = QuantBuffers::<Metal, bf16>::allocate(context, &input);
        let mut kernel = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            context,
            bf16::data_type(),
            bf16::data_type(),
            bf16::data_type(),
        )
        .expect("MatmulKernel");

        group.bench_function(BenchmarkId::new("uzu", shape), |b| {
            bench_chunked(b, |reps| {
                let mut encoder = Encoder::<Metal>::new(context).expect("encoder");
                for _ in 0..reps {
                    kernel.encode(quant_arguments(&mut buffers, &input), &mut encoder).expect("uzu quant gemv encode");
                }
                encoder.end_encoding().submit().wait_until_completed().unwrap();
            });
        });

        // --- mlx ---
        // Quantize a bf16 weight [n, k] with MLX's default affine (scale+bias)
        // scheme, then dispatch with transpose=true so it computes x @ W^T.
        let w_data = random_bf16(n * k, 7);
        let x_data = random_bf16(m * k, 8);
        let mw = mlx::Array::from_slice(&w_data, &[n as i32, k as i32]);
        let (mw_q, scales, biases) =
            mlx::ops::quantize(&mw, QUANT_GROUP_SIZE as i32, QUANT_BITS as i32).expect("mlx quantize");
        let mx = mlx::Array::from_slice(&x_data, &[m as i32, k as i32]);
        mlx::transforms::eval([&mx, &mw_q, &scales, &biases]).expect("mlx materialize inputs");

        group.bench_function(BenchmarkId::new("mlx", shape), |b| {
            bench_chunked(b, |reps| {
                let ys: Vec<mlx::Array> = (0..reps)
                    .map(|_| {
                        mlx::ops::quantized_matmul(
                            &mx,
                            &mw_q,
                            &scales,
                            Some(&biases),
                            true,
                            QUANT_GROUP_SIZE as i32,
                            QUANT_BITS as i32,
                        )
                        .expect("mlx quantized_matmul")
                    })
                    .collect();
                mlx::transforms::eval(ys.iter()).expect("mlx eval");
            });
        });
    }

    group.finish();
}

#[uzu_bench]
fn bench_gemv_vs_mlx(c: &mut Criterion) {
    mlx::Device::set_default(&mlx::Device::gpu());
    let context = shared_metal_context();
    bench_fp(c, &context);
    bench_quant(c, &context);
}
