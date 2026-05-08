use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{DeviceExt, MatmulDispatchPath, Metal, MetalContext},
    },
};
use criterion::{BenchmarkId, Criterion, Throughput};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::{assert::assert_eq_float, type_short_name},
    uzu_bench, uzu_test,
};

struct Input<T: ArrayElement + Float> {
    a: Box<[T]>,
    b: Box<[T]>,
    m: usize,
    k: usize,
    n: usize,
}

fn get_test_data<T: ArrayElement + Float>(
    m: usize,
    k: usize,
    n: usize,
) -> (Input<T>, Vec<T>) {
    let a: Vec<T> = (0..m * k).map(|i| T::from(((i % 13) as f32) * 0.1 - 0.6).unwrap()).collect();
    let b: Vec<T> = (0..n * k).map(|i| T::from(((i % 17) as f32) * 0.1 - 0.8).unwrap()).collect();

    let input = Input {
        a: a.into_boxed_slice(),
        b: b.into_boxed_slice(),
        m,
        k,
        n,
    };

    let expected = cpu_reference::<T>(&input);
    (input, expected)
}

fn cpu_reference<T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = <Cpu as Backend>::Context::new().expect("Failed to create CPU context");

    let a_array = context.create_array_from(&[input.m, input.k], &input.a, "");
    let b_array = context.create_array_from(&[input.n, input.k], &input.b, "");
    let d_array = context.create_array_uninitialized(&[input.m, input.n], T::data_type(), "");

    let a_buf = a_array.buffer();
    let a_ref = a_buf.borrow();
    let b_buf = b_array.buffer();
    let b_ref = b_buf.borrow();
    let d_buf = d_array.buffer();
    let mut d_ref = d_buf.borrow_mut();

    let mut kernel = <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        &context,
        MatmulArguments {
            a: a_ref.deref(),
            a_offset: 0,
            b: b_ref.deref(),
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: d_ref.deref_mut(),
            batch_dim: input.m as u32,
            input_dim: input.k as u32,
            output_dim: input.n as u32,
        },
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    drop(d_ref);
    d_array.as_slice().to_vec()
}

fn run_unified_gemm<T: ArrayElement + Float>(
    input: &Input<T>,
    path: MatmulDispatchPath,
) -> Vec<T> {
    let context = MetalContext::new().expect("Failed to create Metal context");

    let a_array = context.create_array_from(&[input.m, input.k], &input.a, "");
    let b_array = context.create_array_from(&[input.n, input.k], &input.b, "");
    let d_array = context.create_array_uninitialized(&[input.m, input.n], T::data_type(), "");

    let a_buf = a_array.buffer();
    let a_ref = a_buf.borrow();
    let b_buf = b_array.buffer();
    let b_ref = b_buf.borrow();
    let d_buf = d_array.buffer();
    let mut d_ref = d_buf.borrow_mut();

    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");

    let mut encoder = Encoder::<Metal>::new(&context).unwrap();
    kernel.encode_with_path(
        &context,
        MatmulArguments {
            a: a_ref.deref(),
            a_offset: 0,
            b: b_ref.deref(),
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: d_ref.deref_mut(),
            batch_dim: input.m as u32,
            input_dim: input.k as u32,
            output_dim: input.n as u32,
        },
        &mut encoder,
        path,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    drop(d_ref);
    d_array.as_slice().to_vec()
}

fn test_simdgroup_mma<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) {
    let (input, expected) = get_test_data::<T>(m, k, n);
    let output = run_unified_gemm::<T>(&input, MatmulDispatchPath::UnifiedGemm);
    assert_eq_float(&expected, &output, eps, "unified gemm simdgroup");
}

fn test_mxu_mma<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) {
    let context = MetalContext::new().expect("Failed to create Metal context");
    if !context.device.supports_mxu() {
        eprintln!("Skipping MXU test: device does not support MXU");
        return;
    }
    let (input, expected) = get_test_data::<T>(m, k, n);
    let output = run_unified_gemm::<T>(&input, MatmulDispatchPath::UnifiedGemmMxuMma);
    assert_eq_float(&expected, &output, eps, "unified gemm mxu");
}

fn test_aligned<T: ArrayElement + Float + Debug + Display>(
    m: usize,
    k: usize,
    n: usize,
    eps: f32,
) {
    test_simdgroup_mma::<T>(m, k, n, eps);
}

#[uzu_test]
fn test_f32_aligned_64() {
    test_aligned::<f32>(64, 64, 64, 0.01);
}

#[uzu_test]
fn test_f16_aligned_64() {
    test_aligned::<f16>(64, 64, 64, 0.01);
}

#[uzu_test]
fn test_bf16_aligned_64() {
    test_aligned::<bf16>(64, 64, 64, 0.1);
}

#[uzu_test]
fn test_f32_large_aligned() {
    test_aligned::<f32>(128, 256, 128, 0.01);
}

#[uzu_test]
fn test_f16_large_aligned() {
    test_aligned::<f16>(128, 256, 128, 0.05);
}

#[uzu_test]
fn test_bf16_large_aligned() {
    test_aligned::<bf16>(128, 256, 128, 0.5);
}

#[uzu_test]
fn test_f32_unaligned() {
    test_aligned::<f32>(7, 33, 11, 0.01);
}

#[uzu_test]
fn test_f16_unaligned() {
    test_aligned::<f16>(7, 33, 11, 0.01);
}

#[uzu_test]
fn test_bf16_unaligned() {
    test_aligned::<bf16>(7, 33, 11, 0.1);
}

// MxuMma path (requires MXU-eligible hardware; runs on Apple Silicon GPUs that support metal_perf_primitives).

#[uzu_test]
fn test_mxu_f32_aligned_64() {
    test_mxu_mma::<f32>(64, 64, 64, 0.01);
}

#[uzu_test]
fn test_mxu_f16_aligned_64() {
    test_mxu_mma::<f16>(64, 64, 64, 0.05);
}

#[uzu_test]
fn test_mxu_bf16_aligned_64() {
    test_mxu_mma::<bf16>(64, 64, 64, 0.5);
}

#[uzu_test]
fn test_mxu_f32_large_aligned() {
    test_mxu_mma::<f32>(128, 256, 128, 0.01);
}

#[uzu_test]
fn test_mxu_f16_large_aligned() {
    test_mxu_mma::<f16>(128, 256, 128, 0.05);
}

// Benchmarks

const BENCHMARK_SHAPES: &[(usize, usize, usize)] =
    &[(128, 2048, 8192), (128, 4096, 14336), (256, 4096, 4096), (512, 8192, 2048)];

fn bench_unified_gemm_path(
    criterion: &mut Criterion,
    group_label: &str,
    path: MatmulDispatchPath,
) {
    let metal_context = MetalContext::new().unwrap();
    if matches!(path, MatmulDispatchPath::UnifiedGemmMxuMma) && !metal_context.device.supports_mxu() {
        return;
    }

    let mut matmul_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&metal_context, bf16::data_type())
            .expect("MatmulKernel");

    let mut benchmark_group =
        criterion.benchmark_group(format!("{}/Kernel/Matmul/{}", type_short_name::<Metal>(), group_label));

    for &(batch_dim, input_dim, output_dim) in BENCHMARK_SHAPES {
        let left_array = metal_context.create_array_uninitialized(&[batch_dim, input_dim], bf16::data_type(), "");
        let right_array = metal_context.create_array_uninitialized(&[output_dim, input_dim], bf16::data_type(), "");
        let destination_array =
            metal_context.create_array_uninitialized(&[batch_dim, output_dim], bf16::data_type(), "");

        let floating_point_operations = 2 * batch_dim * input_dim * output_dim;
        benchmark_group.throughput(Throughput::Elements(floating_point_operations as u64));

        benchmark_group.bench_function(
            BenchmarkId::new("BF16", format!("M[{batch_dim}]K[{input_dim}]N[{output_dim}]")),
            |bencher| {
                let left_buffer = left_array.buffer();
                let left_ref = left_buffer.borrow();
                let right_buffer = right_array.buffer();
                let right_ref = right_buffer.borrow();
                let destination_buffer = destination_array.buffer();

                bencher.iter_custom(|iteration_count| {
                    let mut encoder = Encoder::<Metal>::new(&metal_context).unwrap();

                    for _ in 0..iteration_count {
                        let mut destination_ref = destination_buffer.borrow_mut();
                        matmul_kernel.encode_with_path(
                            &metal_context,
                            MatmulArguments {
                                a: left_ref.deref(),
                                a_offset: 0,
                                b: right_ref.deref(),
                                ab_scale: 1.0,
                                c: MatmulArgumentC::None,
                                d: destination_ref.deref_mut(),
                                batch_dim: batch_dim as u32,
                                input_dim: input_dim as u32,
                                output_dim: output_dim as u32,
                            },
                            &mut encoder,
                            path,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            },
        );
    }
}

#[uzu_bench]
fn bench_unified_gemm(criterion: &mut Criterion) {
    bench_unified_gemm_path(criterion, "UNIFIED_GEMM", MatmulDispatchPath::UnifiedGemm);
}

#[uzu_bench]
fn bench_unified_gemm_mxu(criterion: &mut Criterion) {
    bench_unified_gemm_path(criterion, "UNIFIED_GEMM_MXU", MatmulDispatchPath::UnifiedGemmMxuMma);
}
