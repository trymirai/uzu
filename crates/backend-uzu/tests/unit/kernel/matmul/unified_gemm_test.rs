use std::fmt::{Debug, Display};

use backend_uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            AllocationType, Backend, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{DeviceExt, MatmulDispatchPath, Metal, MetalContext},
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec},
    },
    uzu_test,
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

    let b_array = context.create_array_from(&[input.n, input.k], &input.b);
    let a_allocation = alloc_allocation_with_data::<Cpu, T>(&context, &input.a);
    let mut d_allocation = context
        .create_allocation(input.m * input.n * std::mem::size_of::<T>(), AllocationType::Global)
        .expect("Failed to create d allocation");

    let mut kernel = <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create CPU MatmulKernel");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        MatmulArguments {
            a: &a_allocation,
            a_offset: 0,
            b: b_array.allocation(),
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: &mut d_allocation,
            batch_dim: input.m as u32,
            input_dim: input.k as u32,
            output_dim: input.n as u32,
        },
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Cpu, T>(&d_allocation)
}

fn run_unified_gemm<T: ArrayElement + Float>(
    input: &Input<T>,
    path: MatmulDispatchPath,
) -> Vec<T> {
    let context = MetalContext::new().expect("Failed to create Metal context");

    let b_array = context.create_array_from(&[input.n, input.k], &input.b);
    let a_allocation = alloc_allocation_with_data::<Metal, T>(&context, &input.a);
    let mut d_allocation = context
        .create_allocation(input.m * input.n * std::mem::size_of::<T>(), AllocationType::Global)
        .expect("Failed to create d allocation");

    let mut kernel = <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&context, T::data_type())
        .expect("Failed to create MatmulKernel");

    let mut encoder = Encoder::<Metal>::new(&context).unwrap();
    kernel.encode_with_path(
        MatmulArguments {
            a: &a_allocation,
            a_offset: 0,
            b: b_array.allocation(),
            ab_scale: 1.0,
            c: MatmulArgumentC::None,
            d: &mut d_allocation,
            batch_dim: input.m as u32,
            input_dim: input.k as u32,
            output_dim: input.n as u32,
        },
        &mut encoder,
        path,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec::<Metal, T>(&d_allocation)
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
