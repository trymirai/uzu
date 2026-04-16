#![cfg(metal_backend)]

use std::{
    ops::Deref,
    ptr,
    rc::Rc,
};

use bytemuck;
use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::{
        common::{
            Allocation, AllocationType, Backend, Buffer, Context, Encoder,
            kernel::{
                ManualKernels,
                matmul::{MatmulArgumentC, MatmulArguments, MatmulKernel},
            },
        },
        cpu::Cpu,
        metal::{DeviceExt, MatmulDispatchPath, Metal, MetalContext},
    },
};

use crate::{
    common::{assert::assert_eq_float, type_short_name},
    uzu_bench, uzu_test,
};

struct Input {
    left: Box<[bf16]>,
    right: Box<[bf16]>,
    destination_prefill: Option<Box<[bf16]>>,
    batch_dim: usize,
    input_dim: usize,
    output_dim: usize,
    ab_scale: f32,
}

fn allocation_from_slice<T: ArrayElement, B: Backend>(
    context: &B::Context,
    data: &[T],
) -> Allocation<B> {
    let allocation = context
        .create_allocation(data.len() * std::mem::size_of::<T>(), AllocationType::Global)
        .expect("Failed to create allocation");
    let bytes = bytemuck::cast_slice(data);
    let (buffer, range) = allocation.as_buffer_range();
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), (buffer.cpu_ptr().as_ptr() as *mut u8).add(range.start), bytes.len());
    }
    allocation
}

fn get_test_data(
    batch_dim: usize,
    input_dim: usize,
    output_dim: usize,
    ab_scale: f32,
    accumulate: bool,
) -> Input {
    let left: Vec<bf16> = (0..batch_dim * input_dim).map(|i| bf16::from_f32(((i % 13) as f32) * 0.1 - 0.6)).collect();
    let right: Vec<bf16> = (0..output_dim * input_dim).map(|i| bf16::from_f32(((i % 17) as f32) * 0.1 - 0.8)).collect();
    let destination_prefill = if accumulate {
        Some((0..batch_dim * output_dim).map(|i| bf16::from_f32(((i % 7) as f32) * 0.03 - 0.09)).collect())
    } else {
        None
    };

    Input {
        left: left.into_boxed_slice(),
        right: right.into_boxed_slice(),
        destination_prefill,
        batch_dim,
        input_dim,
        output_dim,
        ab_scale,
    }
}

fn get_cpu_output(input: &Input) -> Vec<bf16> {
    let cpu_context = <Cpu as Backend>::Context::new().expect("CPU context");

    let right_array = cpu_context.create_array_from(&[input.output_dim, input.input_dim], &input.right, "");
    let left_allocation = allocation_from_slice::<bf16, Cpu>(&cpu_context, &input.left);
    let mut destination_allocation = if let Some(ref prefill) = input.destination_prefill {
        allocation_from_slice::<bf16, Cpu>(&cpu_context, prefill)
    } else {
        cpu_context
            .create_allocation(
                input.batch_dim * input.output_dim * std::mem::size_of::<bf16>(),
                AllocationType::Global,
            )
            .expect("Failed to create allocation")
    };

    let right_buffer = right_array.buffer();
    let right_ref = right_buffer.borrow();

    let argument_c = if input.destination_prefill.is_some() {
        MatmulArgumentC::Accumulate
    } else {
        MatmulArgumentC::None
    };

    let mut matmul_kernel =
        <<Cpu as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&cpu_context, bf16::data_type())
            .expect("CPU MatmulKernel");

    let mut encoder = Encoder::new(cpu_context.as_ref()).expect("encoder");
    matmul_kernel.encode(
        &cpu_context,
        MatmulArguments {
            a: &left_allocation,
            b: right_ref.deref(),
            ab_scale: input.ab_scale,
            c: argument_c,
            d: &mut destination_allocation,
            batch_dim: input.batch_dim as u32,
            input_dim: input.input_dim as u32,
            output_dim: input.output_dim as u32,
        },
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let (buffer, range) = destination_allocation.as_buffer_range();
    let bytes = unsafe {
        std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start), range.len())
    };
    bytemuck::cast_slice(bytes).to_vec()
}

fn get_mpp_output(
    metal_context: &MetalContext,
    matmul_kernel: &mut <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel,
    input: &Input,
) -> Vec<bf16> {
    let right_array = metal_context.create_array_from(&[input.output_dim, input.input_dim], &input.right, "");
    let left_allocation = allocation_from_slice::<bf16, Metal>(metal_context, &input.left);
    let mut destination_allocation = if let Some(ref prefill) = input.destination_prefill {
        allocation_from_slice::<bf16, Metal>(metal_context, prefill)
    } else {
        metal_context
            .create_allocation(
                input.batch_dim * input.output_dim * std::mem::size_of::<bf16>(),
                AllocationType::Global,
            )
            .expect("Failed to create allocation")
    };

    let right_buffer = right_array.buffer();
    let right_ref = right_buffer.borrow();

    let argument_c = if input.destination_prefill.is_some() {
        MatmulArgumentC::Accumulate
    } else {
        MatmulArgumentC::None
    };

    let mut encoder = Encoder::new(metal_context).expect("encoder");
    matmul_kernel.encode_with_path(
        metal_context,
        MatmulArguments {
            a: &left_allocation,
            b: right_ref.deref(),
            ab_scale: input.ab_scale,
            c: argument_c,
            d: &mut destination_allocation,
            batch_dim: input.batch_dim as u32,
            input_dim: input.input_dim as u32,
            output_dim: input.output_dim as u32,
        },
        &mut encoder,
        MatmulDispatchPath::GemmMpp,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let (buffer, range) = destination_allocation.as_buffer_range();
    let bytes = unsafe {
        std::slice::from_raw_parts((buffer.cpu_ptr().as_ptr() as *const u8).add(range.start), range.len())
    };
    bytemuck::cast_slice(bytes).to_vec()
}

fn create_metal_context_and_matmul_kernel()
-> (Rc<MetalContext>, <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel) {
    let metal_context = MetalContext::new().expect("Metal context");
    let matmul_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&metal_context, bf16::data_type())
            .expect("MatmulKernel");
    (metal_context, matmul_kernel)
}

fn test_mpp_correctness(
    batch_dim: usize,
    input_dim: usize,
    output_dim: usize,
    ab_scale: f32,
    accumulate: bool,
    tolerance: f32,
) {
    let (metal_context, mut matmul_kernel) = create_metal_context_and_matmul_kernel();
    if !metal_context.device.supports_mxu() {
        eprintln!("Skipping MPP test: device does not support MXU");
        return;
    }

    let input = get_test_data(batch_dim, input_dim, output_dim, ab_scale, accumulate);
    let expected = get_cpu_output(&input);
    let actual = get_mpp_output(&metal_context, &mut matmul_kernel, &input);
    assert_eq_float(&expected, &actual, tolerance, "GEMM_MPP vs CPU reference");
}

// Basic correctness — explicitly dispatched to MPP kernel

#[uzu_test]
fn test_bf16_aligned() {
    test_mpp_correctness(64, 2048, 2048, 1.0, false, 1.0);
}

#[uzu_test]
fn test_bf16_unaligned_m() {
    test_mpp_correctness(33, 2048, 2048, 1.0, false, 1.0);
}

#[uzu_test]
fn test_bf16_unaligned_n() {
    test_mpp_correctness(64, 2048, 33, 1.0, false, 1.0);
}

#[uzu_test]
fn test_bf16_large() {
    test_mpp_correctness(128, 4096, 14336, 1.0, false, 2.0);
}

#[uzu_test]
fn test_bf16_small_batch() {
    test_mpp_correctness(1, 2048, 2048, 1.0, false, 1.0);
}

// ab_scale

#[uzu_test]
fn test_bf16_ab_scale() {
    test_mpp_correctness(64, 2048, 2048, 0.5, false, 1.0);
}

// accumulate

#[uzu_test]
fn test_bf16_accumulate() {
    test_mpp_correctness(64, 2048, 2048, 1.0, true, 1.0);
}

// scale + accumulate combined

#[uzu_test]
fn test_bf16_scale_and_accumulate() {
    test_mpp_correctness(64, 2048, 2048, 0.5, true, 1.0);
}

// Benchmarks

const BENCHMARK_SHAPES: &[(usize, usize, usize)] =
    &[(128, 2048, 8192), (128, 4096, 14336), (256, 4096, 4096), (512, 8192, 2048)];

#[uzu_bench]
fn bench_gemm_mpp(criterion: &mut Criterion) {
    let metal_context = MetalContext::new().unwrap();
    if !metal_context.device.supports_mxu() {
        return;
    }

    let mut matmul_kernel =
        <<Metal as Backend>::Kernels as ManualKernels>::MatmulKernel::new(&metal_context, bf16::data_type())
            .expect("MatmulKernel");

    let mut benchmark_group =
        criterion.benchmark_group(format!("{}/Kernel/Matmul/GEMM_MPP", type_short_name::<Metal>()));

    for &(batch_dim, input_dim, output_dim) in BENCHMARK_SHAPES {
        let right_array = metal_context.create_array_uninitialized(&[output_dim, input_dim], bf16::data_type(), "");
        let left_allocation = metal_context
            .create_allocation(batch_dim * input_dim * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("Failed to create allocation");
        let mut destination_allocation = metal_context
            .create_allocation(batch_dim * output_dim * std::mem::size_of::<bf16>(), AllocationType::Global)
            .expect("Failed to create allocation");

        let floating_point_operations = 2 * batch_dim * input_dim * output_dim;
        benchmark_group.throughput(Throughput::Elements(floating_point_operations as u64));

        benchmark_group.bench_function(
            BenchmarkId::new("BF16", format!("M[{batch_dim}]K[{input_dim}]N[{output_dim}]")),
            |bencher| {
                let right_buffer = right_array.buffer();
                let right_ref = right_buffer.borrow();

                bencher.iter_custom(|iteration_count| {
                    let mut encoder = Encoder::<Metal>::new(&metal_context).unwrap();

                    for _ in 0..iteration_count {
                        matmul_kernel.encode_with_path(
                            &metal_context,
                            MatmulArguments {
                                a: &left_allocation,
                                b: right_ref.deref(),
                                ab_scale: 1.0,
                                c: MatmulArgumentC::None,
                                d: &mut destination_allocation,
                                batch_dim: batch_dim as u32,
                                input_dim: input_dim as u32,
                                output_dim: output_dim as u32,
                            },
                            &mut encoder,
                            MatmulDispatchPath::GemmMpp,
                        );
                    }

                    encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
                })
            },
        );
    }
}
