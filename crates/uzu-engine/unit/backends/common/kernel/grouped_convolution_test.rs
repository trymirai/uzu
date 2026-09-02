#[cfg(backend = "metal")]
use std::{mem::size_of, time::Duration};

#[cfg(backend = "metal")]
use criterion::{Criterion, Throughput};
use half::bf16;
#[cfg(backend = "metal")]
use uzu_engine_macros::uzu_bench;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::GroupedConvolutionKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{
            alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend, submit_encoder,
        },
    },
};
#[cfg(backend = "metal")]
use crate::{
    backends::{common::Allocation, metal::Metal},
    tests::{cold_pool::ColdPool, matmul::iter_encode_loop_named},
};

#[derive(Clone, Copy)]
struct Shape {
    sequence_length: usize,
    model_dim: usize,
    group_size: usize,
    kernel_size: usize,
}

impl Shape {
    fn groups(self) -> usize {
        self.model_dim / self.group_size
    }

    fn input_len(self) -> usize {
        self.sequence_length * self.model_dim
    }

    fn coefficients_len(self) -> usize {
        self.sequence_length * 2 * self.kernel_size * self.groups()
    }

    fn base_kernel_len(self) -> usize {
        2 * self.kernel_size * self.model_dim
    }

    fn stage_offsets(
        self,
        stage: u32,
    ) -> (usize, usize) {
        let stage = stage as usize;
        let element_size = bf16::data_type().size_in_bytes();
        (
            stage * self.kernel_size * self.groups() * element_size,
            stage * self.kernel_size * self.model_dim * element_size,
        )
    }
}

fn run<B: Backend>(
    shape: Shape,
    input: &[bf16],
    coefficients: &[bf16],
    base_kernel: &[bf16],
    stage: u32,
) -> Vec<f32> {
    assert_eq!(input.len(), shape.input_len());
    assert_eq!(coefficients.len(), shape.coefficients_len());
    assert_eq!(base_kernel.len(), shape.base_kernel_len());
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::GroupedConvolutionKernel::new(
        &context,
        bf16::data_type(),
        shape.model_dim as u32,
        shape.group_size as u32,
        shape.kernel_size as u32,
    )
    .unwrap();
    let input = alloc_allocation_with_data::<B, bf16>(&context, input);
    let coefficients = alloc_allocation_with_data::<B, bf16>(&context, coefficients);
    let base_kernel = alloc_allocation_with_data::<B, bf16>(&context, base_kernel);
    let mut output = alloc_allocation::<B, bf16>(&context, shape.input_len());
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    let (coefficient_offset, base_kernel_offset) = shape.stage_offsets(stage);
    kernel.encode(
        &input,
        (&coefficients, coefficient_offset),
        (&base_kernel, base_kernel_offset),
        &mut output,
        shape.sequence_length as u32,
        &mut encoder,
    );
    submit_encoder(encoder);
    allocation_to_vec::<B, bf16>(&output).into_iter().map(|value| value.to_f32()).collect()
}

#[uzu_test]
fn test_bf16_golden() {
    let shape = Shape {
        sequence_length: 4,
        model_dim: 4,
        group_size: 2,
        kernel_size: 2,
    };
    const EXPECTED: [[f32; 16]; 2] = [
        [1.1, 0.75, 0.425, 0.1875, 0.2, -0.075, -0.25, -0.3, -0.3, -0.175, 0.15, 0.5, 0.8, 1.325, 2.15, 2.9],
        [0.1, 0.0, -0.075, -0.0625, -0.8, -0.575, -0.25, 0.2, 0.7, 1.325, 2.15, 3.0, 3.8, 4.825, 6.15, 7.4],
    ];
    let input = (0..16).map(|index| bf16::from_f32(index as f32 * 0.25 - 1.0)).collect::<Vec<_>>();
    let coefficients = (0..32).map(|index| bf16::from_f32(index as f32 * 0.05 - 0.4)).collect::<Vec<_>>();
    let base_kernel = (0..16).map(|index| bf16::from_f32(index as f32 * 0.1 - 0.7)).collect::<Vec<_>>();
    for stage in [0u32, 1] {
        let expected = run::<Cpu>(shape, &input, &coefficients, &base_kernel, stage);
        assert_eq_float(&EXPECTED[stage as usize], &expected, 0.05, "grouped convolution CPU reference");
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(shape, &input, &coefficients, &base_kernel, stage);
            assert_eq_float(&expected, &actual, 0.05, "grouped convolution backend parity");
        });
    }
}

#[uzu_test]
fn test_bf16_dflash_v2_shape() {
    let shape = Shape {
        sequence_length: 16,
        model_dim: 6656,
        group_size: 16,
        kernel_size: 2,
    };
    let input =
        (0..shape.input_len()).map(|index| bf16::from_f32((index % 31) as f32 * 0.05 - 0.75)).collect::<Vec<_>>();
    let coefficients = (0..shape.coefficients_len())
        .map(|index| bf16::from_f32((index % 17) as f32 * 0.02 - 0.16))
        .collect::<Vec<_>>();
    let base_kernel =
        (0..shape.base_kernel_len()).map(|index| bf16::from_f32((index % 13) as f32 * 0.03 - 0.18)).collect::<Vec<_>>();
    for stage in [0u32, 1] {
        let expected = run::<Cpu>(shape, &input, &coefficients, &base_kernel, stage);
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(shape, &input, &coefficients, &base_kernel, stage);
            assert_eq_float(&expected, &actual, 0.01, &format!("grouped convolution DFlash V2, stage {stage}"));
        });
    }
}

#[cfg(backend = "metal")]
struct BenchBuffers {
    input: Allocation<Metal>,
    coefficients: Allocation<Metal>,
    input_stage_output: Allocation<Metal>,
    output_stage_output: Allocation<Metal>,
}

#[cfg(backend = "metal")]
#[uzu_bench]
fn bench_dflash_v2(c: &mut Criterion) {
    const BENCHMARK: &str = "Metal/Kernel/GroupedConvolution";
    const MODEL_DIM: usize = 6656;
    const GROUP_SIZE: usize = 16;
    const KERNEL_SIZE: usize = 2;

    let context = crate::tests::util::shared_metal_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::GroupedConvolutionKernel::new(
        &context,
        bf16::data_type(),
        MODEL_DIM as u32,
        GROUP_SIZE as u32,
        KERNEL_SIZE as u32,
    )
    .expect("kernel");
    let base_kernel =
        alloc_allocation_with_data::<Metal, bf16>(&context, &vec![bf16::from_f32(0.01); 2 * KERNEL_SIZE * MODEL_DIM]);
    let mut group = c.benchmark_group(BENCHMARK);
    group.sample_size(30).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));

    for sequence_length in [2usize, 4, 8, 16] {
        let shape = Shape {
            sequence_length,
            model_dim: MODEL_DIM,
            group_size: GROUP_SIZE,
            kernel_size: KERNEL_SIZE,
        };
        let buffer_elements = 3 * shape.input_len() + shape.coefficients_len();
        let mut buffers = ColdPool::new(buffer_elements * size_of::<bf16>(), || BenchBuffers {
            input: alloc_allocation_with_data::<Metal, bf16>(&context, &vec![bf16::from_f32(0.1); shape.input_len()]),
            coefficients: alloc_allocation_with_data::<Metal, bf16>(
                &context,
                &vec![bf16::from_f32(0.02); shape.coefficients_len()],
            ),
            input_stage_output: alloc_allocation::<Metal, bf16>(&context, shape.input_len()),
            output_stage_output: alloc_allocation::<Metal, bf16>(&context, shape.input_len()),
        });
        group.throughput(Throughput::Elements((2 * shape.input_len()) as u64));
        group.bench_function(format!("T{sequence_length}"), |bencher| {
            iter_encode_loop_named::<Metal, _>(
                &context,
                bencher,
                &format!("{BENCHMARK}/T{sequence_length}"),
                |encoder| {
                    let buffers = buffers.next_mut();
                    let (input_coefficient_offset, input_base_kernel_offset) = shape.stage_offsets(0);
                    kernel.encode(
                        &buffers.input,
                        (&buffers.coefficients, input_coefficient_offset),
                        (&base_kernel, input_base_kernel_offset),
                        &mut buffers.input_stage_output,
                        shape.sequence_length as u32,
                        encoder,
                    );
                    let (output_coefficient_offset, output_base_kernel_offset) = shape.stage_offsets(1);
                    kernel.encode(
                        &buffers.input,
                        (&buffers.coefficients, output_coefficient_offset),
                        (&base_kernel, output_base_kernel_offset),
                        &mut buffers.output_stage_output,
                        shape.sequence_length as u32,
                        encoder,
                    );
                },
            );
        });
    }
}
