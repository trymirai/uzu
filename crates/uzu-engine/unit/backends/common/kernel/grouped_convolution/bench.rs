use std::{mem::size_of, time::Duration};

use criterion::{Criterion, Throughput};
use half::bf16;
use uzu_engine_macros::uzu_bench;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Allocation, Backend, Kernels, kernel::GroupedConvolutionKernel},
        metal::Metal,
    },
    tests::{
        cold_pool::ColdPool,
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop_named,
    },
};

struct BenchBuffers {
    input: Allocation<Metal>,
    coefficients: Allocation<Metal>,
    input_stage_output: Allocation<Metal>,
    output_stage_output: Allocation<Metal>,
}

#[uzu_bench]
fn bench_grouped_convolution(c: &mut Criterion) {
    const BENCHMARK: &str = "Metal/Kernel/GroupedConvolution";
    const MODEL_DIM: usize = 6656;
    const GROUP_SIZE: usize = 16;
    const KERNEL_SIZE: usize = 2;
    const COEFFICIENT_STAGE_BYTES: usize = KERNEL_SIZE * (MODEL_DIM / GROUP_SIZE) * size_of::<bf16>();
    const BASE_KERNEL_STAGE_BYTES: usize = KERNEL_SIZE * MODEL_DIM * size_of::<bf16>();

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
        let input_len = sequence_length * MODEL_DIM;
        let coefficients_len = sequence_length * 2 * KERNEL_SIZE * (MODEL_DIM / GROUP_SIZE);
        let buffer_elements = 3 * input_len + coefficients_len;
        let mut buffers = ColdPool::new(buffer_elements * size_of::<bf16>(), || BenchBuffers {
            input: alloc_allocation_with_data::<Metal, bf16>(&context, &vec![bf16::from_f32(0.1); input_len]),
            coefficients: alloc_allocation_with_data::<Metal, bf16>(
                &context,
                &vec![bf16::from_f32(0.02); coefficients_len],
            ),
            input_stage_output: alloc_allocation::<Metal, bf16>(&context, input_len),
            output_stage_output: alloc_allocation::<Metal, bf16>(&context, input_len),
        });
        group.throughput(Throughput::Elements((2 * input_len) as u64));
        group.bench_function(format!("T{sequence_length}"), |bencher| {
            iter_encode_loop_named::<Metal, _>(
                &context,
                bencher,
                &format!("{BENCHMARK}/T{sequence_length}"),
                |encoder| {
                    let buffers = buffers.next_mut();
                    kernel.encode(
                        &buffers.input,
                        (&buffers.coefficients, 0),
                        (&base_kernel, 0),
                        &mut buffers.input_stage_output,
                        sequence_length as u32,
                        encoder,
                    );
                    kernel.encode(
                        &buffers.input,
                        (&buffers.coefficients, COEFFICIENT_STAGE_BYTES),
                        (&base_kernel, BASE_KERNEL_STAGE_BYTES),
                        &mut buffers.output_stage_output,
                        sequence_length as u32,
                        encoder,
                    );
                },
            );
        });
    }
}
