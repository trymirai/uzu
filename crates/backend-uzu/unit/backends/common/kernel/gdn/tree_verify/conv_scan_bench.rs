#![cfg(metal_backend)]

use std::{mem::size_of, time::Duration};

use criterion::{BenchmarkId, Criterion};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{Allocation, Backend, Kernels, kernel::DeltaNetConvTreeScanKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop_named,
    },
};

const CONV_DIM: usize = 10_240;
const TOTAL_PROJ_DIM: usize = 16_480;
const KERNEL_SIZE: usize = 4;
const STATE_STRIDE: usize = KERNEL_SIZE - 1;

struct Buffers {
    input: Allocation<Metal>,
    output: Allocation<Metal>,
    suffix_state: Allocation<Metal>,
}

#[uzu_bench]
fn bench_delta_net_conv_tree_scan(c: &mut Criterion) {
    let context = crate::tests::util::shared_metal_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::DeltaNetConvTreeScanKernel::new(
        &context,
        DataType::BF16,
        KERNEL_SIZE as u32,
        true,
    )
    .expect("kernel");
    let weights = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.01; CONV_DIM * KERNEL_SIZE]);
    let bias = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.0; CONV_DIM]);
    let base_state = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.0; CONV_DIM * STATE_STRIDE]);
    let mut group = c.benchmark_group("Metal/Kernel/GDNTreeVerify/DeltaNetConvTreeScan");
    group.sample_size(30).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));

    for tree_size in [49usize, 64, 128] {
        let parents = alloc_allocation_with_data::<Metal, i32>(
            &context,
            &(0..tree_size)
                .map(|node| {
                    if node == 0 {
                        -1
                    } else {
                        ((node - 1) / 2) as i32
                    }
                })
                .collect::<Vec<_>>(),
        );
        let input_len = tree_size * TOTAL_PROJ_DIM;
        let state_len = tree_size * CONV_DIM * STATE_STRIDE;
        let mut buffers = ColdPool::new(2 * input_len * size_of::<bf16>() + state_len * size_of::<f32>(), || Buffers {
            input: alloc_allocation_with_data::<Metal, bf16>(&context, &vec![bf16::from_f32(0.1); input_len]),
            output: alloc_allocation::<Metal, bf16>(&context, input_len),
            suffix_state: alloc_allocation::<Metal, f32>(&context, state_len),
        });
        let path = format!("Metal/Kernel/GDNTreeVerify/DeltaNetConvTreeScan/T{tree_size}");
        group.bench_function(BenchmarkId::from_parameter(format!("T{tree_size}")), |bencher| {
            iter_encode_loop_named::<Metal, _>(&context, bencher, &path, |encoder| {
                let buffers = buffers.next_mut();
                kernel.encode(
                    &buffers.input,
                    &weights,
                    Some(&bias),
                    &base_state,
                    &parents,
                    &mut buffers.output,
                    &mut buffers.suffix_state,
                    tree_size as u32,
                    TOTAL_PROJ_DIM as u32,
                    CONV_DIM as u32,
                    encoder,
                );
            });
        });
    }
}
