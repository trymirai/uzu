#[cfg(metal_backend)]
use std::time::Duration;

#[cfg(metal_backend)]
use criterion::Criterion;
use half::bf16;
use num_traits::Float;
#[cfg(metal_backend)]
use proc_macros::uzu_bench;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::StateAdvanceKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec},
    },
};
#[cfg(metal_backend)]
use crate::{
    backends::metal::Metal,
    data_type::DataType,
    tests::{cold_pool::ColdPool, helpers::allocation_size_bytes, matmul::iter_encode_loop_named},
};

const TREE_SIZE: usize = 128;
const HEAD_DIM: usize = 128;
const PATH: &[u32] = &[0, 1, 3, 7, 15, 31, 47, 63, 79, 95, 111, 119, 123, 125, 126, 127];
const TEST_NUM_V_HEADS: usize = 6;
const TEST_NUM_K_HEADS: usize = 2;

fn run<B: Backend, T: ArrayElement + Float>(accepted_indices: &[u32]) -> Vec<f32> {
    let context = B::Context::new().expect("context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(&context, T::data_type(), HEAD_DIM as u32)
            .expect("kernel");
    let key_len = TREE_SIZE * TEST_NUM_K_HEADS * HEAD_DIM;
    let value_len = TREE_SIZE * TEST_NUM_V_HEADS * HEAD_DIM;
    let scalar_len = TREE_SIZE * TEST_NUM_V_HEADS;
    let state_len = TEST_NUM_V_HEADS * HEAD_DIM * HEAD_DIM;
    let k_norm = (0..key_len).map(|i| (i % 17) as f32 * 0.002 - 0.016).collect::<Vec<_>>();
    let v = (0..value_len).map(|i| T::from((i % 19) as f32 * 0.003 - 0.027).unwrap()).collect::<Vec<_>>();
    let log_decay = (0..scalar_len).map(|i| -0.01 - (i % 7) as f32 * 0.002).collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.2 + (i % 5) as f32 * 0.03).collect::<Vec<_>>();
    let initial_state = (0..state_len).map(|i| (i % 23) as f32 * 0.001 - 0.011).collect::<Vec<_>>();

    let k_norm = alloc_allocation_with_data::<B, f32>(&context, &k_norm);
    let v = alloc_allocation_with_data::<B, T>(&context, &v);
    let log_decay = alloc_allocation_with_data::<B, f32>(&context, &log_decay);
    let beta = alloc_allocation_with_data::<B, f32>(&context, &beta);
    let accepted_len = accepted_indices.len();
    let accepted_indices = alloc_allocation_with_data::<B, u32>(&context, accepted_indices);
    let mut committed_state = alloc_allocation_with_data::<B, f32>(&context, &initial_state);

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode(
        &k_norm,
        &v,
        &log_decay,
        &beta,
        &accepted_indices,
        &mut committed_state,
        accepted_len as u32,
        TEST_NUM_V_HEADS as u32,
        TEST_NUM_K_HEADS as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&committed_state)
}

#[uzu_test]
fn test_state_advance() {
    for path in [&PATH[..1], &PATH[..8], PATH] {
        for_each_non_cpu_backend!(|B| {
            assert_eq_float(&run::<Cpu, f32>(path), &run::<B, f32>(path), 2e-5, "F32 state");
            assert_eq_float(&run::<Cpu, bf16>(path), &run::<B, bf16>(path), 2e-5, "BF16 state");
        });
    }
}

#[cfg(metal_backend)]
#[uzu_bench]
fn bench_state_advance(c: &mut Criterion) {
    const NUM_V_HEADS: usize = 48;
    const NUM_K_HEADS: usize = 16;
    const BENCHMARK: &str = "Metal/Kernel/GDNTreeVerify/StateAdvance";

    let context = crate::tests::util::shared_metal_context();
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::StateAdvanceKernel::new(&context, DataType::BF16, HEAD_DIM as u32)
            .expect("kernel");
    let k_norm = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.001; TREE_SIZE * NUM_K_HEADS * HEAD_DIM]);
    let v = alloc_allocation_with_data::<Metal, bf16>(
        &context,
        &vec![bf16::from_f32(0.01); TREE_SIZE * NUM_V_HEADS * HEAD_DIM],
    );
    let log_decay = alloc_allocation_with_data::<Metal, f32>(&context, &vec![-0.01; TREE_SIZE * NUM_V_HEADS]);
    let beta = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.2; TREE_SIZE * NUM_V_HEADS]);
    let initial_state = vec![0.01; NUM_V_HEADS * HEAD_DIM * HEAD_DIM];
    let mut group = c.benchmark_group(BENCHMARK);
    group.sample_size(30).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));

    for accepted_len in [1usize, 4, 8, 16] {
        let accepted_indices = alloc_allocation_with_data::<Metal, u32>(&context, &PATH[..accepted_len]);
        let mut committed_states = ColdPool::new(allocation_size_bytes::<f32>(initial_state.len()), || {
            alloc_allocation_with_data::<Metal, f32>(&context, &initial_state)
        });
        group.bench_function(format!("L{accepted_len}"), |bencher| {
            iter_encode_loop_named::<Metal, _>(&context, bencher, &format!("{BENCHMARK}/L{accepted_len}"), |encoder| {
                kernel.encode(
                    &k_norm,
                    &v,
                    &log_decay,
                    &beta,
                    &accepted_indices,
                    committed_states.next_mut(),
                    accepted_len as u32,
                    NUM_V_HEADS as u32,
                    NUM_K_HEADS as u32,
                    encoder,
                );
            });
        });
    }
}
