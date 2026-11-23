use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend, Kernels,
            kernel::{
                AttentionArguments, AttentionInput, AttentionKernel, AttentionKernelConfig, attention::KeyBlocks,
            },
        },
        metal::Metal,
    },
    data_type::DataType,
    encodable_block::mixer::attention::KVCacheState,
    tests::{helpers::create_buffer_with_data, matmul::iter_encode_loop_named, util::shared_metal_context},
};

// The attention layers of Qwen3.5-4B: sixteen gated query heads over four key-value heads of 256 dimensions.
const HEAD_DIM: u32 = 256;
const NUM_Q_HEADS: u32 = 16;
const NUM_KV_HEADS: u32 = 4;
const PROJECTION_DIM: u32 = (2 * NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;
const PREFIX_LENGTHS: [u32; 2] = [1024, 8192];
// One token and a pair of tokens decode through different kernels than a larger suffix does.
const SUFFIX_LENGTHS: [u32; 3] = [1, 2, 64];

#[uzu_bench]
fn bench_attention(c: &mut Criterion) {
    const BENCHMARK: &str = "Metal/Kernel/Attention/Qwen3.5-4B";

    let context = shared_metal_context();
    let mut rng = SmallRng::seed_from_u64(42);
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionKernel::new(
        &context,
        AttentionKernelConfig {
            head_dim: HEAD_DIM,
            num_groups: NUM_KV_HEADS,
            num_q_heads: NUM_Q_HEADS,
            has_sinks: false,
            gate_row_stride: Some(PROJECTION_DIM),
            is_kv_cache_ring: false,
            is_causal: true,
            sliding_window_size: None,
            scale: None,
            data_type: DataType::BF16,
            sparse: None,
        },
    )
    .expect("kernel");
    let layout = kernel.kv_layout();
    let mut codes = |count: usize| -> Vec<i8> { (0..count).map(|_| rng.random_range(-127i8..=127i8)).collect() };

    let mut group = c.benchmark_group(BENCHMARK);
    group.sample_size(20).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(2));
    for prefix_length in PREFIX_LENGTHS {
        for suffix_length in SUFFIX_LENGTHS {
            let keys = prefix_length + suffix_length;
            let query_count = (NUM_Q_HEADS * suffix_length) as usize;
            let scales = |count: usize| vec![bf16::from_f32(0.01); count];
            let query_codes = create_buffer_with_data::<Metal, i8>(&context, &codes(query_count * HEAD_DIM as usize));
            let query_scales = create_buffer_with_data::<Metal, bf16>(&context, &scales(query_count));
            let k_codes = create_buffer_with_data::<Metal, i8>(&context, &codes(layout.code_bytes(keys)));
            let v_codes = create_buffer_with_data::<Metal, i8>(&context, &codes(layout.code_bytes(keys)));
            let scale_count = layout.scale_bytes(keys) / size_of::<bf16>();
            let k_scales = create_buffer_with_data::<Metal, bf16>(&context, &scales(scale_count));
            let v_scales = create_buffer_with_data::<Metal, bf16>(&context, &scales(scale_count));
            let gate = create_buffer_with_data::<Metal, bf16>(
                &context,
                &vec![bf16::from_f32(0.5); (suffix_length * PROJECTION_DIM) as usize],
            );

            group.throughput(Throughput::Elements(u64::from(suffix_length)));
            let id = BenchmarkId::new(format!("prefix{prefix_length}"), suffix_length);
            group.bench_function(id, |bencher| {
                iter_encode_loop_named::<Metal, _>(
                    &context,
                    bencher,
                    &format!("{BENCHMARK}/prefix{prefix_length}/{suffix_length}"),
                    |command_buffer| {
                        kernel
                            .encode(
                                AttentionInput {
                                    query_codes: &query_codes,
                                    query_scales: &query_scales,
                                    k_codes: &k_codes,
                                    k_scales: &k_scales,
                                    v_codes: &v_codes,
                                    v_scales: &v_scales,
                                },
                                AttentionArguments {
                                    suffix_length,
                                    trie: None::<&<Metal as Backend>::GlobalGpuBuffer>,
                                    key_blocks: None::<KeyBlocks<&<Metal as Backend>::ConstantBuffer>>,
                                    sinks: None,
                                    cache: KVCacheState::new(prefix_length, None),
                                    gate: Some(&gate),
                                },
                                command_buffer,
                            )
                            .expect("encode");
                    },
                );
            });
        }
    }
}
