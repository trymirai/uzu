use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{Backend, Kernels, gpu_types::qkv_norm::QKVNormParams, kernel::AttentionPrepareKernel},
        metal::Metal,
    },
    data_type::DataType,
    tests::{
        attention::kv_layout,
        helpers::{create_buffer, create_buffer_with_data},
        matmul::iter_encode_loop_named,
        util::shared_metal_context,
    },
};

// The attention layers of Qwen3.5-4B: sixteen query heads over four key-value heads of 256 dimensions, with RoPE
// and a scaled RMS norm on queries and keys.
const HEAD_DIM: u32 = 256;
const NUM_Q_HEADS: u32 = 16;
const NUM_KV_HEADS: u32 = 4;
const PREFIX_TOKENS: u32 = 1024;
const CACHE_TOKENS: u32 = 4096;
const SUFFIX_LENGTHS: [u32; 3] = [1, 64, 1024];

#[uzu_bench]
fn bench_attention_prepare(c: &mut Criterion) {
    const BENCHMARK: &str = "Metal/Kernel/AttentionPrepare/Qwen3.5-4B";

    let context = shared_metal_context();
    let mut rng = SmallRng::seed_from_u64(42);
    let layout = kv_layout::<Metal>(&context, HEAD_DIM, NUM_KV_HEADS);
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        HEAD_DIM,
        true,
        NUM_KV_HEADS,
        layout.backend_opaque_layout,
        true,
        true,
        true,
        false,
        false,
        0,
        false,
    )
    .expect("kernel");
    let norm_params = QKVNormParams {
        enabled: [1, 1, 0],
        full_layer: [0; 3],
        epsilon: [1e-6; 3],
        scale_offset: [1.0, 1.0, 0.0],
    };
    let mut norm_scales = || {
        let scales: Vec<f32> = (0..HEAD_DIM).map(|_| rng.random_range(-0.2f32..0.2f32)).collect();
        create_buffer_with_data::<Metal, f32>(&context, &scales)
    };
    let (query_norm_scales, key_norm_scales) = (norm_scales(), norm_scales());
    let rope_inverse_frequencies = create_buffer_with_data::<Metal, f32>(
        &context,
        &(0..HEAD_DIM / 2).map(|pair| 10_000f32.powf(-2.0 * pair as f32 / HEAD_DIM as f32)).collect::<Vec<_>>(),
    );
    let mut k_codes = create_buffer::<Metal, u8>(&context, layout.code_bytes(CACHE_TOKENS));
    let mut v_codes = create_buffer::<Metal, u8>(&context, layout.code_bytes(CACHE_TOKENS));
    let mut k_scales = create_buffer::<Metal, u8>(&context, layout.scale_bytes(CACHE_TOKENS));
    let mut v_scales = create_buffer::<Metal, u8>(&context, layout.scale_bytes(CACHE_TOKENS));
    // Queries, keys, values and one gate value per query head.
    let input_row_stride = (2 * NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;

    let mut group = c.benchmark_group(BENCHMARK);
    group.sample_size(30).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));
    for suffix_length in SUFFIX_LENGTHS {
        let query_count = (NUM_Q_HEADS * suffix_length) as usize;
        let qkvg: Vec<bf16> =
            (0..suffix_length * input_row_stride).map(|_| bf16::from_f32(rng.random_range(-1.0f32..1.0f32))).collect();
        let qkvg = create_buffer_with_data::<Metal, bf16>(&context, &qkvg);
        let mut q_codes = create_buffer::<Metal, i8>(&context, query_count * HEAD_DIM as usize);
        let mut q_scales = create_buffer::<Metal, bf16>(&context, query_count);

        group.throughput(Throughput::Elements(u64::from(suffix_length)));
        group.bench_function(BenchmarkId::from_parameter(suffix_length), |bencher| {
            iter_encode_loop_named::<Metal, _>(
                &context,
                bencher,
                &format!("{BENCHMARK}/{suffix_length}"),
                |command_buffer| {
                    kernel.encode(
                        &qkvg,
                        &mut q_codes,
                        &mut q_scales,
                        Some(&mut k_codes),
                        Some(&mut k_scales),
                        Some(&mut v_codes),
                        Some(&mut v_scales),
                        Some(&rope_inverse_frequencies),
                        Some(&query_norm_scales),
                        Some(&key_norm_scales),
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        None::<&mut <Metal as Backend>::GlobalGpuBuffer>,
                        norm_params,
                        NUM_Q_HEADS,
                        Some(HEAD_DIM),
                        Some(1.0),
                        Some(PREFIX_TOKENS),
                        PREFIX_TOKENS,
                        input_row_stride,
                        suffix_length,
                        NUM_Q_HEADS / NUM_KV_HEADS * suffix_length,
                        None,
                        None::<&<Metal as Backend>::GlobalGpuBuffer>,
                        command_buffer,
                    );
                },
            );
        });
    }
}
