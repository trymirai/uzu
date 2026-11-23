use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rand_distr::Normal;
use uzu_engine_macros::uzu_bench;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, Context, CpuBuffer, Kernels,
            kernel::delta_net_chunked_prefill::{DeltaNetChunkedPrefill, DeltaNetChunkedPrefillArgs},
        },
        metal::Metal,
    },
    data_type::DataType,
    tests::{helpers::create_buffer_with_data, matmul::iter_encode_loop_named, util::shared_metal_context},
};

// The delta net layers of Qwen3.5-4B.
const NUM_V_HEADS: usize = 32;
const NUM_K_HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const KEY_DIM: usize = NUM_K_HEADS * HEAD_DIM;
const VALUE_DIM: usize = NUM_V_HEADS * HEAD_DIM;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
const TOTAL_PROJ_DIM: usize = CONV_DIM + VALUE_DIM + 2 * NUM_V_HEADS;
const SUFFIX_LENGTHS: [usize; 2] = [256, 1024];

#[uzu_bench]
fn bench_delta_net_chunked_prefill(c: &mut Criterion) {
    const BENCHMARK: &str = "Metal/Kernel/GDNChunkedPrefill/Qwen3.5-4B";

    let context = shared_metal_context();
    let Some(kernel) = <<Metal as Backend>::Kernels as Kernels>::DeltaNetChunkedPrefill::new(
        &context,
        DataType::BF16,
        HEAD_DIM as u32,
        (NUM_V_HEADS / NUM_K_HEADS) as u32,
    )
    .expect("kernel") else {
        eprintln!("chunked prefill is unsupported on this device");
        return;
    };
    let mut rng = SmallRng::seed_from_u64(7);
    let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
    // a_log = ln U(1, 16) and dt_bias = softplus^-1(logU(1e-3, 1e-1)) as in Qwen3-Next
    let spread = |head: usize| (head as f32 + 0.5) / NUM_V_HEADS as f32;
    let a_log: Vec<f32> = (0..NUM_V_HEADS).map(|head| (1.0 + 15.0 * spread(head)).ln()).collect();
    let dt_bias: Vec<f32> =
        (0..NUM_V_HEADS).map(|head| (1e-3f32.ln() + 100f32.ln() * spread(head)).exp().exp_m1().ln()).collect();
    let a_log = create_buffer_with_data::<Metal, f32>(&context, &a_log);
    let dt_bias = create_buffer_with_data::<Metal, f32>(&context, &dt_bias);
    let mut state = create_buffer_with_data::<Metal, f32>(&context, &vec![0.0; NUM_V_HEADS * HEAD_DIM * HEAD_DIM]);

    let mut group = c.benchmark_group(BENCHMARK);
    group.sample_size(20).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(2));
    for suffix_len in SUFFIX_LENGTHS {
        let in_projected_data: Vec<bf16> = (0..suffix_len * TOTAL_PROJ_DIM)
            .map(|index| {
                let x: f32 = rng.sample(normal);
                let column = index % TOTAL_PROJ_DIM;
                bf16::from_f32(if column < CONV_DIM {
                    x / (1.0 + (-x).exp())
                } else if column >= TOTAL_PROJ_DIM - NUM_V_HEADS {
                    0.5 * x
                } else {
                    x
                })
            })
            .collect();
        let mut setup = context.create_command_buffer(None, None).unwrap();
        let mut in_projected = setup.allocate_scratch(size_of_val(in_projected_data.as_slice())).unwrap();
        in_projected.copyin(&in_projected_data);
        let mut delta_output = setup.allocate_scratch(suffix_len * VALUE_DIM * size_of::<bf16>()).unwrap();

        group.throughput(Throughput::Elements(suffix_len as u64));
        group.bench_function(BenchmarkId::from_parameter(suffix_len), |bencher| {
            iter_encode_loop_named::<Metal, _>(
                &context,
                bencher,
                &format!("{BENCHMARK}/{suffix_len}"),
                |command_buffer| {
                    kernel
                        .encode(
                            DeltaNetChunkedPrefillArgs {
                                in_projected: &in_projected,
                                a_log: &a_log,
                                dt_bias: &dt_bias,
                                ssm_state: &mut state,
                                delta_output: &mut delta_output,
                                num_heads: NUM_V_HEADS as u32,
                                num_groups: NUM_K_HEADS as u32,
                                value_head_dim: HEAD_DIM as u32,
                                key_dim: KEY_DIM as u32,
                                value_dim: VALUE_DIM as u32,
                                suffix_len: suffix_len as u32,
                            },
                            command_buffer,
                        )
                        .expect("encode");
                },
            );
        });
    }
}
