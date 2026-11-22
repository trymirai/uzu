//! Mixture-of-experts layer visits with Inkling 2 Mini's shapes: 2304-wide tokens, 128 experts of hidden width 1152,
//! eight active. Each visit runs the whole block (router, gate/up with the activation, down with the combine). The
//! throughput counts the bytes the visit must read: the float32 router and every expert its tokens route to.
//! Inputs rotate through enough batches that the routed experts change from one visit to the next, so weights come
//! from DRAM rather than the system cache.

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use uzu_engine_macros::uzu_bench;

use super::moe_fixtures::{
    HostMoe, INKLING_EXPERTS, INKLING_HIDDEN_DIM, INKLING_MODEL_DIM, INKLING_TOP_K, random_bf16,
};
use crate::{
    backends::common::{Backend, gpu_types::moe::MoeExpertFormat},
    encodable_block::mlp::moe::MixtureOfExpertsMlp,
    tests::{
        helpers::{create_buffer, create_buffer_with_data, create_context},
        matmul::iter_encode_loop_named,
        metrics::wait_gpu_cooldown,
        util::type_short_name,
    },
};

/// Distinct input batches per token count: enough for decode to cycle through every expert, fewer for prefill, which
/// routes to nearly all of them in every batch.
fn input_batches(tokens: u32) -> usize {
    if tokens <= 16 {
        16
    } else {
        4
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Experts the batch routes to, by the same float32 sigmoid routing the router kernels apply.
fn routed_experts(
    host: &HostMoe,
    input: &[bf16],
    tokens: u32,
) -> usize {
    let model_dim = host.model_dim as usize;
    let mut used = vec![false; host.experts as usize];
    for token in 0..tokens as usize {
        let x = &input[token * model_dim..(token + 1) * model_dim];
        let mut gating: Vec<(f32, usize)> = (0..host.experts as usize)
            .map(|expert| {
                let row = &host.router_weights[expert * model_dim..(expert + 1) * model_dim];
                let logit: f32 = row.iter().zip(x).map(|(&w, &v)| w * v.to_f32()).sum();
                (sigmoid(logit) + host.expert_biases[expert], expert)
            })
            .collect();
        gating.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for &(_, expert) in &gating[..host.top_k as usize] {
            used[expert] = true;
        }
    }
    used.iter().filter(|&&used| used).count()
}

fn bench_format<B: Backend>(
    criterion: &mut Criterion,
    context: &B::Context,
    format: MoeExpertFormat,
) {
    let host = HostMoe::random(format, INKLING_MODEL_DIM, INKLING_HIDDEN_DIM, INKLING_EXPERTS, INKLING_TOP_K, 2026);
    let moe = MixtureOfExpertsMlp::<B>::from_parts(host.parts::<B>(context), context)
        .unwrap_or_else(|error| panic!("cannot build the block: {error}"));
    let expert_bytes = host.up.bytes_per_expert() + host.down.bytes_per_expert();
    let router_bytes = host.router_weights.len() * size_of::<f32>();
    let group_path = format!("{}/Kernel/MoE/Inkling/{format:?}", type_short_name::<B>());
    let mut group = criterion.benchmark_group(group_path.clone());
    for tokens in [1u32, 2, 3, 4, 5, 8, 16, 512] {
        let batch_count = input_batches(tokens);
        let batches: Vec<Vec<bf16>> = (0..batch_count)
            .map(|batch| random_bf16((tokens * host.model_dim) as usize, 7000 + 100 * u64::from(tokens) + batch as u64))
            .collect();
        let routed: usize = batches.iter().map(|batch| routed_experts(&host, batch, tokens)).sum();
        let bytes = (routed * expert_bytes) / batch_count + router_bytes;
        let inputs: Vec<B::GlobalGpuBuffer> =
            batches.iter().map(|batch| create_buffer_with_data::<B, bf16>(context, batch)).collect();
        let mut output = create_buffer::<B, bf16>(context, (tokens * host.model_dim) as usize);
        group.throughput(Throughput::BytesDecimal(bytes as u64));
        wait_gpu_cooldown();
        group.bench_function(BenchmarkId::from_parameter(format!("T{tokens}")), |bencher| {
            let benchmark_path = format!("{group_path}/T{tokens}");
            let mut batch = 0;
            iter_encode_loop_named::<B, _>(context, bencher, &benchmark_path, |command_buffer| {
                moe.encode_rows(&inputs[batch % batch_count], &mut output, tokens, command_buffer).unwrap();
                batch += 1;
            });
        });
    }
}

// Metal only: the loop reads GPU execution time, and the CPU twins are references, not candidates.
#[uzu_bench]
fn bench_moe_inkling(criterion: &mut Criterion) {
    #[cfg(backend = "metal")]
    {
        type B = crate::backends::metal::Metal;
        let context = create_context::<B>();
        for format in [MoeExpertFormat::Nvfp4, MoeExpertFormat::AffineU4] {
            bench_format::<B>(criterion, &context, format);
        }
    }
    let _ = criterion;
}
