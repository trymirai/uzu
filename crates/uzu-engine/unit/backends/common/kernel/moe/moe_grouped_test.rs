use futures::executor::block_on;
use half::{bf16, f16};
use uzu_engine_macros::uzu_test;

use super::moe_fixtures::{
    HostExperts, HostMoe, HostRoutes, INKLING_ACTIVATION, INKLING_EXPERTS, INKLING_HIDDEN_DIM, INKLING_MODEL_DIM,
    INKLING_ROUTED_SCALING_FACTOR, INKLING_TOP_K, Rng, random_bf16, random_f16,
};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::moe::MoeExpertFormat,
            kernel::{
                MoeCombineRoutesKernel, MoeGroupedAffineDownKernel, MoeGroupedAffineUpKernel, MoeGroupedDownKernel,
                MoeGroupedUpKernel, MoeRouteFinalizeKernel, MoeRouterLogitsKernel,
            },
        },
        cpu::Cpu,
    },
    data_type::DataType,
    encodable_block::mlp::moe::MixtureOfExpertsMlp,
    tests::{
        assert::{assert_eq_float, assert_parity},
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
    },
};

fn submit<B: Backend>(command_buffer: <B::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding) {
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
}

fn run_router_logits<B: Backend>(
    input: &[bf16],
    weights: &[f32],
    biases: Option<&[f32]>,
    tokens: u32,
    model_dim: u32,
    experts: u32,
) -> Vec<f32> {
    let context = create_context::<B>();
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let weights = create_buffer_with_data::<B, f32>(&context, weights);
    let biases = biases.map(|biases| create_buffer_with_data::<B, f32>(&context, biases));
    let mut logits = create_buffer::<B, f32>(&context, (tokens * experts) as usize);
    let kernel =
        <B::Kernels as Kernels>::MoeRouterLogitsKernel::new(&context, DataType::BF16, biases.is_some()).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&input, &weights, biases.as_ref(), &mut logits, tokens, model_dim, experts, &mut command_buffer);
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &logits)
}

#[uzu_test]
fn moe_router_logits_matches_cpu() {
    let (model_dim, experts) = (INKLING_MODEL_DIM, INKLING_EXPERTS);
    let mut rng = Rng::new(5);
    let spread = 4.0 / (model_dim as f32).sqrt();
    let weights: Vec<f32> = (0..experts * model_dim).map(|_| spread * rng.symmetric()).collect();
    let biases: Vec<f32> = (0..experts).map(|_| rng.symmetric()).collect();
    for tokens in [33u32, 100, 512] {
        let input = random_bf16((tokens * model_dim) as usize, 9 + u64::from(tokens));
        for biases in [None, Some(biases.as_slice())] {
            let expected = run_router_logits::<Cpu>(&input, &weights, biases, tokens, model_dim, experts);
            for_each_non_cpu_backend!(|B| {
                let actual = run_router_logits::<B>(&input, &weights, biases, tokens, model_dim, experts);
                assert_eq_float(&expected, &actual, 2e-4, &format!("{tokens} tokens, biases {}", biases.is_some()));
            });
        }
    }
}

struct FinalizeResult {
    route_experts: Vec<u32>,
    route_weights: Vec<f32>,
    expert_counts: Vec<u32>,
    expert_routes: Vec<Vec<u32>>,
    active_experts: Vec<(u32, u32)>,
}

fn run_route_finalize<B: Backend>(
    logits: &[f32],
    expert_biases: &[f32],
    tokens: u32,
    experts: u32,
    top_k: u32,
) -> FinalizeResult {
    let context = create_context::<B>();
    let routes = (tokens * top_k) as usize;
    let logits = create_buffer_with_data::<B, f32>(&context, logits);
    let expert_biases = create_buffer_with_data::<B, f32>(&context, expert_biases);
    let mut route_experts = create_buffer::<B, u32>(&context, routes);
    let mut route_weights = create_buffer::<B, f32>(&context, routes);
    let mut expert_counts = create_buffer::<B, u32>(&context, experts as usize);
    let mut expert_routes = create_buffer::<B, u32>(&context, (experts * tokens) as usize);
    let mut active_experts = create_buffer::<B, u32>(&context, experts as usize + 1);
    let kernel = <B::Kernels as Kernels>::MoeRouteFinalizeKernel::new(&context, true, true, false).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &logits,
        Some(&expert_biases),
        &mut route_experts,
        &mut route_weights,
        &mut expert_counts,
        &mut expert_routes,
        &mut active_experts,
        tokens,
        experts,
        top_k,
        INKLING_ROUTED_SCALING_FACTOR,
        &mut command_buffer,
    );
    submit::<B>(command_buffer);
    let expert_counts: Vec<u32> = buffer_to_vec::<B, _>(&context, &expert_counts);
    let all_routes: Vec<u32> = buffer_to_vec::<B, _>(&context, &expert_routes);
    let active_experts: Vec<u32> = buffer_to_vec::<B, _>(&context, &active_experts);
    let expert_routes: Vec<Vec<u32>> = (0..experts as usize)
        .map(|expert| {
            let start = expert * tokens as usize;
            let mut routes = all_routes[start..start + expert_counts[expert] as usize].to_vec();
            routes.sort_unstable();
            routes
        })
        .collect();
    FinalizeResult {
        route_experts: buffer_to_vec::<B, _>(&context, &route_experts),
        route_weights: buffer_to_vec::<B, _>(&context, &route_weights),
        active_experts: super::moe_route_test::decode_active(&active_experts, &expert_routes),
        expert_routes,
        expert_counts,
    }
}

#[uzu_test]
fn moe_route_finalize_matches_cpu() {
    let experts = INKLING_EXPERTS;
    let mut rng = Rng::new(17);
    let biases: Vec<f32> = (0..experts).map(|_| 0.05 * rng.symmetric()).collect();
    for tokens in [33u32, 512, 1024] {
        let logits: Vec<f32> = (0..tokens * experts).map(|_| 2.0 * rng.symmetric()).collect();
        let expected = run_route_finalize::<Cpu>(&logits, &biases, tokens, experts, INKLING_TOP_K);
        for_each_non_cpu_backend!(|B| {
            let actual = run_route_finalize::<B>(&logits, &biases, tokens, experts, INKLING_TOP_K);
            let label = format!("{tokens} tokens");
            assert_eq!(expected.route_experts, actual.route_experts, "{label}: route experts");
            assert_eq_float(&expected.route_weights, &actual.route_weights, 1e-5, &format!("{label}: route weights"));
            assert_eq!(expected.expert_counts, actual.expert_counts, "{label}: expert counts");
            assert_eq!(expected.expert_routes, actual.expert_routes, "{label}: expert routes");
            assert_eq!(expected.active_experts, actual.active_experts, "{label}: active experts");
        });
    }
}

struct Tables<B: Backend> {
    route_weights: B::GlobalGpuBuffer,
    expert_counts: B::GlobalGpuBuffer,
    expert_routes: B::GlobalGpuBuffer,
}

fn upload_tables<B: Backend>(
    routes: &HostRoutes,
    context: &B::Context,
) -> Tables<B> {
    Tables {
        route_weights: create_buffer_with_data::<B, f32>(context, &routes.route_weights),
        expert_counts: create_buffer_with_data::<B, u32>(context, &routes.expert_counts),
        expert_routes: create_buffer_with_data::<B, u32>(context, &routes.expert_routes),
    }
}

fn run_grouped_up<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    input: &[bf16],
) -> Vec<f16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.cols, experts.rows / 2);
    let weights = experts.upload::<B>(&context);
    let tables = upload_tables::<B>(routes, &context);
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    let mut hidden =
        create_buffer_with_data::<B, f16>(&context, &vec![f16::from_f32(-7.0); routes_count * hidden_dim as usize]);
    let activation = INKLING_ACTIVATION;
    let kernel =
        <B::Kernels as Kernels>::MoeGroupedUpKernel::new(&context, DataType::BF16, experts.format, true, true).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &input,
        &weights.codes,
        &weights.scales,
        weights.global_scales.as_ref(),
        &tables.expert_counts,
        &tables.expert_routes,
        &tables.route_weights,
        &mut hidden,
        routes.tokens,
        experts.experts,
        model_dim,
        hidden_dim,
        routes.top_k,
        activation.up_soft_cap.unwrap(),
        activation.gate_soft_cap.unwrap(),
        activation.alpha,
        &mut command_buffer,
    );
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &hidden)
}

fn run_grouped_down<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    hidden: &[f16],
) -> Vec<bf16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.rows, experts.cols);
    let weights = experts.upload::<B>(&context);
    let tables = upload_tables::<B>(routes, &context);
    let hidden = create_buffer_with_data::<B, f16>(&context, hidden);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    let mut outputs = create_buffer::<B, bf16>(&context, routes_count * model_dim as usize);
    let kernel = <B::Kernels as Kernels>::MoeGroupedDownKernel::new(&context, DataType::BF16, experts.format).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &hidden,
        &weights.codes,
        &weights.scales,
        weights.global_scales.as_ref(),
        &tables.expert_counts,
        &tables.expert_routes,
        &mut outputs,
        routes.tokens,
        experts.experts,
        model_dim,
        hidden_dim,
        &mut command_buffer,
    );
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &outputs)
}

// (model_dim, hidden_dim, experts, tokens): Inkling's shapes over few experts, and a narrow block over Inkling's 128
// experts where some experts receive more than one tile of routes.
const SHAPES: [(u32, u32, u32, &[u32]); 2] =
    [(INKLING_MODEL_DIM, INKLING_HIDDEN_DIM, 16, &[33, 40]), (512, 256, 128, &[64, 600])];

fn compare_grouped(format: MoeExpertFormat) {
    for (model_dim, hidden_dim, experts, token_counts) in SHAPES {
        let up = HostExperts::random(format, experts, 2 * hidden_dim, model_dim, 51);
        let down = HostExperts::random(format, experts, model_dim, hidden_dim, 52);
        for &tokens in token_counts {
            let routes = HostRoutes::random(tokens, experts, INKLING_TOP_K, 0.5, 61 + u64::from(tokens));
            let label = format!("{format:?}, {model_dim}x{hidden_dim}, {experts} experts, {tokens} tokens");
            let input = random_bf16((tokens * model_dim) as usize, 71 + u64::from(tokens));
            let expected_hidden = run_grouped_up::<Cpu>(&up, &routes, &input);
            let hidden = random_f16((tokens * INKLING_TOP_K * hidden_dim) as usize, 2.0, 81 + u64::from(tokens));
            let expected_outputs = run_grouped_down::<Cpu>(&down, &routes, &hidden);
            // The matrix units take NVFP4 weights exactly, while affine weights round to bfloat16 for the up projection
            // (as the reference dequantization does) and to half for the down projection.
            let tolerance = if format == MoeExpertFormat::Nvfp4 {
                4e-3
            } else {
                2.5e-2
            };
            for_each_non_cpu_backend!(|B| {
                assert_parity(
                    &format!("{label}: up"),
                    &expected_hidden,
                    &run_grouped_up::<B>(&up, &routes, &input),
                    tolerance,
                    tolerance,
                );
                assert_parity(
                    &format!("{label}: down"),
                    &expected_outputs,
                    &run_grouped_down::<B>(&down, &routes, &hidden),
                    1.6e-2,
                    2e-2,
                );
            });
        }
    }
}

#[uzu_test]
fn moe_grouped_nvfp4_matches_cpu() {
    compare_grouped(MoeExpertFormat::Nvfp4);
}

#[uzu_test]
fn moe_grouped_affine_matches_cpu() {
    compare_grouped(MoeExpertFormat::AffineU4);
}

fn run_grouped_affine_up<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    input: &[bf16],
) -> Vec<f16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.cols, experts.rows / 2);
    let weights = experts.upload::<B>(&context);
    let tables = upload_tables::<B>(routes, &context);
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    let mut hidden =
        create_buffer_with_data::<B, f16>(&context, &vec![f16::from_f32(-7.0); routes_count * hidden_dim as usize]);
    let activation = INKLING_ACTIVATION;
    let kernel = <B::Kernels as Kernels>::MoeGroupedAffineUpKernel::new(&context, DataType::BF16, true, true).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &input,
        &weights.codes,
        &weights.scales,
        &tables.expert_counts,
        &tables.expert_routes,
        &tables.route_weights,
        &mut hidden,
        routes.tokens,
        experts.experts,
        model_dim,
        hidden_dim,
        routes.top_k,
        activation.up_soft_cap.unwrap(),
        activation.gate_soft_cap.unwrap(),
        activation.alpha,
        &mut command_buffer,
    );
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &hidden)
}

fn run_grouped_affine_down<B: Backend>(
    experts: &HostExperts,
    routes: &HostRoutes,
    hidden: &[f16],
) -> Vec<bf16> {
    let context = create_context::<B>();
    let (model_dim, hidden_dim) = (experts.rows, experts.cols);
    let weights = experts.upload::<B>(&context);
    let tables = upload_tables::<B>(routes, &context);
    let hidden = create_buffer_with_data::<B, f16>(&context, hidden);
    let routes_count = (routes.tokens * routes.top_k) as usize;
    let mut outputs = create_buffer::<B, bf16>(&context, routes_count * model_dim as usize);
    let kernel = <B::Kernels as Kernels>::MoeGroupedAffineDownKernel::new(&context, DataType::BF16).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &hidden,
        &weights.codes,
        &weights.scales,
        &tables.expert_counts,
        &tables.expert_routes,
        &mut outputs,
        routes.tokens,
        experts.experts,
        model_dim,
        hidden_dim,
        &mut command_buffer,
    );
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &outputs)
}

// The affine kernels that stage each weight tile once for all of an expert's routes, against the format-generic
// reference: few routes per expert, and experts with more than one pass of routes.
#[uzu_test]
fn moe_grouped_affine_kernels_match_cpu() {
    let format = MoeExpertFormat::AffineU4;
    for (model_dim, hidden_dim, experts, token_counts) in SHAPES {
        let up = HostExperts::random(format, experts, 2 * hidden_dim, model_dim, 53);
        let down = HostExperts::random(format, experts, model_dim, hidden_dim, 54);
        for &tokens in token_counts {
            let routes = HostRoutes::random(tokens, experts, INKLING_TOP_K, 0.5, 63 + u64::from(tokens));
            let label = format!("affine kernels, {model_dim}x{hidden_dim}, {experts} experts, {tokens} tokens");
            let input = random_bf16((tokens * model_dim) as usize, 73 + u64::from(tokens));
            let expected_hidden = run_grouped_up::<Cpu>(&up, &routes, &input);
            let hidden = random_f16((tokens * INKLING_TOP_K * hidden_dim) as usize, 2.0, 83 + u64::from(tokens));
            let expected_outputs = run_grouped_down::<Cpu>(&down, &routes, &hidden);
            assert_eq!(expected_hidden, run_grouped_affine_up::<Cpu>(&up, &routes, &input), "{label}: CPU up twins");
            for_each_non_cpu_backend!(|B| {
                // Up weights round to bfloat16, as the reference dequantization does, and down weights to half.
                assert_parity(
                    &format!("{label}: up"),
                    &expected_hidden,
                    &run_grouped_affine_up::<B>(&up, &routes, &input),
                    2.5e-2,
                    2.5e-2,
                );
                assert_parity(
                    &format!("{label}: down"),
                    &expected_outputs,
                    &run_grouped_affine_down::<B>(&down, &routes, &hidden),
                    1.6e-2,
                    2e-2,
                );
            });
        }
    }
}

fn run_combine<B: Backend>(
    outputs: &[bf16],
    tokens: u32,
    model_dim: u32,
    top_k: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let outputs = create_buffer_with_data::<B, bf16>(&context, outputs);
    let mut combined = create_buffer::<B, bf16>(&context, (tokens * model_dim) as usize);
    let kernel = <B::Kernels as Kernels>::MoeCombineRoutesKernel::new(&context, DataType::BF16).unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(&outputs, &mut combined, tokens, model_dim, top_k, &mut command_buffer);
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &combined)
}

#[uzu_test]
fn moe_combine_routes_matches_cpu() {
    for tokens in [33u32, 257] {
        let outputs = random_bf16((tokens * INKLING_TOP_K * INKLING_MODEL_DIM) as usize, 91 + u64::from(tokens));
        let expected = run_combine::<Cpu>(&outputs, tokens, INKLING_MODEL_DIM, INKLING_TOP_K);
        for_each_non_cpu_backend!(|B| {
            assert_parity(
                &format!("{tokens} tokens"),
                &expected,
                &run_combine::<B>(&outputs, tokens, INKLING_MODEL_DIM, INKLING_TOP_K),
                8e-3,
                8e-3,
            );
        });
    }
}

fn run_block<B: Backend>(
    host: &HostMoe,
    input: &[bf16],
    tokens: u32,
) -> Vec<bf16> {
    let context = create_context::<B>();
    let moe = MixtureOfExpertsMlp::<B>::from_parts(host.parts::<B>(&context), &context)
        .unwrap_or_else(|error| panic!("cannot build the block: {error}"));
    let input = create_buffer_with_data::<B, bf16>(&context, input);
    let mut output = create_buffer::<B, bf16>(&context, (tokens * host.model_dim) as usize);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    moe.encode_rows(&input, &mut output, tokens, &mut command_buffer).unwrap();
    submit::<B>(command_buffer);
    buffer_to_vec::<B, _>(&context, &output)
}

// The whole block, router included, over Inkling's 128 experts at a narrow width: the small-batch path at 1, 8 and
// 32 tokens and the grouped path beyond.
#[uzu_test]
fn moe_block_matches_cpu() {
    for format in [MoeExpertFormat::Nvfp4, MoeExpertFormat::AffineU4] {
        let host = HostMoe::random(format, 512, 256, INKLING_EXPERTS, INKLING_TOP_K, 101);
        for tokens in [1u32, 8, 32, 33, 200] {
            let input = random_bf16((tokens * host.model_dim) as usize, 111 + u64::from(tokens));
            let expected = run_block::<Cpu>(&host, &input, tokens);
            for_each_non_cpu_backend!(|B| {
                assert_parity(
                    &format!("{format:?} block, {tokens} tokens"),
                    &expected,
                    &run_block::<B>(&host, &input, tokens),
                    2e-2,
                    3e-2,
                );
            });
        }
    }
}
