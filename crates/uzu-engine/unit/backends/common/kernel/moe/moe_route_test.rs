use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use super::moe_fixtures::{
    INKLING_EXPERTS, INKLING_MODEL_DIM, INKLING_ROUTED_SCALING_FACTOR, INKLING_TOP_K, Rng, random_bf16,
};
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::moe::moe_active_entry_parts, kernel::MoeRouteKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend},
    },
};

struct RouteCase<'a> {
    input: &'a [bf16],
    router_weights: &'a [f32],
    expert_biases: Option<&'a [f32]>,
    tokens: u32,
    model_dim: u32,
    experts: u32,
    top_k: u32,
    bias_selects: bool,
    bias_weights: bool,
}

#[derive(Debug)]
struct RouteResult {
    logits: Vec<f32>,
    route_experts: Vec<u32>,
    route_weights: Vec<f32>,
    expert_counts: Vec<u32>,
    /// Each expert's routes, sorted: the kernels promise no order within an expert.
    expert_routes: Vec<Vec<u32>>,
    /// (expert, route count) of every active expert; each entry's first route was checked against the expert's list.
    active_experts: Vec<(u32, u32)>,
    finish_counter: u32,
}

/// Decodes the active-expert table into (expert, count), checking that every entry's first route is one of the
/// expert's routes (`expert_routes` sorted per expert).
pub fn decode_active(
    table: &[u32],
    expert_routes: &[Vec<u32>],
) -> Vec<(u32, u32)> {
    table[1..=table[0] as usize]
        .iter()
        .map(|&entry| {
            let (expert, count, first_route) = moe_active_entry_parts(entry);
            assert!(
                expert_routes[expert as usize].contains(&first_route),
                "expert {expert}: first route {first_route}"
            );
            (expert, count)
        })
        .collect()
}

/// The router kernel variants: the largest batch each takes.
const MAX_TOKENS: [u32; 6] = [1, 2, 4, 8, 16, 32];

fn run_route<B: Backend>(
    case: &RouteCase,
    max_tokens: u32,
) -> RouteResult {
    let context = create_context::<B>();
    let (tokens, experts) = (case.tokens, case.experts);
    let routes = (tokens * case.top_k) as usize;
    let input = create_buffer_with_data::<B, bf16>(&context, case.input);
    let router_weights = create_buffer_with_data::<B, f32>(&context, case.router_weights);
    let expert_biases = case.expert_biases.map(|biases| create_buffer_with_data::<B, f32>(&context, biases));
    let mut logits = create_buffer::<B, f32>(&context, (tokens * experts) as usize);
    let mut route_experts = create_buffer::<B, u32>(&context, routes);
    let mut route_weights = create_buffer::<B, f32>(&context, routes);
    let mut expert_counts = create_buffer::<B, u32>(&context, experts as usize);
    let mut expert_routes = create_buffer::<B, u32>(&context, (experts * tokens) as usize);
    let mut active_experts = create_buffer::<B, u32>(&context, experts as usize + 1);
    let mut finish_counter = create_buffer_with_data::<B, u32>(&context, &[0]);

    let kernel = <B::Kernels as Kernels>::MoeRouteKernel::new(
        &context,
        DataType::BF16,
        max_tokens,
        false,
        expert_biases.is_some(),
        case.bias_selects,
        case.bias_weights,
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &input,
        &router_weights,
        None::<&B::GlobalGpuBuffer>,
        expert_biases.as_ref(),
        &mut logits,
        &mut route_experts,
        &mut route_weights,
        &mut expert_counts,
        &mut expert_routes,
        &mut active_experts,
        &mut finish_counter,
        tokens,
        case.model_dim,
        experts,
        case.top_k,
        INKLING_ROUTED_SCALING_FACTOR,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    let expert_counts: Vec<u32> = buffer_to_vec::<B, _>(&context, &expert_counts);
    let all_routes: Vec<u32> = buffer_to_vec::<B, _>(&context, &expert_routes);
    let expert_routes: Vec<Vec<u32>> = (0..experts as usize)
        .map(|expert| {
            let start = expert * tokens as usize;
            let mut routes = all_routes[start..start + expert_counts[expert] as usize].to_vec();
            routes.sort_unstable();
            routes
        })
        .collect();
    let active_experts: Vec<u32> = buffer_to_vec::<B, _>(&context, &active_experts);
    let active_experts = decode_active(&active_experts, &expert_routes);
    RouteResult {
        logits: buffer_to_vec::<B, _>(&context, &logits),
        route_experts: buffer_to_vec::<B, _>(&context, &route_experts),
        route_weights: buffer_to_vec::<B, _>(&context, &route_weights),
        expert_counts,
        expert_routes,
        active_experts,
        finish_counter: buffer_to_vec::<B, u32>(&context, &finish_counter)[0],
    }
}

fn assert_routes_match(
    expected: &RouteResult,
    actual: &RouteResult,
    label: &str,
) {
    assert_eq_float(&expected.logits, &actual.logits, 1e-4, &format!("{label}: logits"));
    assert_eq!(expected.route_experts, actual.route_experts, "{label}: route experts");
    assert_eq_float(&expected.route_weights, &actual.route_weights, 1e-5, &format!("{label}: route weights"));
    assert_eq!(expected.expert_counts, actual.expert_counts, "{label}: expert counts");
    assert_eq!(expected.expert_routes, actual.expert_routes, "{label}: expert routes");
    assert_eq!(expected.active_experts, actual.active_experts, "{label}: active experts");
    assert_eq!(actual.finish_counter, 0, "{label}: the last threadgroup resets the counter");
}

fn router_weights(
    experts: u32,
    model_dim: u32,
    seed: u64,
) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    let spread = 4.0 / (model_dim as f32).sqrt();
    (0..experts * model_dim).map(|_| spread * rng.symmetric()).collect()
}

// Inkling's router at 1 (decode), 2, 8 (a speculative block) and 32 tokens, with the gating bias.
#[uzu_test]
fn moe_route_matches_cpu() {
    let (experts, model_dim) = (INKLING_EXPERTS, INKLING_MODEL_DIM);
    let weights = router_weights(experts, model_dim, 7);
    let mut rng = Rng::new(11);
    let biases: Vec<f32> = (0..experts).map(|_| 0.05 * rng.symmetric()).collect();
    for tokens in [1u32, 2, 8, 32] {
        let input = random_bf16((tokens * model_dim) as usize, 100 + u64::from(tokens));
        for (bias_selects, bias_weights) in [(true, false), (false, true), (true, true)] {
            let case = RouteCase {
                input: &input,
                router_weights: &weights,
                expert_biases: Some(&biases),
                tokens,
                model_dim,
                experts,
                top_k: INKLING_TOP_K,
                bias_selects,
                bias_weights,
            };
            let expected = run_route::<Cpu>(&case, 32);
            let sums: Vec<f32> =
                expected.route_weights.chunks(INKLING_TOP_K as usize).map(|row| row.iter().sum()).collect();
            if !bias_weights {
                assert_eq_float(
                    &vec![INKLING_ROUTED_SCALING_FACTOR; tokens as usize],
                    &sums,
                    1e-3,
                    "route weights sum to the scaling factor",
                );
            }
            for max_tokens in MAX_TOKENS.into_iter().filter(|&max_tokens| tokens <= max_tokens) {
                for_each_non_cpu_backend!(|B| {
                    let actual = run_route::<B>(&case, max_tokens);
                    assert_routes_match(
                        &expected,
                        &actual,
                        &format!(
                            "{tokens} tokens up to {max_tokens}, bias selects {bias_selects}, bias weights {bias_weights}"
                        ),
                    );
                });
            }
        }
    }
}

// Unbiased routing over 64 experts, and twin experts with identical router rows: their scores tie exactly, and
// the lower index goes first. One token takes the single-token routing path, five the general one.
#[uzu_test]
fn moe_route_ties_and_no_bias() {
    let (experts, model_dim) = (64u32, 512u32);
    let row = model_dim as usize;
    for tokens in [1u32, 5] {
        let input = random_bf16((tokens * model_dim) as usize, 5 + u64::from(tokens));
        let mut weights = router_weights(experts, model_dim, 3);
        if tokens == 1 {
            // Rows along the input give experts 9 and 40 the top score, so that the single token meets the tie.
            for (index, value) in input.iter().enumerate() {
                weights[9 * row + index] = 0.1 * value.to_f32();
            }
        }
        for (twin, original) in [(40usize, 9usize), (17, 50), (63, 0)] {
            let source = weights[original * row..(original + 1) * row].to_vec();
            weights[twin * row..(twin + 1) * row].copy_from_slice(&source);
        }
        let case = RouteCase {
            input: &input,
            router_weights: &weights,
            expert_biases: None,
            tokens,
            model_dim,
            experts,
            top_k: 6,
            bias_selects: false,
            bias_weights: false,
        };
        let expected = run_route::<Cpu>(&case, 32);
        if tokens == 1 {
            assert_eq!(expected.route_experts[..2], [9, 40], "the tied experts lead, the lower first");
        }
        for (token, picks) in expected.route_experts.chunks(6).enumerate() {
            for (twin, original) in [(40u32, 9u32), (17, 50), (63, 0)] {
                let (low, high) = (twin.min(original), twin.max(original));
                if let Some(high_pick) = picks.iter().position(|&expert| expert == high) {
                    let low_pick = picks.iter().position(|&expert| expert == low);
                    assert!(
                        low_pick.is_some_and(|low_pick| low_pick < high_pick),
                        "token {token}: {low} ranks above {high}"
                    );
                }
            }
        }
        for max_tokens in MAX_TOKENS.into_iter().filter(|&max_tokens| tokens <= max_tokens) {
            for_each_non_cpu_backend!(|B| {
                assert_routes_match(
                    &expected,
                    &run_route::<B>(&case, max_tokens),
                    &format!("ties, {tokens} tokens up to {max_tokens}"),
                );
            });
        }
    }
}
