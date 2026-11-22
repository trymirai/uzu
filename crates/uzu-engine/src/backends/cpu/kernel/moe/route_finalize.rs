use uzu_engine_macros::kernel;

use crate::backends::cpu::kernel::moe::expert_math::{RouteConfig, RouteOutputs, route_tokens};

#[kernel(MoeRouteFinalize)]
pub fn moe_route_finalize(
    logits: *const f32,
    #[optional(has_expert_biases)] expert_biases: Option<*const f32>,
    route_experts: *mut u32,
    route_weights: *mut f32,
    expert_counts: *mut u32,
    expert_routes: *mut u32,
    active_experts: *mut u32,
    tokens: u32,
    experts: u32,
    top_k: u32,
    routed_scaling_factor: f32,
    #[specialize] has_expert_biases: bool,
    #[specialize] bias_selects: bool,
    #[specialize] bias_weights: bool,
) {
    assert_eq!(expert_biases.is_some(), has_expert_biases);
    let (tokens, experts) = (tokens as usize, experts as usize);
    let logits = unsafe { std::slice::from_raw_parts(logits, tokens * experts) };
    let expert_biases = expert_biases.map(|biases| unsafe { std::slice::from_raw_parts(biases, experts) });
    let config = RouteConfig {
        tokens,
        experts,
        top_k: top_k as usize,
        routed_scaling_factor,
        expert_biases,
        bias_selects,
        bias_weights,
    };
    let outputs = RouteOutputs {
        route_experts,
        route_weights,
        expert_counts,
        expert_routes,
        active_experts,
    };
    unsafe { route_tokens(logits, &config, &outputs) };
}
