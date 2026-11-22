use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{
    backends::cpu::kernel::moe::expert_math::{RouteConfig, RouteOutputs, route_tokens},
    data_type::DataTypeRepr,
};

#[kernel(MoeRoute)]
#[variants(T, bf16)]
#[variants(MAX_TOKENS, 1, 2, 4, 8, 16, 32)]
pub fn moe_route<T: DataTypeRepr + Float, const MAX_TOKENS: u32>(
    input: *const T,
    router_weights: *const f32,
    #[optional(has_router_biases)] router_biases: Option<*const f32>,
    #[optional(has_expert_biases)] expert_biases: Option<*const f32>,
    logits: *mut f32,
    route_experts: *mut u32,
    route_weights: *mut f32,
    expert_counts: *mut u32,
    expert_routes: *mut u32,
    active_experts: *mut u32,
    finish_counter: *mut u32,
    tokens: u32,
    model_dim: u32,
    experts: u32,
    top_k: u32,
    routed_scaling_factor: f32,
    #[specialize] has_router_biases: bool,
    #[specialize] has_expert_biases: bool,
    #[specialize] bias_selects: bool,
    #[specialize] bias_weights: bool,
) {
    assert_eq!(router_biases.is_some(), has_router_biases);
    assert_eq!(expert_biases.is_some(), has_expert_biases);
    assert!(tokens <= MAX_TOKENS);
    let (tokens, model_dim, experts) = (tokens as usize, model_dim as usize, experts as usize);
    let logits = unsafe { std::slice::from_raw_parts_mut(logits, tokens * experts) };
    for token in 0..tokens {
        for expert in 0..experts {
            let mut logit = 0.0f32;
            for k in 0..model_dim {
                let x = unsafe { *input.add(token * model_dim + k) }.to_f32().unwrap();
                logit += unsafe { *router_weights.add(expert * model_dim + k) } * x;
            }
            if let Some(biases) = router_biases {
                logit += unsafe { *biases.add(expert) };
            }
            logits[token * experts + expert] = logit;
        }
    }
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
    unsafe {
        route_tokens(logits, &config, &outputs);
        *finish_counter = 0;
    }
}
