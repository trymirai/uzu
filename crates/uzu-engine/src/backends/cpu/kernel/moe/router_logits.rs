use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(MoeRouterLogits)]
#[variants(T, bf16)]
pub fn moe_router_logits<T: DataTypeRepr + Float>(
    input: *const T,
    router_weights: *const f32,
    #[optional(has_router_biases)] router_biases: Option<*const f32>,
    logits: *mut f32,
    tokens: u32,
    model_dim: u32,
    experts: u32,
    #[specialize] has_router_biases: bool,
) {
    assert_eq!(router_biases.is_some(), has_router_biases);
    let (tokens, model_dim, experts) = (tokens as usize, model_dim as usize, experts as usize);
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
            unsafe {
                *logits.add(token * experts + expert) = logit;
            }
        }
    }
}
