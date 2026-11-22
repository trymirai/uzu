//! Scalar reference arithmetic shared by the CPU twins of the mixture-of-experts kernels.

use half::bf16;

use crate::backends::common::gpu_types::moe::{
    MOE_MAX_EXPERTS, MOE_MAX_TOP_K, MoeExpertFormat, moe_active_entry, moe_active_entry_parts, nvfp4_repacked_code,
};

const E2M1_VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

pub const NVFP4_GROUP_SIZE: usize = 16;
pub const AFFINE_GROUP_SIZE: usize = 64;

pub fn e2m1(code: u8) -> f32 {
    E2M1_VALUES[usize::from(code & 0xF)]
}

pub fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 {
        -1.0
    } else {
        1.0
    };
    let exponent = i32::from((byte >> 3) & 0xF);
    let mantissa = f32::from(byte & 0x7);
    if exponent == 0 {
        sign * mantissa / 8.0 * 2f32.powi(-6)
    } else {
        sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One expert matrix of a stacked `[experts, rows, cols]` tensor, as the expert kernels read it: NVFP4 codes in the
/// repacked word layout, affine codes in checkpoint order.
pub struct ExpertMatrix {
    pub format: MoeExpertFormat,
    pub codes: *const u8,
    pub scales: *const u8,
    pub global_scale: f32,
    pub cols: usize,
}

impl ExpertMatrix {
    /// # Safety
    /// The buffers must hold `experts` stacked matrices of the format, with `expert` one of them.
    pub unsafe fn new(
        format: MoeExpertFormat,
        codes: *const u8,
        scales: *const u8,
        global_scales: Option<*const f32>,
        expert: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        let scale_bytes = match format {
            MoeExpertFormat::Nvfp4 => rows * cols / NVFP4_GROUP_SIZE,
            MoeExpertFormat::AffineU4 => rows * cols / AFFINE_GROUP_SIZE * 4,
        };
        unsafe {
            Self {
                format,
                codes: codes.add(expert * rows * cols / 2),
                scales: scales.add(expert * scale_bytes),
                global_scale: global_scales.map_or(1.0, |scales| *scales.add(expert)),
                cols,
            }
        }
    }

    /// The dequantized weight at (row, col).
    pub fn weight(
        &self,
        row: usize,
        col: usize,
    ) -> f32 {
        match self.format {
            MoeExpertFormat::Nvfp4 => {
                let word = unsafe { self.codes.add(row * self.cols / 2 + col / 8 * 4).cast::<u32>().read_unaligned() };
                let code = nvfp4_repacked_code(u32::from_le(word), (col % 8) as u32);
                let scale = unsafe { *self.scales.add(row * self.cols / NVFP4_GROUP_SIZE + col / NVFP4_GROUP_SIZE) };
                e2m1(code) * e4m3(scale) * self.global_scale
            },
            MoeExpertFormat::AffineU4 => {
                let byte = unsafe { *self.codes.add(row * self.cols / 2 + col / 2) };
                let code = if col.is_multiple_of(2) {
                    byte & 0xF
                } else {
                    byte >> 4
                };
                let pair = unsafe {
                    self.scales
                        .cast::<u32>()
                        .add(row * self.cols / AFFINE_GROUP_SIZE + col / AFFINE_GROUP_SIZE)
                        .read_unaligned()
                };
                let scale = bf16::from_bits(pair as u16).to_f32();
                let bias = bf16::from_bits((pair >> 16) as u16).to_f32();
                f32::from(code) * scale + bias
            },
        }
    }

    /// The dequantized weights of `row`.
    pub fn row(
        &self,
        row: usize,
    ) -> Vec<f32> {
        (0..self.cols).map(|col| self.weight(row, col)).collect()
    }
}

/// The float32 dot product of `weights` with `values`, summed in order.
pub fn dot(
    weights: &[f32],
    values: &[f32],
) -> f32 {
    weights.iter().zip(values).fold(0.0, |acc, (&weight, &value)| acc + weight * value)
}

/// The gated expert activation in float32, as the expert kernels apply it.
pub fn expert_activation(
    up: f32,
    gate: f32,
    up_soft_cap: Option<f32>,
    gate_soft_cap: Option<f32>,
    activation_alpha: f32,
) -> f32 {
    let up = up_soft_cap.map_or(up, |cap| cap * (up / cap).tanh());
    let linear = gate_soft_cap.map_or(gate, |cap| cap * (gate / cap).tanh());
    up * linear * sigmoid(activation_alpha * gate)
}

/// The (expert, routes) of every active-expert entry, the routes in table order.
///
/// # Safety
/// The tables must hold `tokens` tokens' routing.
pub unsafe fn active_expert_routes(
    active_experts: *const u32,
    expert_routes: *const u32,
    tokens: usize,
) -> Vec<(usize, Vec<usize>)> {
    let active = unsafe { *active_experts } as usize;
    (0..active)
        .map(|slot| {
            let (expert, count, first_route) = moe_active_entry_parts(unsafe { *active_experts.add(1 + slot) });
            let routes = (0..count as usize)
                .map(|index| {
                    if index == 0 {
                        first_route as usize
                    } else {
                        unsafe { *expert_routes.add(expert as usize * tokens + index) as usize }
                    }
                })
                .collect();
            (expert as usize, routes)
        })
        .collect()
}

/// Routing tables, as the routing kernels write them.
pub struct RouteOutputs {
    pub route_experts: *mut u32,
    pub route_weights: *mut f32,
    pub expert_counts: *mut u32,
    pub expert_routes: *mut u32,
    pub active_experts: *mut u32,
}

pub struct RouteConfig<'a> {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
    pub routed_scaling_factor: f32,
    pub expert_biases: Option<&'a [f32]>,
    pub bias_selects: bool,
    pub bias_weights: bool,
}

/// Selects each token's experts from its sigmoid scores and groups the routes by expert, each expert's routes in
/// increasing order.
///
/// # Safety
/// The outputs must hold the routing tables of `config.tokens` tokens.
pub unsafe fn route_tokens(
    logits: &[f32],
    config: &RouteConfig,
    outputs: &RouteOutputs,
) {
    assert!(config.experts <= MOE_MAX_EXPERTS as usize);
    assert!(config.top_k > 0 && config.top_k <= (MOE_MAX_TOP_K as usize).min(config.experts));
    let mut counts = vec![0u32; config.experts];
    for token in 0..config.tokens {
        let row = &logits[token * config.experts..(token + 1) * config.experts];
        let scores: Vec<f32> = row.iter().map(|&logit| sigmoid(logit)).collect();
        let bias = |expert: usize| config.expert_biases.map_or(0.0, |biases| biases[expert]);
        let gating = |expert: usize| {
            let score = if config.bias_selects {
                scores[expert] + bias(expert)
            } else {
                scores[expert]
            };
            if score.is_nan() {
                f32::NEG_INFINITY
            } else {
                score
            }
        };
        let mut order: Vec<usize> = (0..config.experts).collect();
        order.sort_by(|&a, &b| gating(b).total_cmp(&gating(a)).then(a.cmp(&b)));
        let chosen = &order[..config.top_k];
        let weights: Vec<f32> = chosen
            .iter()
            .map(|&expert| {
                if config.bias_weights {
                    scores[expert] + bias(expert)
                } else {
                    scores[expert]
                }
            })
            .collect();
        let mut sum = 0.0f32;
        for &weight in &weights {
            sum += weight;
        }
        for (pick, (&expert, &weight)) in chosen.iter().zip(&weights).enumerate() {
            let route = token * config.top_k + pick;
            unsafe {
                *outputs.route_experts.add(route) = expert as u32;
                *outputs.route_weights.add(route) = weight / (sum + 1e-6) * config.routed_scaling_factor;
                *outputs.expert_routes.add(expert * config.tokens + counts[expert] as usize) = route as u32;
            }
            counts[expert] += 1;
        }
    }
    let mut active = 0usize;
    for (expert, &count) in counts.iter().enumerate() {
        unsafe {
            *outputs.expert_counts.add(expert) = count;
            if count > 0 {
                let first_route = *outputs.expert_routes.add(expert * config.tokens);
                *outputs.active_experts.add(1 + active) = moe_active_entry(expert as u32, count, first_route);
                active += 1;
            }
        }
    }
    unsafe {
        *outputs.active_experts = active as u32;
    }
}
