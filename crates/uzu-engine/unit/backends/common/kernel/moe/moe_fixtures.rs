//! Synthetic Inkling-shaped expert stacks and routing tables for the mixture-of-experts kernel tests and benches.

use half::{bf16, f16};

use crate::{
    backends::common::{
        Backend,
        gpu_types::moe::{MoeExpertFormat, moe_active_entry},
    },
    data_type::DataType,
    encodable_block::mlp::moe::{
        ExpertWeights, MixtureOfExpertsParts, MoeActivation, MoeRouting, RouterWeights, weights::repack_nvfp4_codes,
    },
    tests::helpers::create_buffer_with_data,
};

/// Inkling 2 Mini: 2304-wide model, 128 routed experts of hidden width 1152, eight active per token.
pub const INKLING_MODEL_DIM: u32 = 2304;
pub const INKLING_HIDDEN_DIM: u32 = 1152;
pub const INKLING_EXPERTS: u32 = 128;
pub const INKLING_TOP_K: u32 = 8;
pub const INKLING_ROUTED_SCALING_FACTOR: f32 = 8.0;
pub const INKLING_ACTIVATION: MoeActivation = MoeActivation {
    up_soft_cap: Some(25.0),
    gate_soft_cap: Some(4.0),
    alpha: 1.0,
};

/// Deterministic xorshift stream.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1).
    pub fn symmetric(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    pub fn below(
        &mut self,
        bound: u32,
    ) -> u32 {
        (self.next_u64() % u64::from(bound)) as u32
    }

    pub fn bytes(
        &mut self,
        count: usize,
    ) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count + 8);
        while bytes.len() < count {
            bytes.extend_from_slice(&self.next_u64().to_le_bytes());
        }
        bytes.truncate(count);
        bytes
    }
}

/// Host copy of one stacked expert projection `[experts, rows, cols]`.
pub struct HostExperts {
    pub format: MoeExpertFormat,
    pub experts: u32,
    pub rows: u32,
    pub cols: u32,
    pub codes: Vec<u8>,
    /// E4M3 bytes for NVFP4, little-endian (scale, bias) bfloat16 words for affine.
    pub scales: Vec<u8>,
    pub global_scales: Option<Vec<f32>>,
}

impl HostExperts {
    /// Random weights whose dot products with inputs in [-1, 1] come out near 1, NVFP4 codes repacked as the loader
    /// leaves them.
    pub fn random(
        format: MoeExpertFormat,
        experts: u32,
        rows: u32,
        cols: u32,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let values = experts as usize * rows as usize * cols as usize;
        let mut codes = rng.bytes(values / 2);
        let spread = 1.0 / (cols as f32).sqrt();
        if format == MoeExpertFormat::Nvfp4 {
            repack_nvfp4_codes(&mut codes);
        }
        match format {
            MoeExpertFormat::Nvfp4 => {
                // E4M3 bytes 0x30..0x40 are 0.5..1.875.
                let scales = (0..values / 16).map(|_| 0x30 + rng.below(16) as u8).collect();
                let global_scales = (0..experts).map(|_| spread * 0.5 * (1.0 + 0.25 * rng.symmetric())).collect();
                Self {
                    format,
                    experts,
                    rows,
                    cols,
                    codes,
                    scales,
                    global_scales: Some(global_scales),
                }
            },
            MoeExpertFormat::AffineU4 => {
                let groups = cols / 64;
                let mut scales = Vec::with_capacity(values / 64 * 4);
                for _ in 0..experts as usize * rows as usize * groups as usize {
                    let scale = spread * 0.25 * (1.0 + 0.5 * rng.symmetric());
                    let bias = -7.5 * scale * (1.0 + 0.1 * rng.symmetric());
                    let word =
                        u32::from(bf16::from_f32(scale).to_bits()) | (u32::from(bf16::from_f32(bias).to_bits()) << 16);
                    scales.extend_from_slice(&word.to_le_bytes());
                }
                Self {
                    format,
                    experts,
                    rows,
                    cols,
                    codes,
                    scales,
                    global_scales: None,
                }
            },
        }
    }

    /// Bytes the kernels read per expert.
    pub fn bytes_per_expert(&self) -> usize {
        (self.codes.len() + self.scales.len()) / self.experts as usize
    }

    pub fn upload<B: Backend>(
        &self,
        context: &B::Context,
    ) -> ExpertWeights<B> {
        ExpertWeights {
            format: self.format,
            codes: create_buffer_with_data::<B, u8>(context, &self.codes),
            scales: create_buffer_with_data::<B, u8>(context, &self.scales),
            global_scales: self.global_scales.as_ref().map(|scales| create_buffer_with_data::<B, f32>(context, scales)),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Routing tables in the layout the routing kernels write, each expert's routes in increasing order.
pub struct HostRoutes {
    pub tokens: u32,
    pub top_k: u32,
    pub route_weights: Vec<f32>,
    pub expert_counts: Vec<u32>,
    pub expert_routes: Vec<u32>,
    pub active_experts: Vec<u32>,
}

impl HostRoutes {
    /// Every token picks `top_k` distinct experts. `overlap` in [0, 1] steers how often consecutive tokens repeat
    /// each other's picks, as tokens of one speculative block tend to.
    pub fn random(
        tokens: u32,
        experts: u32,
        top_k: u32,
        overlap: f32,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let mut picks: Vec<Vec<u32>> = Vec::new();
        for token in 0..tokens as usize {
            let mut chosen: Vec<u32> = Vec::new();
            while chosen.len() < top_k as usize {
                let candidate = if token > 0 && (rng.symmetric() + 1.0) * 0.5 < overlap {
                    picks[token - 1][rng.below(top_k) as usize]
                } else {
                    rng.below(experts)
                };
                if !chosen.contains(&candidate) {
                    chosen.push(candidate);
                }
            }
            picks.push(chosen);
        }
        let weights = (0..tokens * top_k).map(|_| 0.5 + 0.5 * (rng.symmetric() + 1.0)).collect();
        Self::from_picks(experts, top_k, &picks, weights)
    }

    pub fn from_picks(
        experts: u32,
        top_k: u32,
        picks: &[Vec<u32>],
        route_weights: Vec<f32>,
    ) -> Self {
        let tokens = picks.len() as u32;
        let mut expert_counts = vec![0u32; experts as usize];
        let mut expert_routes = vec![0u32; (experts * tokens) as usize];
        for (token, chosen) in picks.iter().enumerate() {
            assert_eq!(chosen.len(), top_k as usize);
            for (pick, &expert) in chosen.iter().enumerate() {
                let route = token as u32 * top_k + pick as u32;
                let count = &mut expert_counts[expert as usize];
                expert_routes[(expert * tokens + *count) as usize] = route;
                *count += 1;
            }
        }
        let mut active_experts = vec![0u32; experts as usize + 1];
        let mut active = 0;
        for (expert, &count) in expert_counts.iter().enumerate() {
            if count > 0 {
                let first_route = expert_routes[expert * tokens as usize];
                active_experts[1 + active] = moe_active_entry(expert as u32, count, first_route);
                active += 1;
            }
        }
        active_experts[0] = active as u32;
        Self {
            tokens,
            top_k,
            route_weights,
            expert_counts,
            expert_routes,
            active_experts,
        }
    }
}

pub fn random_bf16(
    count: usize,
    seed: u64,
) -> Vec<bf16> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| bf16::from_f32(rng.symmetric())).collect()
}

pub fn random_f16(
    count: usize,
    scale: f32,
    seed: u64,
) -> Vec<f16> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| f16::from_f32(scale * rng.symmetric())).collect()
}

/// Host copy of a whole block with random weights.
pub struct HostMoe {
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub experts: u32,
    pub top_k: u32,
    pub router_weights: Vec<f32>,
    pub expert_biases: Vec<f32>,
    pub up: HostExperts,
    pub down: HostExperts,
}

impl HostMoe {
    pub fn random(
        format: MoeExpertFormat,
        model_dim: u32,
        hidden_dim: u32,
        experts: u32,
        top_k: u32,
        seed: u64,
    ) -> Self {
        let mut rng = Rng::new(seed);
        let spread = 4.0 / (model_dim as f32).sqrt();
        Self {
            model_dim,
            hidden_dim,
            experts,
            top_k,
            router_weights: (0..experts * model_dim).map(|_| spread * rng.symmetric()).collect(),
            expert_biases: (0..experts).map(|_| 0.05 * rng.symmetric()).collect(),
            up: HostExperts::random(format, experts, 2 * hidden_dim, model_dim, seed + 1),
            down: HostExperts::random(format, experts, model_dim, hidden_dim, seed + 2),
        }
    }

    pub fn parts<B: Backend>(
        &self,
        context: &B::Context,
    ) -> MixtureOfExpertsParts<B> {
        MixtureOfExpertsParts {
            model_dim: self.model_dim,
            hidden_dim: self.hidden_dim,
            experts: self.experts,
            data_type: DataType::BF16,
            routing: MoeRouting {
                top_k: self.top_k,
                routed_scaling_factor: INKLING_ROUTED_SCALING_FACTOR,
                bias_selects: true,
                bias_weights: false,
            },
            activation: INKLING_ACTIVATION,
            router: RouterWeights {
                weights: create_buffer_with_data::<B, f32>(context, &self.router_weights),
                biases: None,
                expert_biases: Some(create_buffer_with_data::<B, f32>(context, &self.expert_biases)),
            },
            up: self.up.upload(context),
            down: self.down.upload(context),
        }
    }
}
