//! Rope (Rotary Position Embedding) encodable.

use std::ops::{Deref, DerefMut};

use crate::{
    DataType,
    backends::common::{
        Backend, Encoder,
        kernel::{Kernels, RopeKernel},
    },
    config::RoPEConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
};

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
enum RopeScalingType {
    Unscaled = 0,
    Linear = 1,
    Llama = 2,
    Yarn = 3,
}

#[derive(Clone, Copy, Debug)]
struct RopeKernelConfig {
    base: f32,
    max_sequence_length: usize,
    rotary_dim: Option<usize>,
    rotary_frequency_dim: Option<usize>,
    uses_proportional_layout: bool,
    scaling_type: RopeScalingType,
    scaling_factor: f32,
    original_context_length: usize,
    low_frequency_factor: f32,
    high_frequency_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    truncate: bool,
    attention_scaling_factor: f32,
}

impl RopeKernelConfig {
    fn new(
        rope_config: &RoPEConfig,
        attention_partial_rope_dim: Option<usize>,
    ) -> Self {
        let common = rope_config.common();
        let rotary_dim = common.partial_rotary_dim.or(attention_partial_rope_dim).or(common.head_dim);
        let uses_proportional_layout =
            rotary_dim.zip(common.head_dim).is_some_and(|(rotary_dim, head_dim)| rotary_dim < head_dim);

        match rope_config {
            RoPEConfig::Unscaled(_) => Self {
                base: common.base,
                max_sequence_length: common.max_sequence_length,
                rotary_dim,
                rotary_frequency_dim: common.head_dim,
                uses_proportional_layout,
                scaling_type: RopeScalingType::Unscaled,
                scaling_factor: 1.0,
                original_context_length: common.max_sequence_length,
                low_frequency_factor: 1.0,
                high_frequency_factor: 1.0,
                beta_fast: 1.0,
                beta_slow: 1.0,
                truncate: false,
                attention_scaling_factor: 1.0,
            },
            RoPEConfig::LinearScalingRoPEConfig {
                scaling_factor,
                ..
            } => Self {
                base: common.base,
                max_sequence_length: common.max_sequence_length,
                rotary_dim,
                rotary_frequency_dim: common.head_dim,
                uses_proportional_layout,
                scaling_type: RopeScalingType::Linear,
                scaling_factor: *scaling_factor,
                original_context_length: common.max_sequence_length,
                low_frequency_factor: 1.0,
                high_frequency_factor: 1.0,
                beta_fast: 1.0,
                beta_slow: 1.0,
                truncate: false,
                attention_scaling_factor: 1.0,
            },
            RoPEConfig::Llama {
                scaling_factor,
                original_context_length,
                low_frequency_factor,
                high_frequency_factor,
                ..
            } => Self {
                base: common.base,
                max_sequence_length: common.max_sequence_length,
                rotary_dim,
                rotary_frequency_dim: common.head_dim,
                uses_proportional_layout,
                scaling_type: RopeScalingType::Llama,
                scaling_factor: *scaling_factor,
                original_context_length: *original_context_length,
                low_frequency_factor: *low_frequency_factor,
                high_frequency_factor: *high_frequency_factor,
                beta_fast: 1.0,
                beta_slow: 1.0,
                truncate: false,
                attention_scaling_factor: 1.0,
            },
            RoPEConfig::YARN {
                scaling_factor,
                original_context_length,
                beta_fast,
                beta_slow,
                truncate,
                ..
            } => Self {
                base: common.base,
                max_sequence_length: common.max_sequence_length,
                rotary_dim,
                rotary_frequency_dim: common.head_dim,
                uses_proportional_layout,
                scaling_type: RopeScalingType::Yarn,
                scaling_factor: *scaling_factor,
                original_context_length: *original_context_length,
                low_frequency_factor: 1.0,
                high_frequency_factor: 1.0,
                beta_fast: *beta_fast,
                beta_slow: *beta_slow,
                truncate: *truncate,
                attention_scaling_factor: 0.1 * scaling_factor.ln() + 1.0,
            },
        }
    }
}

pub struct Rope<B: Backend> {
    kernel: <B::Kernels as Kernels>::RopeKernel,
    config: RopeKernelConfig,
}

impl<B: Backend> Rope<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        rope_config: &RoPEConfig,
        attention_partial_rope_dim: Option<usize>,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            kernel: <B::Kernels as Kernels>::RopeKernel::new(context, data_type)?,
            config: RopeKernelConfig::new(rope_config, attention_partial_rope_dim),
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        use_rope: bool,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let token_positions = state.array(ArrayId::TokenPositions);
        let qkv = state.array(ArrayId::QKV);
        let rotated_queries = state.array(ArrayId::RotatedQueries);
        let rotated_keys = state.array(ArrayId::RotatedKeys);

        let suffix_length = qkv.shape()[0];
        let num_heads = rotated_queries.shape()[0];
        let head_dim = rotated_queries.shape()[2];
        let num_groups = rotated_keys.shape()[0];
        let rope_dim = if use_rope {
            self.config.rotary_dim.unwrap_or(head_dim)
        } else {
            0
        };

        let rotary_frequency_dim = self.config.rotary_frequency_dim.unwrap_or(rope_dim);
        // Proportional NeoX RoPE keeps rotary pairs in the full-head layout:
        // for head_dim=512 and rope_dim=128 the pair stride is 256, not 64.
        let rotary_pair_stride = if rope_dim == 0 {
            0
        } else if self.config.uses_proportional_layout && rotary_frequency_dim == head_dim {
            head_dim / 2
        } else {
            rope_dim / 2
        };

        self.kernel.encode(
            qkv.buffer().borrow().deref(),
            (token_positions.buffer().borrow().deref(), token_positions.offset()),
            rotated_queries.buffer().borrow_mut().deref_mut(),
            rotated_keys.buffer().borrow_mut().deref_mut(),
            head_dim as u32,
            rope_dim as u32,
            rotary_pair_stride as u32,
            rotary_frequency_dim as u32,
            self.config.max_sequence_length as u32,
            self.config.scaling_type as u32,
            self.config.base,
            self.config.scaling_factor,
            self.config.original_context_length as u32,
            self.config.low_frequency_factor,
            self.config.high_frequency_factor,
            self.config.beta_fast,
            self.config.beta_slow,
            u32::from(self.config.truncate),
            self.config.attention_scaling_factor,
            num_heads as u32,
            num_groups as u32,
            suffix_length as u32,
            encoder,
        );
        Ok(())
    }
}
