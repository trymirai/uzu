//! Rope (Rotary Position Embedding) encodable.

use std::ops::{Deref, DerefMut};

use crate::{
    Array, DataType,
    array::ArrayContextExt,
    backends::common::{
        Backend, Encoder,
        kernel::{Kernels, RopeKernel},
    },
    config::RoPEConfig,
    forward_pass::state::{ArrayId, ForwardPassState},
};

#[derive(Clone, Debug)]
struct RopeKernelConfig {
    max_sequence_length: usize,
    attention_scaling_factor: f32,
    inverse_frequencies: Box<[f32]>,
}

impl RopeKernelConfig {
    fn new(
        rope_config: &RoPEConfig,
        fallback_rotary_frequency_dim: usize,
    ) -> Self {
        let common = rope_config.common();
        let rotary_frequency_dim = common.head_dim.unwrap_or(fallback_rotary_frequency_dim);
        match rope_config {
            RoPEConfig::Unscaled(_) => Self {
                max_sequence_length: common.max_sequence_length,
                attention_scaling_factor: 1.0,
                inverse_frequencies: build_inverse_frequencies(rope_config, rotary_frequency_dim),
            },
            RoPEConfig::LinearScalingRoPEConfig {
                ..
            } => Self {
                max_sequence_length: common.max_sequence_length,
                attention_scaling_factor: 1.0,
                inverse_frequencies: build_inverse_frequencies(rope_config, rotary_frequency_dim),
            },
            RoPEConfig::Llama {
                ..
            } => Self {
                max_sequence_length: common.max_sequence_length,
                attention_scaling_factor: 1.0,
                inverse_frequencies: build_inverse_frequencies(rope_config, rotary_frequency_dim),
            },
            RoPEConfig::YARN {
                scaling_factor,
                ..
            } => Self {
                max_sequence_length: common.max_sequence_length,
                attention_scaling_factor: 0.1 * scaling_factor.ln() + 1.0,
                inverse_frequencies: build_inverse_frequencies(rope_config, rotary_frequency_dim),
            },
        }
    }
}

pub struct Rope<B: Backend> {
    kernel: <B::Kernels as Kernels>::RopeKernel,
    config: RopeKernelConfig,
    inverse_frequencies: Array<B>,
}

impl<B: Backend> Rope<B> {
    pub fn new(
        context: &B::Context,
        data_type: DataType,
        rope_config: &RoPEConfig,
        fallback_rotary_frequency_dim: usize,
    ) -> Result<Self, B::Error> {
        let config = RopeKernelConfig::new(rope_config, fallback_rotary_frequency_dim);
        let inverse_frequencies = context.create_array_from(
            &[config.inverse_frequencies.len()],
            &config.inverse_frequencies,
            "rope_inv_freq",
        );

        Ok(Self {
            kernel: <B::Kernels as Kernels>::RopeKernel::new(context, data_type)?,
            config,
            inverse_frequencies,
        })
    }

    pub fn encode(
        &self,
        state: &mut ForwardPassState<B>,
        use_rope: bool,
        num_heads: usize,
        num_groups: usize,
        head_dim: usize,
        rope_dim: usize,
        rotary_pair_stride: usize,
        encoder: &mut Encoder<B>,
    ) -> Result<(), B::Error> {
        let token_positions = state.array(ArrayId::TokenPositions);
        let qkv = state.array(ArrayId::QKV);
        let rotated_queries = state.array(ArrayId::RotatedQueries);
        let rotated_keys = state.array(ArrayId::RotatedKeys);
        let inverse_frequencies = &self.inverse_frequencies;

        let suffix_length = qkv.shape()[0];

        let rope_dim = if use_rope {
            rope_dim
        } else {
            0
        };

        self.kernel.encode(
            qkv.buffer().borrow().deref(),
            (token_positions.buffer().borrow().deref(), token_positions.offset()),
            inverse_frequencies.buffer().borrow().deref(),
            rotated_queries.buffer().borrow_mut().deref_mut(),
            rotated_keys.buffer().borrow_mut().deref_mut(),
            head_dim as u32,
            rope_dim as u32,
            rotary_pair_stride as u32,
            inverse_frequencies.shape()[0] as u32,
            self.config.max_sequence_length as u32,
            self.config.attention_scaling_factor,
            num_heads as u32,
            num_groups as u32,
            suffix_length as u32,
            encoder,
        );
        Ok(())
    }
}

fn build_inverse_frequencies(
    rope_config: &RoPEConfig,
    rotary_frequency_dim: usize,
) -> Box<[f32]> {
    (0..rotary_frequency_dim / 2)
        .map(|frequency_index| inverse_frequency(rope_config, rotary_frequency_dim, frequency_index))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn inverse_frequency(
    rope_config: &RoPEConfig,
    rotary_frequency_dim: usize,
    frequency_index: usize,
) -> f32 {
    let common = rope_config.common();
    let exponent = (2 * frequency_index) as f32 / rotary_frequency_dim as f32;
    let value = 1.0 / common.base.powf(exponent);

    match rope_config {
        RoPEConfig::Unscaled(_) => value,
        RoPEConfig::LinearScalingRoPEConfig {
            scaling_factor,
            ..
        } => value / scaling_factor,
        RoPEConfig::Llama {
            scaling_factor,
            original_context_length,
            low_frequency_factor,
            high_frequency_factor,
            ..
        } => {
            let low_frequency_wavelength = *original_context_length as f32 / *low_frequency_factor;
            let high_frequency_wavelength = *original_context_length as f32 / *high_frequency_factor;
            let wavelength = 2.0 * std::f32::consts::PI / value;
            let scaled = value / *scaling_factor;
            if wavelength > low_frequency_wavelength {
                scaled
            } else if wavelength >= high_frequency_wavelength {
                let smoothing = *original_context_length as f32 / wavelength - *low_frequency_factor;
                let smoothing = smoothing / (*high_frequency_factor - *low_frequency_factor);
                smoothing * value + (1.0 - smoothing) * scaled
            } else {
                value
            }
        },
        RoPEConfig::YARN {
            scaling_factor,
            original_context_length,
            beta_fast,
            beta_slow,
            truncate,
            ..
        } => {
            let scaled = value / *scaling_factor;
            let correction_range = yarn_correction_range(
                *beta_fast,
                *beta_slow,
                rotary_frequency_dim,
                common.base,
                *original_context_length,
                *truncate,
            );
            let ramp = linear_ramp_factor(correction_range.0, correction_range.1, frequency_index as f32);
            let smoothing = 1.0 - ramp;
            scaled * (1.0 - smoothing) + value * smoothing
        },
    }
}

fn yarn_correction_range(
    beta_fast: f32,
    beta_slow: f32,
    dim: usize,
    base: f32,
    original_context_length: usize,
    truncate: bool,
) -> (f32, f32) {
    let correction_dim = |num_rotations: f32| {
        (dim as f32 * (original_context_length as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    let mut low = correction_dim(beta_fast);
    let mut high = correction_dim(beta_slow);
    if truncate {
        low = low.floor();
        high = high.ceil();
    }
    (low.max(0.0), high.min((dim - 1) as f32))
}

fn linear_ramp_factor(
    min_value: f32,
    mut max_value: f32,
    dim: f32,
) -> f32 {
    if min_value == max_value {
        max_value += 0.001;
    }
    ((dim - min_value) / (max_value - min_value)).clamp(0.0, 1.0)
}

#[cfg(test)]
#[path = "../../tests/unit/encodable_block/rope_test.rs"]
mod tests;
