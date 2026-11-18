use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding},
    config::rope::AnyRoPEConfig,
};

pub struct PrecalculatedRoPE<B: Backend> {
    pub cosines: B::ConstantBuffer,
    pub sines: B::ConstantBuffer,
    pub dim: u32,
}

/// Interleaved multimodal RoPE (Qwen3-VL style): each token has `[temporal, height, width]` positions.
/// RoPE pair `i` rotates:
///  - by the height position if `i % 3 == 1 && i < 3 * sections[1]`,
///  - by the width position if `i % 3 == 2 && i < 3 * sections[2]`,
///  - by the temporal position otherwise.
pub struct MultimodalRoPEPositions<'a> {
    pub sections: [u32; 3],
    pub positions: &'a [[u32; 3]],
}

impl<B: Backend> PrecalculatedRoPE<B> {
    pub fn precalculate(
        rope_config: &AnyRoPEConfig,
        token_positions: &[u32],
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        Self::precalculate_with(
            rope_config,
            token_positions.len(),
            |token_index, _pair_index| token_positions[token_index],
            command_buffer,
        )
    }

    pub fn precalculate_multimodal(
        rope_config: &AnyRoPEConfig,
        multimodal_rope: &MultimodalRoPEPositions,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        let [_, height_section, width_section] = multimodal_rope.sections;
        assert_eq!(
            multimodal_rope.sections.iter().sum::<u32>(),
            *rope_config.head_dim() / 2,
            "multimodal RoPE sections must cover every RoPE pair"
        );
        Self::precalculate_with(
            rope_config,
            multimodal_rope.positions.len(),
            |token_index, pair_index| {
                let axis = match pair_index % 3 {
                    1 if pair_index < 3 * height_section => 1,
                    2 if pair_index < 3 * width_section => 2,
                    _ => 0,
                };
                multimodal_rope.positions[token_index][axis]
            },
            command_buffer,
        )
    }

    /// `position(token_index, pair_index)` is the position RoPE pair `pair_index` of token `token_index` rotates by.
    fn precalculate_with(
        rope_config: &AnyRoPEConfig,
        token_count: usize,
        position: impl Fn(usize, u32) -> u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        let head_dim = *rope_config.head_dim();
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "RoPE head_dim must be positive and even");
        let half_dim = head_dim / 2;
        let attention_scaling_factor = match rope_config {
            AnyRoPEConfig::YARNRoPEConfig(config) => 0.1 * config.scaling_factor.ln() + 1.0,
            AnyRoPEConfig::LongRoPEConfig(config) if config.scaling_factor > 1.0 => {
                (1.0 + config.scaling_factor.ln() / (config.original_context_length as f32).ln()).sqrt()
            },
            _ => 1.0,
        };

        let element_count = token_count * head_dim as usize;
        let half_offset = half_dim as usize;
        let mut sines = vec![0.0; element_count];
        let mut cosines = vec![0.0; element_count];
        for pair_index in 0..half_dim {
            let channel_index = pair_index * 2;
            let inverse_frequency = 1.0 / rope_config.base().powf(channel_index as f32 / head_dim as f32);
            let inverse_frequency = match rope_config {
                AnyRoPEConfig::UnscaledRoPEConfig(_) => inverse_frequency,
                AnyRoPEConfig::LinearScalingRoPEConfig(config) => inverse_frequency / config.scaling_factor,
                AnyRoPEConfig::LlamaRoPEConfig(config) => {
                    let low_frequency_wavelength = config.original_context_length as f32 / config.low_frequency_factor;
                    let high_frequency_wavelength =
                        config.original_context_length as f32 / config.high_frequency_factor;
                    let wavelength = 2.0 * std::f32::consts::PI / inverse_frequency;
                    let scaled_frequency = inverse_frequency / config.scaling_factor;

                    if wavelength < high_frequency_wavelength {
                        inverse_frequency
                    } else if wavelength > low_frequency_wavelength {
                        scaled_frequency
                    } else {
                        let smoothing_factor =
                            config.original_context_length as f32 / wavelength - config.low_frequency_factor;
                        let smoothing_factor =
                            smoothing_factor / (config.high_frequency_factor - config.low_frequency_factor);
                        smoothing_factor * inverse_frequency + (1.0 - smoothing_factor) * scaled_frequency
                    }
                },
                AnyRoPEConfig::YARNRoPEConfig(config) => {
                    let dim = config.head_dim as f64;
                    let base = config.base as f64;
                    let original_context_length = config.original_context_length as f64;
                    let beta_fast = config.beta_fast as f64;
                    let beta_slow = config.beta_slow as f64;
                    let mut low = dim * (original_context_length / (beta_fast * 2.0 * std::f64::consts::PI)).ln()
                        / (2.0 * base.ln());
                    let mut high = dim * (original_context_length / (beta_slow * 2.0 * std::f64::consts::PI)).ln()
                        / (2.0 * base.ln());
                    if config.truncate {
                        low = low.floor();
                        high = high.ceil();
                    }
                    let low = low.max(0.0) as f32;
                    let mut high = high.min((config.head_dim - 1) as f64) as f32;
                    if low == high {
                        high += 0.001;
                    }
                    let ramp = ((pair_index as f32 - low) / (high - low)).clamp(0.0, 1.0);
                    let smoothing_factor = 1.0 - ramp;
                    let scaled_frequency = inverse_frequency / config.scaling_factor;
                    scaled_frequency * (1.0 - smoothing_factor) + inverse_frequency * smoothing_factor
                },
                AnyRoPEConfig::LongRoPEConfig(config) => {
                    let factors = if config.max_sequence_length > config.original_context_length {
                        &config.long_factor
                    } else {
                        &config.short_factor
                    };
                    inverse_frequency / factors[pair_index as usize]
                },
            };

            for token_index in 0..token_count {
                let embedding = position(token_index, pair_index) as f32 * inverse_frequency;
                let sine = embedding.sin() * attention_scaling_factor;
                let cosine = embedding.cos() * attention_scaling_factor;
                let pair_offset = token_index * head_dim as usize + pair_index as usize;
                sines[pair_offset] = sine;
                sines[pair_offset + half_offset] = sine;
                cosines[pair_offset] = cosine;
                cosines[pair_offset + half_offset] = cosine;
            }
        }

        Ok(Self {
            cosines: command_buffer.allocate_constant_from_slice(&cosines)?,
            sines: command_buffer.allocate_constant_from_slice(&sines)?,
            dim: *rope_config.head_dim(),
        })
    }
}
