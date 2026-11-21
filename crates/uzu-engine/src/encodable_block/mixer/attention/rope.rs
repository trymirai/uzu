use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding, Context, CpuBuffer, GlobalBufferCpuGpu},
    config::rope::AnyRoPEConfig,
};

pub struct RoPE<B: Backend> {
    pub inverse_frequencies: B::GlobalGpuBuffer,
    pub scale: f32,
    pub dim: u32,
    pub max_sequence_length: u32,
}

/// Positions the rows of one batch rotate by, in place of the ones that follow the context
pub struct RoPEPositions<B: Backend> {
    /// `[row, dim / 2]`
    pub positions: B::ConstantBuffer,
}

impl<B: Backend> RoPEPositions<B> {
    /// `position(row, pair_index)` is the position frequency pair `pair_index` of row `row` rotates by
    pub fn new(
        row_count: usize,
        dim: u32,
        position: impl Fn(usize, u32) -> u32,
        command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
    ) -> Result<Self, B::Error> {
        let positions = (0..row_count)
            .flat_map(|row| (0..dim / 2).map(move |pair_index| (row, pair_index)))
            .map(|(row, pair_index)| position(row, pair_index))
            .collect::<Box<[u32]>>();

        Ok(Self {
            positions: command_buffer.allocate_constant_from_slice(&positions)?,
        })
    }
}

impl<B: Backend> RoPE<B> {
    pub fn new(
        rope_config: &AnyRoPEConfig,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        let head_dim = *rope_config.head_dim();
        assert!(head_dim > 0 && head_dim.is_multiple_of(2), "RoPE head_dim must be positive and even");
        let half_dim = head_dim / 2;
        let scale = match rope_config {
            AnyRoPEConfig::YARNRoPEConfig(config) => 0.1 * config.scaling_factor.ln() + 1.0,
            AnyRoPEConfig::LongRoPEConfig(config) if config.scaling_factor > 1.0 => {
                (1.0 + config.scaling_factor.ln() / (config.original_context_length as f32).ln()).sqrt()
            },
            _ => 1.0,
        };

        let mut inverse_frequencies = Vec::with_capacity(half_dim as usize);
        for pair_index in 0..half_dim {
            let channel_index = pair_index * 2;
            let inverse_frequency = 1.0 / rope_config.base().powf(channel_index as f32 / head_dim as f32);
            let inverse_frequency = match rope_config {
                AnyRoPEConfig::UnscaledRoPEConfig(_) | AnyRoPEConfig::MultimodalRoPEConfig(_) => inverse_frequency,
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

            inverse_frequencies.push(inverse_frequency);
        }

        Self::from_inverse_frequencies(
            &inverse_frequencies,
            scale,
            head_dim,
            *rope_config.max_sequence_length(),
            context,
        )
    }

    /// `inverse_frequencies` has one frequency per pair of the `dim` rotated channels
    pub fn from_inverse_frequencies(
        inverse_frequencies: &[f32],
        scale: f32,
        dim: u32,
        max_sequence_length: u32,
        context: &B::Context,
    ) -> Result<Self, B::Error> {
        assert_eq!(inverse_frequencies.len(), (dim / 2) as usize, "RoPE takes a frequency per channel pair");
        let mut buffer = context.create_buffer_cpu_gpu(size_of_val(inverse_frequencies))?;
        buffer.copyin(inverse_frequencies);
        Ok(Self {
            inverse_frequencies: buffer.handoff()?,
            scale,
            dim,
            max_sequence_length,
        })
    }
}
