use crate::config::rope::AnyRoPEConfig;

pub fn precalculate_rope(
    rope_config: &AnyRoPEConfig,
    token_positions: &[usize],
) -> (Box<[f32]>, Box<[f32]>) {
    let head_dim = *rope_config.head_dim();
    assert!(head_dim > 0 && head_dim.is_multiple_of(2), "RoPE head_dim must be positive and even");
    let half_dim = head_dim / 2;
    let attention_scaling_factor = match rope_config {
        AnyRoPEConfig::YARNRoPEConfig(config) => 0.1 * config.scaling_factor.ln() + 1.0,
        _ => 1.0,
    };

    let element_count = token_positions.len() * head_dim;
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
                let high_frequency_wavelength = config.original_context_length as f32 / config.high_frequency_factor;
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
                let mut low =
                    dim * (original_context_length / (beta_fast * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
                let mut high =
                    dim * (original_context_length / (beta_slow * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
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
        };

        for (token_index, token_position) in token_positions.iter().enumerate() {
            let embedding = *token_position as f32 * inverse_frequency;
            let sine = embedding.sin() * attention_scaling_factor;
            let cosine = embedding.cos() * attention_scaling_factor;
            let row_offset = token_index * head_dim;
            sines[row_offset + pair_index] = sine;
            sines[row_offset + half_dim + pair_index] = sine;
            cosines[row_offset + pair_index] = cosine;
            cosines[row_offset + half_dim + pair_index] = cosine;
        }
    }

    (sines.into_boxed_slice(), cosines.into_boxed_slice())
}
