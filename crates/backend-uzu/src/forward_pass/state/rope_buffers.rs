use half::{bf16, f16};

use crate::{
    DataType,
    array::{Array, ArrayContextExt},
    backends::common::Backend,
    config::RoPEConfig,
    forward_pass::model_shape::ModelShape,
    parameters::ParameterTree,
};

pub struct RopeBuffers<B: Backend> {
    /// [rope_max_sequence_length, rope_dim]
    pub cosines: Array<B>,
    /// [rope_max_sequence_length, rope_dim]
    pub sines: Array<B>,
}

impl<B: Backend> RopeBuffers<B> {
    pub fn new(
        context: &B::Context,
        rope_config: &RoPEConfig,
        model_shape: &ModelShape,
    ) -> Self {
        let common = rope_config.common();
        let rope_dim = common.head_dim.unwrap_or_else(|| model_shape.rope_dim());
        let rope_max_sequence_length = common.max_sequence_length;

        let mut buffers = Self {
            cosines: context.create_array_uninitialized(
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
                "rope_buffers_cosines",
            ),
            sines: context.create_array_uninitialized(
                &[rope_max_sequence_length, rope_dim],
                model_shape.activation_data_type(),
                "rope_buffers_sines",
            ),
        };
        buffers.fill_from_config(rope_config);
        buffers
    }

    pub fn update_data(
        &mut self,
        parameter_tree: &ParameterTree<B::Context>,
        rope_name: &str,
    ) {
        let Ok(rope_tree) = parameter_tree.subtree(rope_name) else {
            return;
        };

        let cosines_view = rope_tree.leaf_array("cosines").unwrap();
        self.cosines.copy_from_array(&cosines_view);

        let sines_view = rope_tree.leaf_array("sines").unwrap();
        self.sines.copy_from_array(&sines_view);
    }

    fn fill_from_config(
        &mut self,
        rope_config: &RoPEConfig,
    ) {
        let sequence_length = self.cosines.shape()[0];
        let rope_dim = self.cosines.shape()[1];
        let values = build_rope_values(rope_config, sequence_length, rope_dim);

        match self.cosines.data_type() {
            DataType::F32 => {
                let cosines = self.cosines.as_slice_mut::<f32>();
                let sines = self.sines.as_slice_mut::<f32>();
                for (index, (cosine, sine)) in values.into_iter().enumerate() {
                    cosines[index] = cosine;
                    sines[index] = sine;
                }
            },
            DataType::F16 => {
                let cosines = self.cosines.as_slice_mut::<f16>();
                let sines = self.sines.as_slice_mut::<f16>();
                for (index, (cosine, sine)) in values.into_iter().enumerate() {
                    cosines[index] = f16::from_f32(cosine);
                    sines[index] = f16::from_f32(sine);
                }
            },
            DataType::BF16 => {
                let cosines = self.cosines.as_slice_mut::<bf16>();
                let sines = self.sines.as_slice_mut::<bf16>();
                for (index, (cosine, sine)) in values.into_iter().enumerate() {
                    cosines[index] = bf16::from_f32(cosine);
                    sines[index] = bf16::from_f32(sine);
                }
            },
            data_type => panic!("Unsupported RoPE buffer data type: {data_type:?}"),
        }
    }
}

fn build_rope_values(
    rope_config: &RoPEConfig,
    sequence_length: usize,
    rope_dim: usize,
) -> Vec<(f32, f32)> {
    let common = rope_config.common();
    let half_rope_dim = rope_dim / 2;
    let mut inverse_frequencies = (0..half_rope_dim)
        .map(|index| 1.0 / common.base.powf((2 * index) as f32 / rope_dim as f32))
        .collect::<Vec<_>>();

    scale_inverse_frequencies(rope_config, &mut inverse_frequencies, rope_dim);
    if let Some(partial_rotary_dim) = common.partial_rotary_dim
        && partial_rotary_dim < rope_dim
    {
        for value in inverse_frequencies.iter_mut().skip(partial_rotary_dim / 2) {
            *value = 0.0;
        }
    }

    let attention_scaling_factor = yarn_attention_scaling_factor(rope_config);
    let mut result = Vec::with_capacity(sequence_length * rope_dim);
    for position in 0..sequence_length {
        let position = position as f32;
        for dimension in 0..rope_dim {
            let frequency = inverse_frequencies[dimension % half_rope_dim];
            let angle = position * frequency;
            result.push((angle.cos() * attention_scaling_factor, angle.sin() * attention_scaling_factor));
        }
    }
    result
}

fn scale_inverse_frequencies(
    rope_config: &RoPEConfig,
    inverse_frequencies: &mut [f32],
    rope_dim: usize,
) {
    match rope_config {
        RoPEConfig::Unscaled(_) => {},
        RoPEConfig::LinearScalingRoPEConfig {
            scaling_factor,
            ..
        } => {
            for value in inverse_frequencies {
                *value /= *scaling_factor;
            }
        },
        RoPEConfig::Llama {
            scaling_factor,
            original_context_length,
            low_frequency_factor,
            high_frequency_factor,
            ..
        } => {
            let low_frequency_wavelength = *original_context_length as f32 / *low_frequency_factor;
            let high_frequency_wavelength = *original_context_length as f32 / *high_frequency_factor;
            for value in inverse_frequencies {
                let wavelength = 2.0 * std::f32::consts::PI / *value;
                let scaled = *value / *scaling_factor;
                if wavelength > low_frequency_wavelength {
                    *value = scaled;
                } else if wavelength >= high_frequency_wavelength {
                    let smoothing = *original_context_length as f32 / wavelength - *low_frequency_factor;
                    let smoothing = smoothing / (*high_frequency_factor - *low_frequency_factor);
                    *value = smoothing * *value + (1.0 - smoothing) * scaled;
                }
            }
        },
        RoPEConfig::YARN {
            common,
            scaling_factor,
            original_context_length,
            beta_fast,
            beta_slow,
            truncate,
        } => {
            let scaled = inverse_frequencies.iter().map(|value| *value / *scaling_factor).collect::<Vec<_>>();
            let (low, high) = yarn_correction_range(
                *beta_fast,
                *beta_slow,
                rope_dim,
                common.base,
                *original_context_length,
                *truncate,
            );
            for (index, value) in inverse_frequencies.iter_mut().enumerate() {
                let ramp = linear_ramp_factor(low, high, index as f32);
                let smoothing = 1.0 - ramp;
                *value = scaled[index] * (1.0 - smoothing) + *value * smoothing;
            }
        },
    }
}

fn yarn_attention_scaling_factor(rope_config: &RoPEConfig) -> f32 {
    match rope_config {
        RoPEConfig::YARN {
            scaling_factor,
            ..
        } => 0.1 * scaling_factor.ln() + 1.0,
        _ => 1.0,
    }
}

fn yarn_correction_range(
    low_rot: f32,
    high_rot: f32,
    dim: usize,
    base: f32,
    original_context_length: usize,
    truncate: bool,
) -> (f32, f32) {
    let correction_dim = |num_rotations: f32| {
        (dim as f32 * (original_context_length as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * base.ln())
    };
    let mut low = correction_dim(low_rot);
    let mut high = correction_dim(high_rot);
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
#[path = "../../../tests/unit/forward_pass/rope_buffers_test.rs"]
mod tests;
