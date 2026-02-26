use crate::audio::{AudioError, AudioResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    Zeros,
    Replicate,
}

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
}

fn validate_lengths(
    lengths: &[i32],
    batch_size: usize,
    frames: usize,
) -> AudioResult<()> {
    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }

    for &length in lengths {
        if length < 0 || length as usize > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length: length.max(0) as usize,
                frames,
            });
        }
    }

    Ok(())
}

pub struct Conv1dSpec<'a> {
    pub input: &'a [f32],
    pub weight: &'a [f32],
    pub bias: &'a [f32],
    pub lengths: &'a [i32],
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub dilation: usize,
    pub padding: usize,
    pub pad_mode: PadMode,
}

pub fn conv1d_reference(spec: Conv1dSpec<'_>) -> AudioResult<Vec<f32>> {
    let Conv1dSpec {
        input,
        weight,
        bias,
        lengths,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        kernel_size,
        stride,
        dilation,
        padding,
        pad_mode,
    } = spec;

    validate_lengths(lengths, batch_size, seq_len_out)?;

    let expected_input = checked_product(&[batch_size, cin, seq_len_in])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }

    let expected_weight = checked_product(&[cout, cin, kernel_size])?;
    if weight.len() != expected_weight {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_weight,
            actual_tokens: weight.len(),
        });
    }

    if bias.len() != cout {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: cout,
            actual_tokens: bias.len(),
        });
    }

    let output_len = checked_product(&[batch_size, cout, seq_len_out])?;
    let mut output = vec![0.0_f32; output_len];

    for batch in 0..batch_size {
        let length = lengths[batch] as usize;
        for out_channel in 0..cout {
            for out_time in 0..seq_len_out {
                let out_index = (batch * cout + out_channel) * seq_len_out + out_time;
                if out_time >= length {
                    output[out_index] = 0.0;
                    continue;
                }

                let mut acc = bias[out_channel];
                let base = (out_time * stride) as isize - padding as isize;

                for in_channel in 0..cin {
                    let input_base = (batch * cin + in_channel) * seq_len_in;
                    let weight_base = (out_channel * cin + in_channel) * kernel_size;
                    for kernel_offset in 0..kernel_size {
                        let x_time_raw = base + (kernel_offset * dilation) as isize;
                        let x_time = match pad_mode {
                            PadMode::Zeros => {
                                if x_time_raw < 0 || x_time_raw >= seq_len_in as isize {
                                    continue;
                                }
                                x_time_raw as usize
                            },
                            PadMode::Replicate => x_time_raw.clamp(0, seq_len_in as isize - 1) as usize,
                        };

                        let x_index = input_base + x_time;
                        let w_index = weight_base + kernel_offset;
                        acc += weight[w_index] * input[x_index];
                    }
                }

                output[out_index] = acc;
            }
        }
    }

    Ok(output)
}

pub struct CausalConv1dSpec<'a> {
    pub input: &'a [f32],
    pub weight: &'a [f32],
    pub bias: &'a [f32],
    pub lengths: &'a [i32],
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len: usize,
    pub kernel_size: usize,
    pub dilation: usize,
}

pub fn causal_conv1d_reference(spec: CausalConv1dSpec<'_>) -> AudioResult<Vec<f32>> {
    let CausalConv1dSpec {
        input,
        weight,
        bias,
        lengths,
        batch_size,
        cin,
        cout,
        seq_len,
        kernel_size,
        dilation,
    } = spec;

    validate_lengths(lengths, batch_size, seq_len)?;

    let expected_input = checked_product(&[batch_size, cin, seq_len])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }

    let expected_weight = checked_product(&[cout, cin, kernel_size])?;
    if weight.len() != expected_weight {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_weight,
            actual_tokens: weight.len(),
        });
    }

    if bias.len() != cout {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: cout,
            actual_tokens: bias.len(),
        });
    }

    let output_len = checked_product(&[batch_size, cout, seq_len])?;
    let mut output = vec![0.0_f32; output_len];
    let pad = (kernel_size - 1) * dilation;

    for batch in 0..batch_size {
        let length = lengths[batch] as usize;
        for out_channel in 0..cout {
            for time in 0..seq_len {
                let out_index = (batch * cout + out_channel) * seq_len + time;
                if time >= length {
                    output[out_index] = 0.0;
                    continue;
                }

                let mut acc = bias[out_channel];
                for in_channel in 0..cin {
                    let input_base = (batch * cin + in_channel) * seq_len;
                    let weight_base = (out_channel * cin + in_channel) * kernel_size;
                    for kernel_offset in 0..kernel_size {
                        let x_time = time as isize + (kernel_offset * dilation) as isize - pad as isize;
                        if x_time < 0 || x_time >= seq_len as isize {
                            continue;
                        }

                        let x_index = input_base + x_time as usize;
                        let w_index = weight_base + kernel_offset;
                        acc += weight[w_index] * input[x_index];
                    }
                }

                output[out_index] = acc;
            }
        }
    }

    Ok(output)
}

pub struct CausalConvTranspose1dSpec<'a> {
    pub input: &'a [f32],
    pub weight: &'a [f32],
    pub bias: &'a [f32],
    pub lengths: &'a [i32],
    pub batch_size: usize,
    pub cin: usize,
    pub cout: usize,
    pub seq_len_in: usize,
    pub seq_len_out: usize,
    pub stride: usize,
    pub groups: usize,
}

pub fn causal_conv_transpose1d_reference(spec: CausalConvTranspose1dSpec<'_>) -> AudioResult<Vec<f32>> {
    let CausalConvTranspose1dSpec {
        input,
        weight,
        bias,
        lengths,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        stride,
        groups,
    } = spec;

    validate_lengths(lengths, batch_size, seq_len_out)?;
    if groups == 0 || cin == 0 || cout == 0 || cin % groups != 0 || cout % groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let expected_input = checked_product(&[batch_size, cin, seq_len_in])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }

    let kernel_size = 2 * stride;
    let expected_weight = checked_product(&[cin, cout / groups, kernel_size])?;
    if weight.len() != expected_weight {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_weight,
            actual_tokens: weight.len(),
        });
    }

    if bias.len() != cout {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: cout,
            actual_tokens: bias.len(),
        });
    }

    let output_len = checked_product(&[batch_size, cout, seq_len_out])?;
    let mut output = vec![0.0_f32; output_len];
    let cout_per_group = cout / groups;
    let cin_per_group = cin / groups;

    for batch in 0..batch_size {
        let length = lengths[batch] as usize;
        for out_channel in 0..cout {
            let group_index = out_channel / cout_per_group;
            let out_channel_in_group = out_channel % cout_per_group;
            let in_channel_begin = group_index * cin_per_group;
            let in_channel_end = in_channel_begin + cin_per_group;

            for out_time in 0..seq_len_out {
                let out_index = (batch * cout + out_channel) * seq_len_out + out_time;
                if out_time >= length {
                    output[out_index] = 0.0;
                    continue;
                }

                let q = out_time / stride;
                let r = out_time % stride;
                let mut acc = bias[out_channel];

                for in_channel in in_channel_begin..in_channel_end {
                    if q >= seq_len_in {
                        continue;
                    }

                    let input_base = (batch * cin + in_channel) * seq_len_in;
                    let weight_base = (in_channel * cout_per_group + out_channel_in_group) * kernel_size;

                    acc += input[input_base + q] * weight[weight_base + r];
                    if q > 0 {
                        acc += input[input_base + (q - 1)] * weight[weight_base + (stride + r)];
                    }
                }

                output[out_index] = acc;
            }
        }
    }

    Ok(output)
}

pub fn causal_conv_transpose1d_lalamo_reference(spec: CausalConvTranspose1dSpec<'_>) -> AudioResult<Vec<f32>> {
    let CausalConvTranspose1dSpec {
        input,
        weight,
        bias,
        lengths,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        stride,
        groups,
    } = spec;

    validate_lengths(lengths, batch_size, seq_len_out)?;
    if groups == 0 || stride == 0 || cin == 0 || cout == 0 || cin % groups != 0 || cout % groups != 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let expected_input = checked_product(&[batch_size, cin, seq_len_in])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }

    let cout_per_group = cout / groups;
    let weight_plane = checked_product(&[cin, cout_per_group])?;
    if weight_plane == 0 || weight.len() % weight_plane != 0 {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: weight_plane,
            actual_tokens: weight.len(),
        });
    }

    let kernel_size = weight.len() / weight_plane;
    if kernel_size == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    if bias.len() != cout {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: cout,
            actual_tokens: bias.len(),
        });
    }

    let output_len = checked_product(&[batch_size, cout, seq_len_out])?;
    let mut output = vec![0.0_f32; output_len];
    let cin_per_group = cin / groups;
    let seq_len_expanded = if seq_len_in == 0 {
        0
    } else {
        (seq_len_in - 1)
            .checked_mul(stride)
            .and_then(|value| value.checked_add(1))
            .ok_or(AudioError::Runtime("expanded sequence length overflow".to_string()))?
    };
    let left_pad = kernel_size - 1;

    for batch in 0..batch_size {
        let length = lengths[batch] as usize;
        for out_channel in 0..cout {
            let group_index = out_channel / cout_per_group;
            let out_channel_in_group = out_channel % cout_per_group;
            let in_channel_begin = group_index * cin_per_group;
            let in_channel_end = in_channel_begin + cin_per_group;

            for out_time in 0..seq_len_out {
                let out_index = (batch * cout + out_channel) * seq_len_out + out_time;
                if out_time >= length {
                    output[out_index] = 0.0;
                    continue;
                }

                let mut acc = bias[out_channel];

                for in_channel in in_channel_begin..in_channel_end {
                    let input_base = (batch * cin + in_channel) * seq_len_in;
                    let weight_base = (in_channel * cout_per_group + out_channel_in_group) * kernel_size;

                    for kernel_offset in 0..kernel_size {
                        let expanded_time = out_time as isize + kernel_offset as isize - left_pad as isize;
                        if expanded_time < 0 || expanded_time >= seq_len_expanded as isize {
                            continue;
                        }

                        let expanded_time = expanded_time as usize;
                        if expanded_time % stride != 0 {
                            continue;
                        }

                        let src_time = expanded_time / stride;
                        if src_time >= seq_len_in {
                            continue;
                        }

                        acc += input[input_base + src_time] * weight[weight_base + kernel_offset];
                    }
                }

                output[out_index] = acc;
            }
        }
    }

    Ok(output)
}

pub struct HalfSnakeSpec<'a> {
    pub input: &'a [f32],
    pub alpha: &'a [f32],
    pub batch_size: usize,
    pub channels: usize,
    pub seq_len: usize,
    pub snake_channels: usize,
    pub negative_slope: f32,
    pub eps: f32,
}

pub fn half_snake_reference(spec: HalfSnakeSpec<'_>) -> AudioResult<Vec<f32>> {
    let HalfSnakeSpec {
        input,
        alpha,
        batch_size,
        channels,
        seq_len,
        snake_channels,
        negative_slope,
        eps,
    } = spec;

    if snake_channels > channels {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let expected_input = checked_product(&[batch_size, channels, seq_len])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }

    if alpha.len() != snake_channels {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: snake_channels,
            actual_tokens: alpha.len(),
        });
    }

    let mut output = vec![0.0_f32; input.len()];
    for batch in 0..batch_size {
        for channel in 0..channels {
            for time in 0..seq_len {
                let index = (batch * channels + channel) * seq_len + time;
                let x = input[index];
                output[index] = if channel < snake_channels {
                    let a = alpha[channel];
                    let sine = (a * x).sin();
                    x + (sine * sine) / (a + eps)
                } else if x >= 0.0 {
                    x
                } else {
                    negative_slope * x
                };
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::{
        CausalConv1dSpec, CausalConvTranspose1dSpec, Conv1dSpec, HalfSnakeSpec, PadMode,
        causal_conv_transpose1d_lalamo_reference, causal_conv_transpose1d_reference, causal_conv1d_reference,
        conv1d_reference, half_snake_reference,
    };

    #[test]
    fn conv1d_reference_zeros_padding_matches_expected() {
        let output = conv1d_reference(Conv1dSpec {
            input: &[1.0, 2.0, 3.0, 4.0],
            weight: &[1.0, 0.0, -1.0],
            bias: &[0.0],
            lengths: &[4],
            batch_size: 1,
            cin: 1,
            cout: 1,
            seq_len_in: 4,
            seq_len_out: 4,
            kernel_size: 3,
            stride: 1,
            dilation: 1,
            padding: 1,
            pad_mode: PadMode::Zeros,
        })
        .expect("conv1d");

        assert_eq!(output, vec![-2.0, -2.0, -2.0, 3.0]);
    }

    #[test]
    fn causal_conv1d_reference_matches_expected() {
        let output = causal_conv1d_reference(CausalConv1dSpec {
            input: &[1.0, 2.0, 3.0, 4.0],
            weight: &[1.0, 1.0, 1.0],
            bias: &[0.0],
            lengths: &[4],
            batch_size: 1,
            cin: 1,
            cout: 1,
            seq_len: 4,
            kernel_size: 3,
            dilation: 1,
        })
        .expect("causal conv1d");

        assert_eq!(output, vec![1.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn causal_conv_transpose_reference_matches_expected() {
        let output = causal_conv_transpose1d_reference(CausalConvTranspose1dSpec {
            input: &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            weight: &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0],
            bias: &[0.0],
            lengths: &[6],
            batch_size: 1,
            cin: 2,
            cout: 1,
            seq_len_in: 3,
            seq_len_out: 6,
            stride: 2,
            groups: 1,
        })
        .expect("causal conv transpose");

        assert_eq!(output, vec![1.0, 12.0, 5.0, 38.0, 9.0, 64.0]);
    }

    #[test]
    fn causal_conv_transpose_lalamo_reference_matches_expected() {
        let output = causal_conv_transpose1d_lalamo_reference(CausalConvTranspose1dSpec {
            input: &[1.0, 2.0, 3.0],
            weight: &[1.0, 2.0, 3.0, 4.0],
            bias: &[0.0],
            lengths: &[6],
            batch_size: 1,
            cin: 1,
            cout: 1,
            seq_len_in: 3,
            seq_len_out: 6,
            stride: 2,
            groups: 1,
        })
        .expect("causal conv transpose lalamo");

        assert_eq!(output, vec![4.0, 3.0, 10.0, 7.0, 16.0, 11.0]);
    }

    #[test]
    fn half_snake_reference_matches_expected() {
        let input = [0.0, core::f32::consts::FRAC_PI_2, 1.0, -1.0, -2.0, 2.0, -3.0, 3.0];
        let output = half_snake_reference(HalfSnakeSpec {
            input: &input,
            alpha: &[1.0, 2.0],
            batch_size: 1,
            channels: 4,
            seq_len: 2,
            snake_channels: 2,
            negative_slope: 0.1,
            eps: 1e-9,
        })
        .expect("half snake");

        let expected = [0.0, core::f32::consts::FRAC_PI_2 + 1.0, 1.4134109, -0.5865891, -0.2, 2.0, -0.3, 3.0];

        for (actual, expected) in output.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() <= 1e-5);
        }
    }
}
