use uzu::audio::{AudioError, AudioResult};

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
}

fn round_ties_to_even(value: f32) -> f32 {
    let floor = value.floor();
    let fraction = value - floor;

    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if (floor as i64 & 1) != 0 {
        floor + 1.0
    } else {
        floor
    }
}

fn validate_fsq_common(
    batch_size: usize,
    num_groups: usize,
    seq_len: usize,
    codebook_dim: usize,
    num_levels: &[i32],
    lengths: &[i32],
) -> AudioResult<()> {
    if num_groups == 0 || codebook_dim == 0 {
        return Err(AudioError::InvalidTokenCardinality);
    }

    if num_levels.len() != codebook_dim {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: codebook_dim,
            actual_tokens: num_levels.len(),
        });
    }

    for &levels in num_levels {
        if levels <= 1 {
            return Err(AudioError::InvalidTokenCardinality);
        }
    }

    if lengths.len() != batch_size {
        return Err(AudioError::InvalidTokenLengths {
            expected_lengths: batch_size,
            actual_lengths: lengths.len(),
        });
    }

    for &length in lengths {
        if length < 0 || length as usize > seq_len {
            return Err(AudioError::InvalidTokenLengthValue {
                length: length.max(0) as usize,
                frames: seq_len,
            });
        }
    }

    Ok(())
}

pub fn fsq_decode_reference(
    tokens: &[i32],
    lengths: &[i32],
    batch_size: usize,
    num_groups: usize,
    seq_len: usize,
    codebook_dim: usize,
    num_levels: &[i32],
) -> AudioResult<Vec<f32>> {
    validate_fsq_common(batch_size, num_groups, seq_len, codebook_dim, num_levels, lengths)?;

    let expected_tokens = checked_product(&[batch_size, num_groups, seq_len])?;
    if tokens.len() != expected_tokens {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens,
            actual_tokens: tokens.len(),
        });
    }

    let output_len = checked_product(&[batch_size, num_groups * codebook_dim, seq_len])?;
    let mut output = vec![0.0_f32; output_len];

    for batch in 0..batch_size {
        let length = lengths[batch] as usize;
        for group in 0..num_groups {
            for time in 0..seq_len {
                if time >= length {
                    continue;
                }

                let token_index = (batch * num_groups + group) * seq_len + time;
                let token = tokens[token_index];
                let mut base = 1_i32;

                for dim in 0..codebook_dim {
                    let levels = num_levels[dim];
                    let scale = levels / 2;
                    let offset = scale;

                    let div = token / base;
                    let mut code_nonnegative = div % levels;
                    if code_nonnegative < 0 {
                        code_nonnegative += levels;
                    }

                    let value = (code_nonnegative - offset) as f32 / scale as f32;
                    let channel = group * codebook_dim + dim;
                    let output_index = (batch * (num_groups * codebook_dim) + channel) * seq_len + time;
                    output[output_index] = value;

                    base =
                        base.checked_mul(levels).ok_or(AudioError::Runtime("fsq decode base overflow".to_string()))?;
                }
            }
        }
    }

    Ok(output)
}

pub fn fsq_encode_reference(
    input: &[f32],
    lengths: &[i32],
    batch_size: usize,
    num_groups: usize,
    seq_len: usize,
    codebook_dim: usize,
    num_levels: &[i32],
    dim_base_index: &[i32],
    eps: f32,
) -> AudioResult<Vec<i32>> {
    validate_fsq_common(batch_size, num_groups, seq_len, codebook_dim, num_levels, lengths)?;

    if dim_base_index.len() != codebook_dim {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: codebook_dim,
            actual_tokens: dim_base_index.len(),
        });
    }

    let expected_input = checked_product(&[batch_size, num_groups * codebook_dim, seq_len])?;
    if input.len() != expected_input {
        return Err(AudioError::InvalidTokenShape {
            expected_tokens: expected_input,
            actual_tokens: input.len(),
        });
    }

    let output_len = checked_product(&[batch_size, num_groups, seq_len])?;
    let mut tokens = vec![0_i32; output_len];

    for batch in 0..batch_size {
        let length = lengths[batch] as usize;
        for group in 0..num_groups {
            for time in 0..seq_len {
                let output_index = (batch * num_groups + group) * seq_len + time;
                if time >= length {
                    tokens[output_index] = 0;
                    continue;
                }

                let mut token = 0_i32;
                for dim in 0..codebook_dim {
                    let levels = num_levels[dim];
                    let scale_i = levels / 2;

                    let output_scale = (levels - 1) as f32 * 0.5 * (1.0 - eps);
                    let output_offset = if levels % 2 == 0 {
                        0.5
                    } else {
                        0.0
                    };
                    let input_shift = (output_offset / output_scale).tan();

                    let input_index =
                        (batch * (num_groups * codebook_dim) + group * codebook_dim + dim) * seq_len + time;
                    let compressed = output_scale * (input[input_index] + input_shift).tanh() - output_offset;
                    let rounded = round_ties_to_even(compressed);
                    let code_nonnegative = (rounded as i32 + scale_i).clamp(0, levels - 1);
                    token += code_nonnegative * dim_base_index[dim];
                }

                tokens[output_index] = token;
            }
        }
    }

    Ok(tokens)
}
