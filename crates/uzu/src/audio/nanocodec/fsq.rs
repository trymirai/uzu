use crate::audio::{AudioError, AudioResult};

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
}

pub fn round_ties_to_even(value: f32) -> f32 {
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

pub fn compute_dim_base_index(num_levels: &[i32]) -> AudioResult<Vec<i32>> {
    if num_levels.is_empty() {
        return Err(AudioError::InvalidTokenCardinality);
    }

    let mut out = vec![0_i32; num_levels.len()];
    let mut base = 1_i32;

    for (index, &levels) in num_levels.iter().enumerate() {
        if levels <= 1 {
            return Err(AudioError::InvalidTokenCardinality);
        }
        out[index] = base;
        base = base.checked_mul(levels).ok_or(AudioError::Runtime("dim_base_index overflow".to_string()))?;
    }

    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::{compute_dim_base_index, fsq_decode_reference, fsq_encode_reference, round_ties_to_even};

    #[test]
    fn dim_base_index_is_computed_correctly() {
        let result = compute_dim_base_index(&[8, 6, 5]).expect("dim_base_index");
        assert_eq!(result, vec![1, 8, 48]);
    }

    #[test]
    fn fsq_decode_reference_masks_values_beyond_lengths() {
        let output = fsq_decode_reference(&[0, 7, 11, 3, 5, 9], &[2], 1, 2, 3, 2, &[8, 5]).expect("decode");

        assert_eq!(output.len(), 1 * (2 * 2) * 3);
        let masked_t2 = [
            output[2],  // g0 d0 t2
            output[5],  // g0 d1 t2
            output[8],  // g1 d0 t2
            output[11], // g1 d1 t2
        ];
        assert_eq!(masked_t2, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn fsq_encode_reference_masks_values_beyond_lengths() {
        let tokens = fsq_encode_reference(
            &[
                -0.9, -0.1, 0.2, 0.7, // g0 d0
                0.8, 0.3, -0.4, -0.7, // g0 d1
                -0.2, 0.0, 0.4, 0.9, // g1 d0
                0.5, -0.5, 0.1, -0.1, // g1 d1
            ],
            &[3],
            1,
            2,
            4,
            2,
            &[8, 6],
            &[1, 8],
            1e-3,
        )
        .expect("encode");

        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[3], 0);
        assert_eq!(tokens[7], 0);
    }

    #[test]
    fn ties_are_rounded_to_even() {
        assert_eq!(round_ties_to_even(2.5), 2.0);
        assert_eq!(round_ties_to_even(3.5), 4.0);
        assert_eq!(round_ties_to_even(-1.5), -2.0);
        assert_eq!(round_ties_to_even(-2.5), -2.0);
    }
}
