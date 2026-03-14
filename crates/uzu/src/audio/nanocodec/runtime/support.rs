#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodedPaddedAudio {
    pub samples: Vec<f32>,
    pub channels: usize,
    pub frames: usize,
    pub lengths: Vec<usize>,
}

fn checked_product(values: &[usize]) -> AudioResult<usize> {
    values
        .iter()
        .try_fold(1usize, |acc, &value| acc.checked_mul(value))
        .ok_or(AudioError::Runtime("dimension product overflow".to_string()))
}

fn usize_to_i32(
    value: usize,
    name: &str,
) -> AudioResult<i32> {
    i32::try_from(value).map_err(|_| AudioError::Runtime(format!("{name} exceeds i32 range")))
}

fn convert_lengths_to_i32(
    lengths: &[usize],
    frames: usize,
) -> AudioResult<Vec<i32>> {
    let mut out = Vec::with_capacity(lengths.len());
    for &length in lengths {
        if length > frames {
            return Err(AudioError::InvalidTokenLengthValue {
                length,
                frames,
            });
        }
        out.push(usize_to_i32(length, "length")?);
    }
    Ok(out)
}

fn checked_mul_i32(
    value: i32,
    mul: usize,
) -> AudioResult<i32> {
    i32::try_from(mul)
        .map_err(|_| AudioError::Runtime("length scaling factor exceeds i32 range".to_string()))?
        .checked_mul(value)
        .ok_or(AudioError::Runtime("scaled length overflow".to_string()))
}

fn checked_div_ceil(
    numerator: usize,
    denominator: usize,
) -> AudioResult<usize> {
    if denominator == 0 {
        return Err(AudioError::Runtime("division by zero".to_string()));
    }
    let addend = denominator.saturating_sub(1);
    numerator
        .checked_add(addend)
        .ok_or(AudioError::Runtime("ceil-division overflow".to_string()))
        .map(|value| value / denominator)
}

fn scale_lengths_i32_in_place(
    source: &[i32],
    destination: &mut [i32],
    factor: usize,
) -> AudioResult<()> {
    if destination.len() != source.len() {
        return Err(AudioError::Runtime(format!(
            "scaled length buffer mismatch: expected {}, got {}",
            source.len(),
            destination.len()
        )));
    }
    for (dst, &src) in destination.iter_mut().zip(source.iter()) {
        *dst = checked_mul_i32(src, factor)?;
    }
    Ok(())
}
