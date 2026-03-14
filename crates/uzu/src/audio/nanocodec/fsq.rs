use crate::audio::{AudioError, AudioResult};

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
