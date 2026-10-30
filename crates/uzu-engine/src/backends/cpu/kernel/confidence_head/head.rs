use half::bf16;

pub const SLOT_NUMBERS: usize = 4;
pub const NONE: u32 = u32::MAX;
pub const LOG_PROB_FLOOR: f32 = -30.0;

pub fn tensor<'a>(
    values: *const f32,
    len: usize,
) -> &'a [f32] {
    unsafe { std::slice::from_raw_parts(values, len) }
}

pub fn weights(
    values: *const bf16,
    len: usize,
) -> Vec<f32> {
    (0..len).map(|index| unsafe { (*values.add(index)).to_f32() }).collect()
}

pub fn dense(
    weights: &[f32],
    biases: &[f32],
    input: &[f32],
) -> Vec<f32> {
    biases
        .iter()
        .enumerate()
        .map(|(unit, bias)| {
            bias + weights[unit * input.len()..(unit + 1) * input.len()]
                .iter()
                .zip(input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
        })
        .collect()
}

pub fn layer_norm(
    input: &[f32],
    scales: &[f32],
    biases: &[f32],
    epsilon: f32,
) -> Vec<f32> {
    let count = input.len() as f32;
    let mean = input.iter().sum::<f32>() / count;
    let variance = input.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / count;
    let scale = (variance + epsilon).sqrt().recip();
    input.iter().zip(scales).zip(biases).map(|((value, gain), bias)| (value - mean) * scale * gain + bias).collect()
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub fn log_sum(values: &[f32]) -> f32 {
    let top = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    top + values.iter().map(|value| (value - top).exp()).sum::<f32>().ln()
}
