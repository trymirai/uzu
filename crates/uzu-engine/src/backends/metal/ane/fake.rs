//! The CPU reference of the FFN program (see [`super::mil`]): the same contract over the same
//! surfaces, in fp32. The test oracle of the ANE program, and the fake for tests of its users.

use half::f16;

use super::{
    AneSurface,
    mil::{FfnShape, INT8_PEAK, INT8_UNIT, INTERMEDIATE_FLOOR, LinearShape},
};

/// A `[channels, tokens]` row-major f32 matrix.
struct Matrix {
    channels: usize,
    tokens: usize,
    values: Vec<f32>,
}

impl Matrix {
    fn zeros(
        channels: usize,
        tokens: usize,
    ) -> Self {
        Self {
            channels,
            tokens,
            values: vec![0.0; channels * tokens],
        }
    }

    fn row(
        &self,
        channel: usize,
    ) -> &[f32] {
        &self.values[channel * self.tokens..(channel + 1) * self.tokens]
    }

    fn row_mut(
        &mut self,
        channel: usize,
    ) -> &mut [f32] {
        &mut self.values[channel * self.tokens..(channel + 1) * self.tokens]
    }
}

fn surface<'a>(
    inputs: &[(&str, &'a AneSurface)],
    name: &str,
) -> &'a AneSurface {
    inputs
        .iter()
        .find(|(input, _)| *input == name)
        .map(|(_, surface)| *surface)
        .unwrap_or_else(|| panic!("no input {name}"))
}

/// Reads the leading `tokens` columns of a token-major surface, dequantized by 1/128 if int8.
fn read_tokens(
    surface: &AneSurface,
    tokens: usize,
) -> Matrix {
    let mut matrix = Matrix::zeros(surface.rows() as usize, tokens);
    let scale = if surface.element() == super::AneElement::Int8 {
        (1.0 / INT8_UNIT) as f32
    } else {
        1.0
    };
    for channel in 0..surface.rows() {
        let row = matrix.row_mut(channel as usize);
        for (token, value) in row.iter_mut().enumerate() {
            *value = surface.get(channel, token as u32) * scale;
        }
    }
    matrix
}

/// `scales[c] * sum_k weights[c][k] / 128 * values[k][t]` over the given weight segments, for the
/// leading `channels` rows of the weights.
fn project(
    segments: &[&AneSurface],
    scales: &AneSurface,
    values: &Matrix,
    channels: usize,
) -> Matrix {
    let tokens = values.tokens;
    let mut result = Matrix::zeros(channels, tokens);
    let mut weights = Vec::new();
    for channel in 0..channels {
        weights.clear();
        for segment in segments {
            for k in 0..segment.width() {
                weights.push(segment.get(channel as u32, k) / INT8_UNIT as f32);
            }
        }
        let scale = scales.get(channel as u32, 0);
        let row = result.row_mut(channel);
        for (k, weight) in weights.iter().enumerate() {
            if *weight == 0.0 {
                continue;
            }
            let input = values.row(k);
            for token in 0..tokens {
                row[token] += weight * input[token];
            }
        }
        for value in row.iter_mut() {
            *value *= scale;
        }
    }
    result
}

/// `out[c] = sum_i rotation[c][i] * values[block(c) + i]`.
fn block_rotation(
    values: &Matrix,
    rotation: &[f16],
    block: usize,
) -> Matrix {
    let mut result = Matrix::zeros(values.channels, values.tokens);
    for channel in 0..values.channels {
        let start = channel / block * block;
        let row = result.row_mut(channel);
        for i in 0..block {
            let weight = rotation[channel * block + i].to_f32();
            if weight == 0.0 {
                continue;
            }
            let input = values.row(start + i);
            for token in 0..values.tokens {
                row[token] += weight * input[token];
            }
        }
    }
    result
}

/// Runs function `ffn{rows}` of the program on the CPU: reads the inputs (by name) and writes the
/// leading `rows` columns of `output`, as the ANE does.
pub fn run_ffn_reference(
    shape: &FfnShape,
    rows: u32,
    inputs: &[(&str, &AneSurface)],
    output: &mut AneSurface,
) {
    let tokens = rows as usize;
    let x = if shape.single_input_surface {
        read_tokens(surface(inputs, "x"), tokens)
    } else {
        let mut x = Matrix::zeros(shape.hidden as usize, tokens);
        for k in 0..shape.input_segments() {
            let segment = read_tokens(surface(inputs, &format!("x{k}")), tokens);
            for channel in 0..shape.segment as usize {
                x.row_mut(k as usize * shape.segment as usize + channel).copy_from_slice(segment.row(channel));
            }
        }
        x
    };
    let token_scales = read_tokens(surface(inputs, "tx"), tokens);

    let segments = |prefix: &str, count: u32| -> Vec<&AneSurface> {
        if shape.single_weight_surfaces {
            vec![surface(inputs, prefix)]
        } else {
            (0..count).map(|k| surface(inputs, &format!("{prefix}{k}"))).collect()
        }
    };
    let mut gate =
        project(&segments("wg", shape.input_segments()), surface(inputs, "sg"), &x, shape.intermediate as usize);
    let mut value =
        project(&segments("wu", shape.input_segments()), surface(inputs, "su"), &x, shape.intermediate as usize);
    if shape.up_rotation_block > 0 {
        gate = block_rotation(&gate, &shape.gate_rotation, shape.up_rotation_block as usize);
        value = block_rotation(&value, &shape.value_rotation, shape.up_rotation_block as usize);
    }

    let mut hidden = Matrix::zeros(shape.intermediate as usize, tokens);
    for channel in 0..shape.intermediate as usize {
        let gate = gate.row(channel);
        let value = value.row(channel);
        let row = hidden.row_mut(channel);
        for token in 0..tokens {
            let g = gate[token] * token_scales.row(0)[token];
            row[token] = g / (1.0 + (-g).exp()) * value[token];
        }
    }
    let rotated = block_rotation(&hidden, &shape.rotation, shape.rotation_block as usize);

    let mut quantized = Matrix::zeros(rotated.channels, tokens);
    let mut intermediate_scales = vec![0.0f32; tokens];
    for token in 0..tokens {
        let peak = (0..rotated.channels).map(|channel| rotated.row(channel)[token].abs()).fold(0.0f32, f32::max);
        let floor = peak.max(INTERMEDIATE_FLOOR as f32);
        for channel in 0..rotated.channels {
            let scaled = rotated.row(channel)[token] * INT8_PEAK as f32 / floor;
            let code = if shape.half_intermediate {
                scaled
            } else {
                scaled.round().clamp(-127.0, 127.0)
            };
            quantized.row_mut(channel)[token] = code / INT8_UNIT as f32;
        }
        intermediate_scales[token] = floor * (INT8_UNIT / INT8_PEAK) as f32;
    }
    let down =
        project(&segments("wd", shape.down_segments()), surface(inputs, "sd"), &quantized, shape.hidden as usize);

    let stride = output.row_stride_elements();
    let values = output.as_f16_mut();
    for channel in 0..shape.hidden as usize {
        for token in 0..tokens {
            values[channel * stride + token] = f16::from_f32(down.row(channel)[token]);
        }
    }
    for token in 0..tokens {
        values[shape.hidden as usize * stride + token] = f16::from_f32(intermediate_scales[token]);
    }
}

/// Runs function `lin{rows}` of the linear program on the CPU: reads the inputs (by name) and
/// writes the leading `output` rows and `rows` columns of `output_surface`.
pub fn run_linear_reference(
    shape: &LinearShape,
    rows: u32,
    inputs: &[(&str, &AneSurface)],
    output_surface: &mut AneSurface,
) {
    let tokens = rows as usize;
    let x = read_tokens(surface(inputs, "x"), tokens);
    let result = project(&[surface(inputs, "w")], surface(inputs, "s"), &x, shape.output as usize);
    let stride = output_surface.row_stride_elements();
    let values = output_surface.as_f16_mut();
    for channel in 0..shape.output as usize {
        for token in 0..tokens {
            values[channel * stride + token] = f16::from_f32(result.row(channel)[token]);
        }
    }
}
