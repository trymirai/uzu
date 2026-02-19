#![cfg(all(feature = "audio-runtime", target_os = "macos"))]

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Context,
            kernel::{AudioFsqDecodeKernel, AudioFsqEncodeKernel, AudioLeakyReluKernel},
        },
        metal::{
            MetalContext,
            kernel::dsl::{AudioFsqDecodeMetalKernel, AudioFsqEncodeMetalKernel, AudioLeakyReluMetalKernel},
        },
    },
};

fn create_test_context() -> std::rc::Rc<MetalContext> {
    <MetalContext as Context>::new().expect("MetalContext")
}

#[test]
fn audio_dsl_leaky_relu_matches_reference() {
    let context = create_test_context();
    let kernel = AudioLeakyReluMetalKernel::new(&context, DataType::F32).expect("audio runtime");

    let input_values: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 3.0];
    let n = input_values.len();

    let mut input = context.create_array(&[n], DataType::F32, "audio_leaky_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);
    let output = context.create_array(&[n], DataType::F32, "audio_leaky_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(input.buffer(), output.buffer(), n as i32, 0.1, &encoder);

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let expected: Vec<f32> = input_values
        .iter()
        .map(|&x| {
            if x >= 0.0 {
                x
            } else {
                x * 0.1
            }
        })
        .collect();
    let actual = output.as_slice::<f32>();

    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (a - e).abs();
        assert!(delta <= 1e-6, "index={index}, actual={a}, expected={e}, delta={delta}");
    }
}

#[test]
fn audio_dsl_fsq_decode_matches_reference() {
    let context = create_test_context();
    let kernel = AudioFsqDecodeMetalKernel::new(&context, DataType::F32).expect("audio runtime");

    let batch_size = 1usize;
    let num_groups = 2usize;
    let seq_len = 3usize;
    let codebook_dim = 2usize;
    let num_levels: Vec<i32> = vec![8, 5];

    let mut tokens = context.create_array(&[batch_size, num_groups, seq_len], DataType::I32, "audio_fsq_tokens");
    tokens.as_slice_mut::<i32>().copy_from_slice(&[
        0, 7, 11, // g0 tokens at t0..t2
        3, 5, 9, // g1 tokens at t0..t2
    ]);

    let mut lengths = context.create_array(&[batch_size], DataType::I32, "audio_fsq_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&[2]);

    let output =
        context.create_array(&[batch_size, num_groups * codebook_dim, seq_len], DataType::F32, "audio_fsq_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(
        tokens.buffer(),
        output.buffer(),
        lengths.buffer(),
        num_groups as i32,
        seq_len as i32,
        codebook_dim as i32,
        &num_levels,
        batch_size as i32,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let expected =
        reference_fsq_decode(&[0, 7, 11, 3, 5, 9], 2, batch_size, num_groups, seq_len, codebook_dim, &num_levels);
    let actual = output.as_slice::<f32>();

    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (a - e).abs();
        assert!(delta <= 1e-5, "index={index}, actual={a}, expected={e}, delta={delta}");
    }
}

#[test]
fn audio_dsl_fsq_encode_matches_reference() {
    let context = create_test_context();
    let kernel = AudioFsqEncodeMetalKernel::new(&context, DataType::F32).expect("audio runtime");

    let batch_size = 1usize;
    let num_groups = 2usize;
    let codebook_dim = 2usize;
    let seq_len = 4usize;
    let eps = 1e-3_f32;
    let num_levels: Vec<i32> = vec![8, 6];
    let dim_base_index: Vec<i32> = vec![1, 8];

    let input_values = vec![
        -0.9, -0.1, 0.2, 0.7, // g0 d0
        0.8, 0.3, -0.4, -0.7, // g0 d1
        -0.2, 0.0, 0.4, 0.9, // g1 d0
        0.5, -0.5, 0.1, -0.1, // g1 d1
    ];
    let lengths_values = vec![3_i32];

    let mut input = context.create_array(
        &[batch_size, num_groups * codebook_dim, seq_len],
        DataType::F32,
        "audio_fsq_encode_input",
    );
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut lengths = context.create_array(&[batch_size], DataType::I32, "audio_fsq_encode_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_values);

    let tokens = context.create_array(&[batch_size, num_groups, seq_len], DataType::I32, "audio_fsq_encode_tokens");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(
        input.buffer(),
        tokens.buffer(),
        lengths.buffer(),
        num_groups as i32,
        seq_len as i32,
        codebook_dim as i32,
        &num_levels,
        &dim_base_index,
        eps,
        batch_size as i32,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let expected = reference_fsq_encode(
        &input_values,
        &lengths_values,
        batch_size,
        num_groups,
        seq_len,
        codebook_dim,
        &num_levels,
        &dim_base_index,
        eps,
    );
    let actual = tokens.as_slice::<i32>();

    assert_eq!(actual, expected);
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

fn reference_fsq_encode(
    input: &[f32],
    lengths: &[i32],
    batch_size: usize,
    num_groups: usize,
    seq_len: usize,
    codebook_dim: usize,
    num_levels: &[i32],
    dim_base_index: &[i32],
    eps: f32,
) -> Vec<i32> {
    let mut tokens = vec![0_i32; batch_size * num_groups * seq_len];

    for batch in 0..batch_size {
        for group in 0..num_groups {
            for time in 0..seq_len {
                let output_index = (batch * num_groups + group) * seq_len + time;
                if (time as i32) >= lengths[batch] {
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

    tokens
}

fn reference_fsq_decode(
    tokens: &[i32],
    length: usize,
    batch_size: usize,
    num_groups: usize,
    seq_len: usize,
    codebook_dim: usize,
    num_levels: &[i32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch_size * num_groups * codebook_dim * seq_len];

    for b in 0..batch_size {
        for g in 0..num_groups {
            for t in 0..seq_len {
                if t >= length {
                    continue;
                }

                let token_index = (b * num_groups + g) * seq_len + t;
                let token = tokens[token_index];

                let mut base = 1i32;
                for d in 0..codebook_dim {
                    let levels = num_levels[d];
                    let scale = levels / 2;
                    let offset = scale;
                    let div = token / base;
                    let mut code_nonneg = div % levels;
                    if code_nonneg < 0 {
                        code_nonneg += levels;
                    }
                    let value = (code_nonneg - offset) as f32 / scale as f32;

                    let out_channel = g * codebook_dim + d;
                    let out_index = (b * (num_groups * codebook_dim) + out_channel) * seq_len + t;
                    out[out_index] = value;

                    base *= levels;
                }
            }
        }
    }

    out
}
