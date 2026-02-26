#![cfg(all(feature = "audio-runtime", target_os = "macos"))]

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    audio::nanocodec::{
        fsq::{fsq_decode_reference, fsq_encode_reference},
        ops::{
            CausalConv1dSpec, CausalConvTranspose1dSpec, Conv1dSpec, HalfSnakeSpec, PadMode,
            causal_conv_transpose1d_reference, causal_conv1d_reference, conv1d_reference, half_snake_reference,
        },
    },
    backends::{
        common::{
            Backend, Context, Kernels,
            kernel::{
                AudioAddKernel, AudioCausalConv1dKernel, AudioCausalConvTranspose1dKernel, AudioClampKernel,
                AudioConv1dKernel, AudioFsqDecodeKernel, AudioFsqEncodeKernel, AudioHalfSnakeKernel,
                AudioLeakyReluKernel, AudioScaleKernel, AudioTanhKernel,
            },
        },
        metal::Metal,
    },
};

fn create_test_context() -> std::rc::Rc<<Metal as Backend>::Context> {
    <Metal as Backend>::Context::new().expect("MetalContext")
}

#[test]
fn audio_conv1d_replicate_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioConv1dKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 2usize;
    let cin = 3usize;
    let cout = 4usize;
    let seq_len_in = 9usize;

    let stride = 3usize;
    let kernel_size = 2 * stride;
    let dilation = 1usize;
    let padding = 2usize;

    let seq_len_out = ((seq_len_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    let lengths_out: [i32; 2] = [2, seq_len_out as i32];

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cout * cin * kernel_size;
    let bias_len = cout;
    let output_len = batch_size * cout * seq_len_out;

    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.07 - 1.3).cos() * 0.2).collect();
    let bias_values: Vec<f32> = (0..bias_len).map(|i| i as f32 * 0.01 - 0.02).collect();

    let mut input = context.create_array(&[batch_size, cin, seq_len_in], DataType::F32, "audio_conv1d_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut weight = context.create_array(&[cout, cin, kernel_size], DataType::F32, "audio_conv1d_weight");
    weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

    let mut bias = context.create_array(&[cout], DataType::F32, "audio_conv1d_bias");
    bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

    let mut lengths = context.create_array(&[batch_size], DataType::I32, "audio_conv1d_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_out);

    let output = context.create_array(&[batch_size, cout, seq_len_out], DataType::F32, "audio_conv1d_output");
    let expected = conv1d_reference(Conv1dSpec {
        input: &input_values,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths_out,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        kernel_size,
        stride,
        dilation,
        padding,
        pad_mode: PadMode::Replicate,
    })
    .expect("reference conv1d");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(
        input.buffer(),
        weight.buffer(),
        bias.buffer(),
        output.buffer(),
        lengths.buffer(),
        cin as i32,
        cout as i32,
        seq_len_in as i32,
        seq_len_out as i32,
        kernel_size as i32,
        stride as i32,
        dilation as i32,
        padding as i32,
        1_i32,
        batch_size as i32,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for index in 0..output_len {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-4,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[test]
fn audio_causal_conv1d_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioCausalConv1dKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 2usize;
    let cin = 3usize;
    let cout = 4usize;
    let seq_len = 8usize;
    let kernel_size = 5usize;
    let dilation = 2usize;

    let lengths: [i32; 2] = [5, 8];

    let input_len = batch_size * cin * seq_len;
    let weight_len = cout * cin * kernel_size;
    let bias_len = cout;
    let output_len = batch_size * cout * seq_len;

    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.01).sin()).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.02 - 0.5).cos() * 0.1).collect();
    let bias_values: Vec<f32> = (0..bias_len).map(|i| i as f32 * 0.001 - 0.002).collect();

    let mut input = context.create_array(&[batch_size, cin, seq_len], DataType::F32, "audio_causal_conv1d_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut weight = context.create_array(&[cout, cin, kernel_size], DataType::F32, "audio_causal_conv1d_weight");
    weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

    let mut bias = context.create_array(&[cout], DataType::F32, "audio_causal_conv1d_bias");
    bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

    let mut lengths_array = context.create_array(&[batch_size], DataType::I32, "audio_causal_conv1d_lengths");
    lengths_array.as_slice_mut::<i32>().copy_from_slice(&lengths);

    let output = context.create_array(&[batch_size, cout, seq_len], DataType::F32, "audio_causal_conv1d_output");
    let expected = causal_conv1d_reference(CausalConv1dSpec {
        input: &input_values,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths,
        batch_size,
        cin,
        cout,
        seq_len,
        kernel_size,
        dilation,
    })
    .expect("reference causal conv1d");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(
        input.buffer(),
        weight.buffer(),
        bias.buffer(),
        output.buffer(),
        lengths_array.buffer(),
        cin as i32,
        cout as i32,
        seq_len as i32,
        kernel_size as i32,
        dilation as i32,
        batch_size as i32,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for index in 0..output_len {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-4,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[test]
fn audio_causal_conv_transpose1d_matches_reference_f32() {
    let context = create_test_context();
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::AudioCausalConvTranspose1dKernel::new(&context, DataType::F32)
            .expect("audio runtime");

    let batch_size = 2usize;
    let cin = 4usize;
    let groups = 2usize;
    let cout = 2usize;
    let stride = 3usize;
    let kernel_size = 2 * stride;

    let seq_len_in = 5usize;
    let seq_len_out = seq_len_in * stride;

    let lengths_out: [i32; 2] = [9, 15];

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cin * (cout / groups) * kernel_size;
    let output_len = batch_size * cout * seq_len_out;

    let input_values: Vec<f32> = (0..input_len).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let weight_values: Vec<f32> = (0..weight_len).map(|i| ((i as f32) * 0.07).cos() * 0.2).collect();
    let bias_values: Vec<f32> = vec![0.01, -0.02];

    let mut input =
        context.create_array(&[batch_size, cin, seq_len_in], DataType::F32, "audio_causal_conv_transpose_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut weight =
        context.create_array(&[cin, cout / groups, kernel_size], DataType::F32, "audio_causal_conv_transpose_weight");
    weight.as_slice_mut::<f32>().copy_from_slice(&weight_values);

    let mut bias = context.create_array(&[cout], DataType::F32, "audio_causal_conv_transpose_bias");
    bias.as_slice_mut::<f32>().copy_from_slice(&bias_values);

    let mut lengths = context.create_array(&[batch_size], DataType::I32, "audio_causal_conv_transpose_lengths");
    lengths.as_slice_mut::<i32>().copy_from_slice(&lengths_out);

    let output =
        context.create_array(&[batch_size, cout, seq_len_out], DataType::F32, "audio_causal_conv_transpose_output");
    let expected = causal_conv_transpose1d_reference(CausalConvTranspose1dSpec {
        input: &input_values,
        weight: &weight_values,
        bias: &bias_values,
        lengths: &lengths_out,
        batch_size,
        cin,
        cout,
        seq_len_in,
        seq_len_out,
        stride,
        groups,
    })
    .expect("reference causal conv transpose1d");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(
        input.buffer(),
        weight.buffer(),
        bias.buffer(),
        output.buffer(),
        lengths.buffer(),
        cin as i32,
        cout as i32,
        seq_len_in as i32,
        seq_len_out as i32,
        stride as i32,
        groups as i32,
        batch_size as i32,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for index in 0..output_len {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-4,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[test]
fn audio_leaky_relu_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioLeakyReluKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let n = 1024usize;
    let slope = 0.01f32;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.12).collect();

    let mut input = context.create_array(&[n], DataType::F32, "audio_leaky_relu_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let output = context.create_array(&[n], DataType::F32, "audio_leaky_relu_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(input.buffer(), output.buffer(), n as i32, slope, &encoder);

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for index in 0..n {
        let expected = if input_values[index] >= 0.0 {
            input_values[index]
        } else {
            slope * input_values[index]
        };
        let delta = (expected - got[index]).abs();
        assert!(delta <= 1e-6, "Mismatch at index={index}: expected {expected}, got {}, delta={delta}", got[index]);
    }
}

#[test]
fn audio_tanh_matches_reference_f32() {
    let context = create_test_context();
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::AudioTanhKernel::new(&context, DataType::F32).expect("audio runtime");

    let n = 1024usize;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.12).collect();

    let mut input = context.create_array(&[n], DataType::F32, "audio_tanh_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let output = context.create_array(&[n], DataType::F32, "audio_tanh_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(input.buffer(), output.buffer(), n as i32, &encoder);

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for index in 0..n {
        let expected = input_values[index].tanh();
        let delta = (expected - got[index]).abs();
        assert!(delta <= 1e-6, "Mismatch at index={index}: expected {expected}, got {}, delta={delta}", got[index]);
    }
}

#[test]
fn audio_add_and_scale_match_reference_f32() {
    let context = create_test_context();
    let add_kernel = <<Metal as Backend>::Kernels as Kernels>::AudioAddKernel::new(&context, DataType::F32)
        .expect("audio add kernel");
    let scale_kernel = <<Metal as Backend>::Kernels as Kernels>::AudioScaleKernel::new(&context, DataType::F32)
        .expect("audio scale kernel");

    let n = 2048usize;
    let scale_value = 1.0f32 / 3.0f32;
    let a_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 - 1.0).collect();
    let b_values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).sin()).collect();

    let mut a = context.create_array(&[n], DataType::F32, "audio_add_a");
    a.as_slice_mut::<f32>().copy_from_slice(&a_values);

    let mut b = context.create_array(&[n], DataType::F32, "audio_add_b");
    b.as_slice_mut::<f32>().copy_from_slice(&b_values);

    let sum = context.create_array(&[n], DataType::F32, "audio_add_sum");
    let scaled = context.create_array(&[n], DataType::F32, "audio_add_scaled");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    add_kernel.encode(a.buffer(), b.buffer(), sum.buffer(), n as i32, &encoder);

    scale_kernel.encode(sum.buffer(), scaled.buffer(), n as i32, scale_value, &encoder);

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let sum_got = sum.as_slice::<f32>();
    let scaled_got = scaled.as_slice::<f32>();

    for index in 0..n {
        let expected_sum = a_values[index] + b_values[index];
        let expected_scaled = expected_sum * scale_value;

        let sum_delta = (expected_sum - sum_got[index]).abs();
        let scaled_delta = (expected_scaled - scaled_got[index]).abs();

        assert!(
            sum_delta <= 1e-6,
            "Add mismatch at index={index}: expected {expected_sum}, got {}, delta={sum_delta}",
            sum_got[index]
        );
        assert!(
            scaled_delta <= 1e-6,
            "Scale mismatch at index={index}: expected {expected_scaled}, got {}, delta={scaled_delta}",
            scaled_got[index]
        );
    }
}

#[test]
fn audio_clamp_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioClampKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let n = 2048usize;
    let min_value = -1.0f32;
    let max_value = 1.0f32;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 10.24).collect();

    let mut input = context.create_array(&[n], DataType::F32, "audio_clamp_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let output = context.create_array(&[n], DataType::F32, "audio_clamp_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(input.buffer(), output.buffer(), n as i32, min_value, max_value, &encoder);

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for index in 0..n {
        let expected = input_values[index].clamp(min_value, max_value);
        let delta = (expected - got[index]).abs();
        assert!(delta <= 1e-6, "Mismatch at index={index}: expected {expected}, got {}, delta={delta}", got[index]);
    }
}

#[test]
fn audio_half_snake_matches_reference_f32() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioHalfSnakeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

    let batch_size = 1usize;
    let channels = 6usize;
    let snake_channels = channels / 2;
    let seq_len = 64usize;
    let n = batch_size * channels * seq_len;

    let negative_slope = 0.01f32;
    let eps = 1e-9f32;

    let input_values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01 - 3.2).sin()).collect();
    let alpha_values: Vec<f32> = (0..snake_channels).map(|i| 0.5 + i as f32 * 0.1).collect();

    let mut input = context.create_array(&[batch_size, channels, seq_len], DataType::F32, "audio_half_snake_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let mut alpha = context.create_array(&[snake_channels], DataType::F32, "audio_half_snake_alpha");
    alpha.as_slice_mut::<f32>().copy_from_slice(&alpha_values);

    let output = context.create_array(&[batch_size, channels, seq_len], DataType::F32, "audio_half_snake_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    kernel.encode(
        input.buffer(),
        alpha.buffer(),
        output.buffer(),
        channels as i32,
        seq_len as i32,
        snake_channels as i32,
        negative_slope,
        eps,
        batch_size as i32,
        &encoder,
    );

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    let expected = half_snake_reference(HalfSnakeSpec {
        input: &input_values,
        alpha: &alpha_values,
        batch_size,
        channels,
        seq_len,
        snake_channels,
        negative_slope,
        eps,
    })
    .expect("reference half snake");
    for index in 0..n {
        let delta = (expected[index] - got[index]).abs();
        assert!(
            delta <= 1e-5,
            "Mismatch at index={index}: expected {}, got {}, delta={delta}",
            expected[index],
            got[index]
        );
    }
}

#[test]
fn audio_fsq_decode_matches_reference() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqDecodeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

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
        fsq_decode_reference(&[0, 7, 11, 3, 5, 9], &[2], batch_size, num_groups, seq_len, codebook_dim, &num_levels)
            .expect("reference decode");
    let actual = output.as_slice::<f32>();

    for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (a - e).abs();
        assert!(delta <= 1e-5, "index={index}, actual={a}, expected={e}, delta={delta}");
    }
}

#[test]
fn audio_fsq_encode_matches_reference() {
    let context = create_test_context();
    let kernel = <<Metal as Backend>::Kernels as Kernels>::AudioFsqEncodeKernel::new(&context, DataType::F32)
        .expect("audio runtime");

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

    let expected = fsq_encode_reference(
        &input_values,
        &lengths_values,
        batch_size,
        num_groups,
        seq_len,
        codebook_dim,
        &num_levels,
        &dim_base_index,
        eps,
    )
    .expect("reference encode");
    let actual = tokens.as_slice::<i32>();

    assert_eq!(actual, expected);
}
