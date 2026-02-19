#![cfg(all(feature = "audio-runtime", target_os = "macos"))]

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Context,
            kernel::audio::{
                AudioAddArguments, AudioClampArguments, AudioElementwiseArguments, AudioHalfSnakeArguments,
                AudioKernelRuntime, AudioScaleArguments,
            },
        },
        metal::{Metal, MetalContext, kernel::MetalAudioKernelRuntime},
    },
};

fn create_test_context() -> std::rc::Rc<MetalContext> {
    <MetalContext as Context>::new().expect("MetalContext")
}

#[test]
fn audio_leaky_relu_matches_reference_f32() {
    let context = create_test_context();
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

    let n = 1024usize;
    let slope = 0.01f32;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.12).collect();

    let mut input = context.create_array(&[n], DataType::F32, "audio_leaky_relu_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let output = context.create_array(&[n], DataType::F32, "audio_leaky_relu_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    runtime
        .encode_leaky_relu(
            &encoder,
            AudioElementwiseArguments::<Metal> {
                input: input.buffer(),
                output: output.buffer(),
                n,
            },
            slope,
        )
        .expect("encode leaky relu");

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
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

    let n = 1024usize;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 5.12).collect();

    let mut input = context.create_array(&[n], DataType::F32, "audio_tanh_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let output = context.create_array(&[n], DataType::F32, "audio_tanh_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    runtime
        .encode_tanh(
            &encoder,
            AudioElementwiseArguments::<Metal> {
                input: input.buffer(),
                output: output.buffer(),
                n,
            },
        )
        .expect("encode tanh");

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
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

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

    runtime
        .encode_add(
            &encoder,
            AudioAddArguments::<Metal> {
                a: a.buffer(),
                b: b.buffer(),
                output: sum.buffer(),
                n,
            },
        )
        .expect("encode add");

    runtime
        .encode_scale(
            &encoder,
            AudioScaleArguments::<Metal> {
                input: sum.buffer(),
                output: scaled.buffer(),
                n,
                scale: scale_value,
            },
        )
        .expect("encode scale");

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
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

    let n = 2048usize;
    let min_value = -1.0f32;
    let max_value = 1.0f32;
    let input_values: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 10.24).collect();

    let mut input = context.create_array(&[n], DataType::F32, "audio_clamp_input");
    input.as_slice_mut::<f32>().copy_from_slice(&input_values);

    let output = context.create_array(&[n], DataType::F32, "audio_clamp_output");

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    runtime
        .encode_clamp(
            &encoder,
            AudioClampArguments::<Metal> {
                input: input.buffer(),
                output: output.buffer(),
                n,
                min_value,
                max_value,
            },
        )
        .expect("encode clamp");

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
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

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

    runtime
        .encode_half_snake(
            &encoder,
            AudioHalfSnakeArguments::<Metal> {
                input: input.buffer(),
                alpha: alpha.buffer(),
                output: output.buffer(),
                batch_size,
                channels,
                seq_len,
                snake_channels,
                negative_slope,
                eps,
            },
        )
        .expect("encode half snake");

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let got = output.as_slice::<f32>();
    for batch in 0..batch_size {
        for channel in 0..channels {
            for time in 0..seq_len {
                let index = (batch * channels + channel) * seq_len + time;
                let x = input_values[index];
                let expected = if channel < snake_channels {
                    let alpha = alpha_values[channel];
                    let sine = (alpha * x).sin();
                    x + (sine * sine) / (alpha + eps)
                } else if x >= 0.0 {
                    x
                } else {
                    negative_slope * x
                };

                let delta = (expected - got[index]).abs();
                assert!(
                    delta <= 1e-5,
                    "Mismatch at index={index}: expected {expected}, got {}, delta={delta}",
                    got[index]
                );
            }
        }
    }
}
