#![cfg(all(feature = "audio-runtime", target_os = "macos"))]

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Context,
            kernel::audio::{AudioCausalConv1dArguments, AudioKernelRuntime},
        },
        metal::{Metal, MetalContext, kernel::MetalAudioKernelRuntime},
    },
};

fn create_test_context() -> std::rc::Rc<MetalContext> {
    <MetalContext as Context>::new().expect("MetalContext")
}

#[test]
fn audio_causal_conv1d_matches_reference_f32() {
    let context = create_test_context();
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

    let batch_size = 2usize;
    let cin = 3usize;
    let cout = 4usize;
    let seq_len = 8usize;
    let kernel_size = 5usize;
    let dilation = 2usize;
    let pad = (kernel_size - 1) * dilation;

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

    let mut expected = vec![0.0f32; output_len];
    for batch in 0..batch_size {
        for out_channel in 0..cout {
            for time in 0..seq_len {
                let out_index = (batch * cout + out_channel) * seq_len + time;
                if (time as i32) >= lengths[batch] {
                    expected[out_index] = 0.0;
                    continue;
                }

                let mut acc = bias_values[out_channel];
                for in_channel in 0..cin {
                    for kernel_offset in 0..kernel_size {
                        let x_time = time as isize + (kernel_offset * dilation) as isize - pad as isize;
                        if x_time < 0 || x_time >= seq_len as isize {
                            continue;
                        }

                        let x_index = (batch * cin + in_channel) * seq_len + x_time as usize;
                        let w_index = (out_channel * cin + in_channel) * kernel_size + kernel_offset;
                        acc += weight_values[w_index] * input_values[x_index];
                    }
                }

                expected[out_index] = acc;
            }
        }
    }

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    runtime
        .encode_causal_conv1d(
            &encoder,
            AudioCausalConv1dArguments::<Metal> {
                input: input.buffer(),
                weight: weight.buffer(),
                bias: bias.buffer(),
                output: output.buffer(),
                lengths: lengths_array.buffer(),
                batch_size,
                cin,
                cout,
                seq_len,
                kernel_size,
                dilation,
            },
        )
        .expect("encode causal conv1d");

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
