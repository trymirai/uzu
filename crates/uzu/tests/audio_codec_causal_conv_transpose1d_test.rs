#![cfg(all(feature = "audio-runtime", target_os = "macos"))]

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Context,
            kernel::audio::{AudioCausalConvTranspose1dArguments, AudioKernelRuntime},
        },
        metal::{Metal, MetalContext, kernel::MetalAudioKernelRuntime},
    },
};

fn create_test_context() -> std::rc::Rc<MetalContext> {
    <MetalContext as Context>::new().expect("MetalContext")
}

#[test]
fn audio_causal_conv_transpose1d_matches_reference_f32() {
    let context = create_test_context();
    let runtime = MetalAudioKernelRuntime::new(&context, DataType::F32).expect("audio runtime");

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

    let cout_per_group = cout / groups;
    let cin_per_group = cin / groups;
    let mut expected = vec![0.0f32; output_len];

    for batch in 0..batch_size {
        for out_channel in 0..cout {
            let group_index = out_channel / cout_per_group;
            let out_channel_in_group = out_channel % cout_per_group;
            let in_channel_begin = group_index * cin_per_group;
            let in_channel_end = in_channel_begin + cin_per_group;

            for out_time in 0..seq_len_out {
                let out_index = (batch * cout + out_channel) * seq_len_out + out_time;
                if (out_time as i32) >= lengths_out[batch] {
                    expected[out_index] = 0.0;
                    continue;
                }

                let q = out_time / stride;
                let r = out_time % stride;
                let mut acc = bias_values[out_channel];

                for in_channel in in_channel_begin..in_channel_end {
                    let input_base = (batch * cin + in_channel) * seq_len_in;
                    let weight_base = (in_channel * cout_per_group + out_channel_in_group) * kernel_size;

                    acc += input_values[input_base + q] * weight_values[weight_base + r];
                    if q > 0 {
                        acc += input_values[input_base + (q - 1)] * weight_values[weight_base + (stride + r)];
                    }
                }

                expected[out_index] = acc;
            }
        }
    }

    let command_buffer = context.command_queue.command_buffer().expect("command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("compute encoder");

    runtime
        .encode_causal_conv_transpose1d(
            &encoder,
            AudioCausalConvTranspose1dArguments::<Metal> {
                input: input.buffer(),
                weight: weight.buffer(),
                bias: bias.buffer(),
                output: output.buffer(),
                lengths: lengths.buffer(),
                batch_size,
                cin,
                cout,
                seq_len_in,
                seq_len_out,
                stride,
                groups,
            },
        )
        .expect("encode causal conv transpose1d");

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
