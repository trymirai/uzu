#![cfg(all(feature = "audio-runtime", target_os = "macos"))]

use metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use uzu::{
    DataType,
    array::ArrayContextExt,
    backends::{
        common::{
            Context,
            kernel::AudioConv1dKernel,
        },
        metal::{MetalContext, kernel::dsl::AudioConv1dMetalKernel},
    },
};

fn create_test_context() -> std::rc::Rc<MetalContext> {
    <MetalContext as Context>::new().expect("MetalContext")
}

#[test]
fn audio_conv1d_replicate_matches_reference_f32() {
    let context = create_test_context();
    let kernel = AudioConv1dMetalKernel::new(&context, DataType::F32).expect("audio runtime");

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

    let mut expected = vec![0.0f32; output_len];
    for batch in 0..batch_size {
        for out_channel in 0..cout {
            for out_time in 0..seq_len_out {
                let out_index = (batch * cout + out_channel) * seq_len_out + out_time;
                if (out_time as i32) >= lengths_out[batch] {
                    expected[out_index] = 0.0;
                    continue;
                }

                let mut acc = bias_values[out_channel];
                let base = (out_time * stride) as isize - padding as isize;

                for in_channel in 0..cin {
                    let x_base = (batch * cin + in_channel) * seq_len_in;
                    let w_base = (out_channel * cin + in_channel) * kernel_size;
                    for kernel_offset in 0..kernel_size {
                        let mut x_time = base + (kernel_offset * dilation) as isize;
                        if x_time < 0 {
                            x_time = 0;
                        } else if x_time >= seq_len_in as isize {
                            x_time = seq_len_in as isize - 1;
                        }

                        let x_index = x_base + x_time as usize;
                        let w_index = w_base + kernel_offset;
                        acc += weight_values[w_index] * input_values[x_index];
                    }
                }

                expected[out_index] = acc;
            }
        }
    }

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
