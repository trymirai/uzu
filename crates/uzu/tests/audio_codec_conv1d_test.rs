#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};

use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::{Conv1dArguments, Conv1dKernel},
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn audio_conv1d_replicate_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 2usize;
    let cin = 3usize;
    let cout = 4usize;
    let seq_len_in = 9usize;

    let stride = 3usize;
    let kernel_size = 2 * stride;
    let dilation = 1usize;
    let padding = 2usize; // get_down_sample_padding(k=6, stride=3) = 2
    let pad_mode = 1usize; // replicate

    let seq_len_out =
        ((seq_len_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;

    let lengths_out: [i32; 2] = [2, seq_len_out as i32];

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cout * cin * kernel_size;
    let bias_len = cout;
    let output_len = batch_size * cout * seq_len_out;

    let input_buf = device.new_buffer(
        (input_len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let weight_buf = device.new_buffer(
        (weight_len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bias_buf = device.new_buffer(
        (bias_len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lengths_buf = device.new_buffer(
        (batch_size * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (output_len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..input_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();
    let weight: Vec<f32> = (0..weight_len)
        .map(|i| ((i as f32) * 0.07 - 1.3).cos() * 0.2)
        .collect();
    let bias: Vec<f32> = (0..bias_len).map(|i| i as f32 * 0.01 - 0.02).collect();

    unsafe {
        let in_ptr = input_buf.contents() as *mut f32;
        for (i, &v) in input.iter().enumerate() {
            in_ptr.add(i).write(v);
        }
        let w_ptr = weight_buf.contents() as *mut f32;
        for (i, &v) in weight.iter().enumerate() {
            w_ptr.add(i).write(v);
        }
        let b_ptr = bias_buf.contents() as *mut f32;
        for (i, &v) in bias.iter().enumerate() {
            b_ptr.add(i).write(v);
        }
        let l_ptr = lengths_buf.contents() as *mut i32;
        l_ptr.add(0).write(lengths_out[0]);
        l_ptr.add(1).write(lengths_out[1]);
    }

    // Reference
    let mut expected = vec![0.0f32; output_len];
    for b in 0..batch_size {
        for oc in 0..cout {
            for t_out in 0..seq_len_out {
                let out_idx = (b * cout + oc) * seq_len_out + t_out;
                if (t_out as i32) >= lengths_out[b] {
                    expected[out_idx] = 0.0;
                    continue;
                }
                let mut acc = bias[oc];
                let base = (t_out * stride) as isize - padding as isize;
                for ic in 0..cin {
                    let x_base = (b * cin + ic) * seq_len_in;
                    let w_base = (oc * cin + ic) * kernel_size;
                    for k in 0..kernel_size {
                        let mut x_t = base + (k * dilation) as isize;
                        if x_t < 0 {
                            x_t = 0;
                        } else if x_t >= seq_len_in as isize {
                            x_t = seq_len_in as isize - 1;
                        }
                        let x_t = x_t as usize;
                        let x_idx = x_base + x_t;
                        let w_idx = w_base + k;
                        acc += weight[w_idx] * input[x_idx];
                    }
                }
                expected[out_idx] = acc;
            }
        }
    }

    // Run kernel
    let kernel =
        Conv1dKernel::new(&context, KernelDataType::Float32).expect("kernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(
            &encoder,
            Conv1dArguments {
                input: &input_buf,
                weight: &weight_buf,
                bias: &bias_buf,
                output: &output_buf,
                lengths: &lengths_buf,
                batch_size,
                cin,
                cout,
                seq_len_in,
                seq_len_out,
                kernel_size,
                stride,
                dilation,
                padding,
                pad_mode,
            },
        )
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let out_ptr = output_buf.contents() as *const f32;
        let got = std::slice::from_raw_parts(out_ptr, output_len);
        for i in 0..output_len {
            let e = expected[i];
            let g = got[i];
            let diff = (e - g).abs();
            assert!(
                diff <= 1e-4,
                "Mismatch at i={i}: expected {e}, got {g}, diff={diff}"
            );
        }
    }
}

