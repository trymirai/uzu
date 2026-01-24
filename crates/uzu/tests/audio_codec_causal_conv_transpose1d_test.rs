#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};

use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::{CausalConvTranspose1dArguments, CausalConvTranspose1dKernel},
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn audio_causal_conv_transpose1d_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 2usize;
    let cin = 4usize;
    let groups = 2usize;
    let cout = 2usize; // cout_per_group = 1
    let stride = 3usize;
    let kernel_size = 2 * stride;

    let seq_len_in = 5usize;
    let seq_len_out = seq_len_in * stride;

    let lengths_out: [i32; 2] = [9, 15]; // test masking on batch 0

    let input_len = batch_size * cin * seq_len_in;
    let weight_len = cin * (cout / groups) * kernel_size;
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
        .map(|i| (i as f32 * 0.013).sin() * 0.5)
        .collect();
    let weight: Vec<f32> = (0..weight_len)
        .map(|i| ((i as f32) * 0.07).cos() * 0.2)
        .collect();
    let bias: Vec<f32> = vec![0.01, -0.02];

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
    let cout_per_group = cout / groups;
    let cin_per_group = cin / groups;
    let mut expected = vec![0.0f32; output_len];
    for b in 0..batch_size {
        for oc in 0..cout {
            let group_idx = oc / cout_per_group;
            let oc_in_group = oc % cout_per_group;
            let ic_begin = group_idx * cin_per_group;
            let ic_end = ic_begin + cin_per_group;
            for t_out in 0..seq_len_out {
                let out_idx = (b * cout + oc) * seq_len_out + t_out;
                if (t_out as i32) >= lengths_out[b] {
                    expected[out_idx] = 0.0;
                    continue;
                }
                let q = t_out / stride;
                let r = t_out % stride;
                let mut acc = bias[oc];
                for ic in ic_begin..ic_end {
                    let in_base = (b * cin + ic) * seq_len_in;
                    let w_base = (ic * cout_per_group + oc_in_group) * kernel_size;
                    acc += input[in_base + q] * weight[w_base + r];
                    if q > 0 {
                        acc += input[in_base + (q - 1)] * weight[w_base + (stride + r)];
                    }
                }
                expected[out_idx] = acc;
            }
        }
    }

    // Run kernel
    let kernel = CausalConvTranspose1dKernel::new(&context, KernelDataType::Float32)
        .expect("kernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(
            &encoder,
            CausalConvTranspose1dArguments {
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
                stride,
                groups,
            },
        )
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Compare
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

