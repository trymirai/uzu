#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};

use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::{FsqEncodeArguments, FsqEncodeKernel},
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

fn round_ties_to_even(x: f32) -> f32 {
    let f = x.floor();
    let frac = x - f;
    if frac < 0.5 {
        f
    } else if frac > 0.5 {
        f + 1.0
    } else {
        let fi = f as i64; // f is integral
        if (fi & 1) != 0 {
            f + 1.0
        } else {
            f
        }
    }
}

#[test]
fn fsq_encode_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 2usize;
    let num_groups = 13usize;
    let codebook_dim = 4usize;
    let seq_len = 7usize;
    let eps = 1e-3f32;

    let num_levels: [i32; 4] = [8, 7, 6, 6];
    let dim_base_index: [i32; 4] = [1, 8, 56, 336];

    let lengths: [i32; 2] = [4, 7];

    let input_len = batch_size * num_groups * codebook_dim * seq_len;
    let output_len = batch_size * num_groups * seq_len;

    let input_buf = device.new_buffer(
        (input_len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (output_len * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lengths_buf = device.new_buffer(
        (batch_size * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..input_len)
        .map(|i| ((i as f32) * 0.01 - 1.7).sin() * 0.9)
        .collect();

    unsafe {
        let in_ptr = input_buf.contents() as *mut f32;
        for (i, &v) in input.iter().enumerate() {
            in_ptr.add(i).write(v);
        }
        let l_ptr = lengths_buf.contents() as *mut i32;
        l_ptr.add(0).write(lengths[0]);
        l_ptr.add(1).write(lengths[1]);
    }

    // Reference
    let mut expected = vec![0i32; output_len];
    for b in 0..batch_size {
        for g in 0..num_groups {
            for t in 0..seq_len {
                let out_idx = (b * num_groups + g) * seq_len + t;
                if (t as i32) >= lengths[b] {
                    expected[out_idx] = 0;
                    continue;
                }
                let mut token = 0i32;
                for d in 0..codebook_dim {
                    let levels = num_levels[d] as i32;
                    let scale_i = levels / 2;
                    let output_scale =
                        ((levels - 1) as f32) * 0.5 * (1.0 - eps);
                    let output_offset = if levels % 2 == 0 { 0.5 } else { 0.0 };
                    let input_shift = (output_offset / output_scale).tan();

                    let x_idx =
                        (b * (num_groups * codebook_dim) + g * codebook_dim + d) * seq_len
                            + t;
                    let x = input[x_idx];
                    let compressed =
                        output_scale * (x + input_shift).tanh() - output_offset;
                    let rounded = round_ties_to_even(compressed);
                    let mut code_nonneg = rounded as i32 + scale_i;
                    if code_nonneg < 0 {
                        code_nonneg = 0;
                    } else if code_nonneg > levels - 1 {
                        code_nonneg = levels - 1;
                    }
                    token += code_nonneg * dim_base_index[d];
                }
                expected[out_idx] = token;
            }
        }
    }

    // Run kernel
    let kernel =
        FsqEncodeKernel::new(&context, KernelDataType::Float32).expect("kernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(
            &encoder,
            FsqEncodeArguments {
                input: &input_buf,
                tokens: &output_buf,
                lengths: &lengths_buf,
                batch_size,
                num_groups,
                seq_len,
                codebook_dim_per_group: codebook_dim,
                num_levels_per_group: Box::new(num_levels),
                dim_base_index: Box::new(dim_base_index),
                eps,
            },
        )
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let out_ptr = output_buf.contents() as *const i32;
        let got = std::slice::from_raw_parts(out_ptr, output_len);
        for i in 0..output_len {
            assert_eq!(got[i], expected[i], "Mismatch at i={i}");
        }
    }
}

