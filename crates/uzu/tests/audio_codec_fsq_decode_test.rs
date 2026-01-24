#![cfg(target_os = "macos")]

use half::{bf16, f16};
use metal::{Device, MTLResourceOptions};

use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::{FsqDecodeArguments, FsqDecodeKernel},
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

fn fsq_decode_reference(
    token: i32,
    num_levels: &[i32],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_levels.len());
    let mut base: i32 = 1;
    for &levels in num_levels {
        let scale = levels / 2;
        let offset = scale;
        let code_nonneg = (token / base).rem_euclid(levels);
        let code = (code_nonneg - offset) as f32 / scale as f32;
        out.push(code);
        base *= levels;
    }
    out
}

#[test]
fn fsq_decode_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 2usize;
    let num_groups = 3usize;
    let seq_len = 5usize;
    let codebook_dim_per_group = 4usize;
    let num_levels_per_group: [i32; 4] = [8, 7, 6, 6];

    let tokens_len = batch_size * num_groups * seq_len;
    let out_channels = num_groups * codebook_dim_per_group;
    let out_len = batch_size * out_channels * seq_len;

    let tokens_buf = device.new_buffer(
        (tokens_len * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lengths_buf = device.new_buffer(
        (batch_size * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (out_len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Fill tokens and lengths
    let lengths: [i32; 2] = [3, 5];
    unsafe {
        let lengths_ptr = lengths_buf.contents() as *mut i32;
        lengths_ptr.add(0).write(lengths[0]);
        lengths_ptr.add(1).write(lengths[1]);
    }

    let tokens: Vec<i32> = (0..tokens_len)
        .map(|i| ((i * 37 + 11) % 2016) as i32)
        .collect();
    unsafe {
        let tokens_ptr = tokens_buf.contents() as *mut i32;
        for (i, &v) in tokens.iter().enumerate() {
            tokens_ptr.add(i).write(v);
        }
    }

    // Run kernel
    let kernel = FsqDecodeKernel::new(&context, KernelDataType::Float32)
        .expect("FsqDecodeKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    kernel
        .encode(
            &encoder,
            FsqDecodeArguments {
                tokens: &tokens_buf,
                lengths: &lengths_buf,
                out: &out_buf,
                batch_size,
                num_groups,
                seq_len,
                codebook_dim_per_group,
                num_levels_per_group: num_levels_per_group.into(),
            },
        )
        .expect("encode");

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back and compare
    let mut expected = vec![0.0f32; out_len];
    for b in 0..batch_size {
        for g in 0..num_groups {
            for t in 0..seq_len {
                let token_idx = (b * num_groups + g) * seq_len + t;
                if (t as i32) >= lengths[b] {
                    continue;
                }
                let token = tokens[token_idx];
                let codes = fsq_decode_reference(token, &num_levels_per_group);
                for d in 0..codebook_dim_per_group {
                    let out_c = g * codebook_dim_per_group + d;
                    let out_idx = (b * out_channels + out_c) * seq_len + t;
                    expected[out_idx] = codes[d];
                }
            }
        }
    }

    unsafe {
        let out_ptr = out_buf.contents() as *const f32;
        let got = std::slice::from_raw_parts(out_ptr, out_len);
        for i in 0..out_len {
            let e = expected[i];
            let g = got[i];
            let diff = (e - g).abs();
            assert!(
                diff <= 1e-6,
                "Mismatch at idx={i}: expected {e}, got {g}, diff={diff}"
            );
        }
    }
}

#[test]
fn fsq_decode_matches_reference_f16() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 2usize;
    let num_groups = 3usize;
    let seq_len = 5usize;
    let codebook_dim_per_group = 4usize;
    let num_levels_per_group: [i32; 4] = [8, 7, 6, 6];

    let tokens_len = batch_size * num_groups * seq_len;
    let out_channels = num_groups * codebook_dim_per_group;
    let out_len = batch_size * out_channels * seq_len;

    let tokens_buf = device.new_buffer(
        (tokens_len * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lengths_buf = device.new_buffer(
        (batch_size * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (out_len * std::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Fill tokens and lengths
    let lengths: [i32; 2] = [3, 5];
    unsafe {
        let lengths_ptr = lengths_buf.contents() as *mut i32;
        lengths_ptr.add(0).write(lengths[0]);
        lengths_ptr.add(1).write(lengths[1]);
    }

    let tokens: Vec<i32> = (0..tokens_len)
        .map(|i| ((i * 37 + 11) % 2016) as i32)
        .collect();
    unsafe {
        let tokens_ptr = tokens_buf.contents() as *mut i32;
        for (i, &v) in tokens.iter().enumerate() {
            tokens_ptr.add(i).write(v);
        }
    }

    // Run kernel
    let kernel = FsqDecodeKernel::new(&context, KernelDataType::Float16)
        .expect("FsqDecodeKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    kernel
        .encode(
            &encoder,
            FsqDecodeArguments {
                tokens: &tokens_buf,
                lengths: &lengths_buf,
                out: &out_buf,
                batch_size,
                num_groups,
                seq_len,
                codebook_dim_per_group,
                num_levels_per_group: num_levels_per_group.into(),
            },
        )
        .expect("encode");

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back and compare (quantized to f16)
    let mut expected = vec![0.0f32; out_len];
    for b in 0..batch_size {
        for g in 0..num_groups {
            for t in 0..seq_len {
                let token_idx = (b * num_groups + g) * seq_len + t;
                if (t as i32) >= lengths[b] {
                    continue;
                }
                let token = tokens[token_idx];
                let codes = fsq_decode_reference(token, &num_levels_per_group);
                for d in 0..codebook_dim_per_group {
                    let out_c = g * codebook_dim_per_group + d;
                    let out_idx = (b * out_channels + out_c) * seq_len + t;
                    expected[out_idx] = f16::from_f32(codes[d]).to_f32();
                }
            }
        }
    }

    unsafe {
        let out_ptr = out_buf.contents() as *const f16;
        let got = std::slice::from_raw_parts(out_ptr, out_len);
        for i in 0..out_len {
            let e = expected[i];
            let g = got[i].to_f32();
            let diff = (e - g).abs();
            assert!(
                diff <= 1e-3,
                "Mismatch at idx={i}: expected {e}, got {g}, diff={diff}"
            );
        }
    }
}

#[test]
fn fsq_decode_matches_reference_bf16() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 2usize;
    let num_groups = 3usize;
    let seq_len = 5usize;
    let codebook_dim_per_group = 4usize;
    let num_levels_per_group: [i32; 4] = [8, 7, 6, 6];

    let tokens_len = batch_size * num_groups * seq_len;
    let out_channels = num_groups * codebook_dim_per_group;
    let out_len = batch_size * out_channels * seq_len;

    let tokens_buf = device.new_buffer(
        (tokens_len * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let lengths_buf = device.new_buffer(
        (batch_size * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(
        (out_len * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Fill tokens and lengths
    let lengths: [i32; 2] = [3, 5];
    unsafe {
        let lengths_ptr = lengths_buf.contents() as *mut i32;
        lengths_ptr.add(0).write(lengths[0]);
        lengths_ptr.add(1).write(lengths[1]);
    }

    let tokens: Vec<i32> = (0..tokens_len)
        .map(|i| ((i * 37 + 11) % 2016) as i32)
        .collect();
    unsafe {
        let tokens_ptr = tokens_buf.contents() as *mut i32;
        for (i, &v) in tokens.iter().enumerate() {
            tokens_ptr.add(i).write(v);
        }
    }

    // Run kernel
    let kernel = FsqDecodeKernel::new(&context, KernelDataType::BFloat16)
        .expect("FsqDecodeKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    kernel
        .encode(
            &encoder,
            FsqDecodeArguments {
                tokens: &tokens_buf,
                lengths: &lengths_buf,
                out: &out_buf,
                batch_size,
                num_groups,
                seq_len,
                codebook_dim_per_group,
                num_levels_per_group: num_levels_per_group.into(),
            },
        )
        .expect("encode");

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // Read back and compare (quantized to bf16)
    let mut expected = vec![0.0f32; out_len];
    for b in 0..batch_size {
        for g in 0..num_groups {
            for t in 0..seq_len {
                let token_idx = (b * num_groups + g) * seq_len + t;
                if (t as i32) >= lengths[b] {
                    continue;
                }
                let token = tokens[token_idx];
                let codes = fsq_decode_reference(token, &num_levels_per_group);
                for d in 0..codebook_dim_per_group {
                    let out_c = g * codebook_dim_per_group + d;
                    let out_idx = (b * out_channels + out_c) * seq_len + t;
                    expected[out_idx] = bf16::from_f32(codes[d]).to_f32();
                }
            }
        }
    }

    unsafe {
        let out_ptr = out_buf.contents() as *const bf16;
        let got = std::slice::from_raw_parts(out_ptr, out_len);
        for i in 0..out_len {
            let e = expected[i];
            let g = got[i].to_f32();
            let diff = (e - g).abs();
            assert!(
                diff <= 5e-3,
                "Mismatch at idx={i}: expected {e}, got {g}, diff={diff}"
            );
        }
    }
}

