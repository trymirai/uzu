#![cfg(target_os = "macos")]

use metal::{Device, MTLResourceOptions};

use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::{AddKernel, ClampKernel, HalfSnakeKernel, LeakyReluKernel, ScaleKernel, TanhKernel},
};

fn create_test_context() -> Result<MTLContext, String> {
    let device = Device::system_default().ok_or("No Metal device available")?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue)
        .map_err(|e| format!("Failed to create MTLContext: {e:?}"))
}

#[test]
fn audio_leaky_relu_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let n = 1024usize;
    let slope = 0.01f32;

    let input_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.01 - 5.12)
        .collect();
    unsafe {
        let ptr = input_buf.contents() as *mut f32;
        for (i, &v) in input.iter().enumerate() {
            ptr.add(i).write(v);
        }
    }

    let kernel = LeakyReluKernel::new(&context, KernelDataType::Float32)
        .expect("LeakyReluKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(&encoder, &input_buf, &output_buf, n, slope)
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let out_ptr = output_buf.contents() as *const f32;
        let got = std::slice::from_raw_parts(out_ptr, n);
        for i in 0..n {
            let x = input[i];
            let expected = if x >= 0.0 { x } else { slope * x };
            let diff = (expected - got[i]).abs();
            assert!(
                diff <= 1e-6,
                "Mismatch at i={i}: expected {expected}, got {}, diff={diff}",
                got[i]
            );
        }
    }
}

#[test]
fn audio_tanh_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let n = 1024usize;

    let input_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.01 - 5.12)
        .collect();
    unsafe {
        let ptr = input_buf.contents() as *mut f32;
        for (i, &v) in input.iter().enumerate() {
            ptr.add(i).write(v);
        }
    }

    let kernel =
        TanhKernel::new(&context, KernelDataType::Float32).expect("TanhKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(&encoder, &input_buf, &output_buf, n)
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let out_ptr = output_buf.contents() as *const f32;
        let got = std::slice::from_raw_parts(out_ptr, n);
        for i in 0..n {
            let expected = input[i].tanh();
            let diff = (expected - got[i]).abs();
            assert!(
                diff <= 1e-6,
                "Mismatch at i={i}: expected {expected}, got {}, diff={diff}",
                got[i]
            );
        }
    }
}

#[test]
fn audio_add_and_scale_match_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let n = 2048usize;

    let a_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let sum_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scaled_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.001 - 1.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002).sin()).collect();

    unsafe {
        let a_ptr = a_buf.contents() as *mut f32;
        let b_ptr = b_buf.contents() as *mut f32;
        for i in 0..n {
            a_ptr.add(i).write(a[i]);
            b_ptr.add(i).write(b[i]);
        }
    }

    let add = AddKernel::new(&context, KernelDataType::Float32)
        .expect("AddKernel");
    let scale = ScaleKernel::new(&context, KernelDataType::Float32)
        .expect("ScaleKernel");

    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    add.encode(&encoder, &a_buf, &b_buf, &sum_buf, n)
        .expect("add encode");
    scale
        .encode(&encoder, &sum_buf, &scaled_buf, n, 1.0 / 3.0)
        .expect("scale encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let sum_ptr = sum_buf.contents() as *const f32;
        let scaled_ptr = scaled_buf.contents() as *const f32;
        let sum_got = std::slice::from_raw_parts(sum_ptr, n);
        let scaled_got = std::slice::from_raw_parts(scaled_ptr, n);

        for i in 0..n {
            let exp_sum = a[i] + b[i];
            let exp_scaled = exp_sum * (1.0 / 3.0);
            let diff_sum = (exp_sum - sum_got[i]).abs();
            let diff_scaled = (exp_scaled - scaled_got[i]).abs();
            assert!(
                diff_sum <= 1e-6,
                "Sum mismatch at i={i}: expected {exp_sum}, got {}, diff={diff_sum}",
                sum_got[i]
            );
            assert!(
                diff_scaled <= 1e-6,
                "Scale mismatch at i={i}: expected {exp_scaled}, got {}, diff={diff_scaled}",
                scaled_got[i]
            );
        }
    }
}

#[test]
fn audio_clamp_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let n = 2048usize;
    let min_v = -1.0f32;
    let max_v = 1.0f32;

    let input_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.01 - 10.24)
        .collect();
    unsafe {
        let ptr = input_buf.contents() as *mut f32;
        for (i, &v) in input.iter().enumerate() {
            ptr.add(i).write(v);
        }
    }

    let kernel =
        ClampKernel::new(&context, KernelDataType::Float32).expect("ClampKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(&encoder, &input_buf, &output_buf, n, min_v, max_v)
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let out_ptr = output_buf.contents() as *const f32;
        let got = std::slice::from_raw_parts(out_ptr, n);
        for i in 0..n {
            let expected = input[i].max(min_v).min(max_v);
            let diff = (expected - got[i]).abs();
            assert!(
                diff <= 1e-6,
                "Mismatch at i={i}: expected {expected}, got {}, diff={diff}",
                got[i]
            );
        }
    }
}

#[test]
fn audio_half_snake_matches_reference_f32() {
    let context = create_test_context().expect("MTLContext");
    let device = &context.device;

    let batch_size = 1usize;
    let channels = 6usize;
    let snake_channels = channels / 2;
    let seq_len = 64usize;
    let n = batch_size * channels * seq_len;
    let slope = 0.01f32;
    let eps = 1e-9f32;

    let input_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let alpha_buf = device.new_buffer(
        (snake_channels * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (n * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let input: Vec<f32> = (0..n)
        .map(|i| (i as f32 * 0.01 - 3.2).sin())
        .collect();
    let alpha: Vec<f32> = (0..snake_channels)
        .map(|i| 0.5 + i as f32 * 0.1)
        .collect();

    unsafe {
        let x_ptr = input_buf.contents() as *mut f32;
        for (i, &v) in input.iter().enumerate() {
            x_ptr.add(i).write(v);
        }
        let a_ptr = alpha_buf.contents() as *mut f32;
        for (i, &v) in alpha.iter().enumerate() {
            a_ptr.add(i).write(v);
        }
    }

    let kernel = HalfSnakeKernel::new(&context, KernelDataType::Float32)
        .expect("HalfSnakeKernel");
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel
        .encode(
            &encoder,
            &input_buf,
            &alpha_buf,
            &output_buf,
            batch_size,
            channels,
            seq_len,
            snake_channels,
            slope,
            eps,
        )
        .expect("encode");
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let out_ptr = output_buf.contents() as *const f32;
        let got = std::slice::from_raw_parts(out_ptr, n);
        for b in 0..batch_size {
            for c in 0..channels {
                for t in 0..seq_len {
                    let idx = (b * channels + c) * seq_len + t;
                    let x = input[idx];
                    let expected = if c < snake_channels {
                        let a = alpha[c];
                        let s = (a * x).sin();
                        x + (s * s) / (a + eps)
                    } else if x >= 0.0 {
                        x
                    } else {
                        slope * x
                    };
                    let diff = (expected - got[idx]).abs();
                    assert!(
                        diff <= 1e-5,
                        "Mismatch at b={b} c={c} t={t}: expected {expected}, got {}, diff={diff}",
                        got[idx]
                    );
                }
            }
        }
    }
}

