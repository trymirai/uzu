#![cfg(target_os = "macos")]

use metal::MTLResourceOptions;
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::ssm::{
        SSDPrefillArguments, SSDPrefillKernel, SSDPrefillMode,
    },
};
use uzu::backends::metal::kernel::ssm::conv1d_scan::{
    Conv1dScanArguments, Conv1dScanKernel,
};
use uzu::config::Activation;

const STORAGE_MODE: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

fn create_context() -> Option<MTLContext> {
    let device = metal::Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn write_buffer(buf: &metal::BufferRef, data: &[f32]) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut f32,
            data.len(),
        );
    }
}

fn read_buffer(buf: &metal::BufferRef, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf.contents() as *const f32,
            out.as_mut_ptr(),
            len,
        );
    }
    out
}

fn zero_buffer(buf: &metal::BufferRef) {
    unsafe {
        std::ptr::write_bytes(
            buf.contents(),
            0,
            buf.length() as usize,
        );
    }
}

#[test]
fn ssd_prefill_single_pass_is_deterministic() {
    let Some(ctx) = create_context() else {
        eprintln!("Skipping SSD prefill determinism test: no Metal device");
        return;
    };
    let kernel =
        SSDPrefillKernel::new(&ctx, KernelDataType::Float32).unwrap();

    let suffix_len = 512usize;
    let num_heads = 32usize;
    let head_dim = 64usize;
    let state_dim = 64usize;
    let group_size = 1i32;
    let state_size = state_dim as i32;
    let chunk_size = uzu::backends::metal::kernel::ssm::ssd_prefill::SSD_PREFILL_CHUNK;
    let chunk_count = (suffix_len + chunk_size - 1) / chunk_size;
    let group_count = num_heads / (group_size as usize);

    let total_pairs = num_heads * head_dim;
    let total_x = suffix_len * total_pairs;
    let total_dt = suffix_len * num_heads;
    let total_cb = suffix_len * group_count * state_dim;
    let total_state = num_heads * head_dim * state_dim;

    let x_data: Vec<f32> = (0..total_x)
        .map(|i| ((i % 17) as f32) * 0.01 - 0.05)
        .collect();
    let dt_data: Vec<f32> = (0..total_dt)
        .map(|i| 0.01 + ((i % 13) as f32) * 0.005)
        .collect();
    let decay_data: Vec<f32> = (0..total_dt)
        .map(|i| 0.5 + ((i % 7) as f32) * 0.01)
        .collect();
    let b_data: Vec<f32> = (0..total_cb)
        .map(|i| ((i % 11) as f32) * 0.02 - 0.05)
        .collect();
    let c_data: Vec<f32> = (0..total_cb)
        .map(|i| ((i % 19) as f32) * 0.01 - 0.02)
        .collect();
    let d_data: Vec<f32> = (0..num_heads)
        .map(|i| ((i % 3) as f32) * 0.05 - 0.05)
        .collect();
    let z_data: Vec<f32> = (0..total_x)
        .map(|i| ((i % 23) as f32) * 0.02 - 0.1)
        .collect();
    let state_init: Vec<f32> = (0..total_state)
        .map(|i| ((i % 29) as f32) * 0.03 - 0.4)
        .collect();

    let device = &ctx.device;
    let x_buf = device.new_buffer_with_data(
        x_data.as_ptr() as *const _,
        (total_x * 4) as u64,
        STORAGE_MODE,
    );
    let dt_buf = device.new_buffer_with_data(
        dt_data.as_ptr() as *const _,
        (total_dt * 4) as u64,
        STORAGE_MODE,
    );
    let decay_buf = device.new_buffer_with_data(
        decay_data.as_ptr() as *const _,
        (total_dt * 4) as u64,
        STORAGE_MODE,
    );
    let b_buf = device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        (total_cb * 4) as u64,
        STORAGE_MODE,
    );
    let c_buf = device.new_buffer_with_data(
        c_data.as_ptr() as *const _,
        (total_cb * 4) as u64,
        STORAGE_MODE,
    );
    let d_buf = device.new_buffer_with_data(
        d_data.as_ptr() as *const _,
        (num_heads * 4) as u64,
        STORAGE_MODE,
    );
    let z_buf = device.new_buffer_with_data(
        z_data.as_ptr() as *const _,
        (total_x * 4) as u64,
        STORAGE_MODE,
    );

    let state_buf =
        device.new_buffer((total_state * 4) as u64, STORAGE_MODE);
    let y_buf = device.new_buffer((total_x * 4) as u64, STORAGE_MODE);

    // Dummy chunk buffers (unused in single-pass mode but required by API).
    let chunk_a_len = chunk_count * total_pairs;
    let chunk_b_len = chunk_a_len * state_dim;
    let chunk_prefix_len = chunk_count * total_pairs * state_dim;
    let chunk_a_buf =
        device.new_buffer((chunk_a_len * 4) as u64, STORAGE_MODE);
    let chunk_b_buf =
        device.new_buffer((chunk_b_len * 4) as u64, STORAGE_MODE);
    let chunk_prefix_buf =
        device.new_buffer((chunk_prefix_len * 4) as u64, STORAGE_MODE);

    let x_strides = [
        num_heads * head_dim,
        head_dim,
        1usize,
    ];
    let dt_strides = [num_heads, 1usize];
    let cb_strides = [
        group_count * state_dim,
        state_dim,
        1usize,
    ];
    let state_strides = [
        head_dim * state_dim,
        state_dim,
        1usize,
    ];

    let mut results = Vec::new();

    for _run in 0..2 {
        write_buffer(&state_buf, &state_init);
        zero_buffer(&y_buf);
        zero_buffer(&chunk_a_buf);
        zero_buffer(&chunk_b_buf);
        zero_buffer(&chunk_prefix_buf);

        let args = SSDPrefillArguments {
            x: &x_buf,
            dt: &dt_buf,
            decay: &decay_buf,
            b: &b_buf,
            c: &c_buf,
            d: &d_buf,
            z: &z_buf,
            state: &state_buf,
            y: &y_buf,
            chunk_a: &chunk_a_buf,
            chunk_b: &chunk_b_buf,
            chunk_prefix: &chunk_prefix_buf,
            suffix_len,
            group_size,
            state_size,
            x_strides,
            dt_strides,
            cb_strides,
            state_strides,
            channels: num_heads,
            head_dim,
        };

        let command_buffer = ctx.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        kernel
            .encode(encoder, args, SSDPrefillMode::SinglePass)
            .unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let y_vec = read_buffer(&y_buf, total_x);
        let state_vec = read_buffer(&state_buf, total_state);
        results.push((y_vec, state_vec));
    }

    let mut max_diff = 0.0f32;
    for (idx, (lhs, rhs)) in results[0].0.iter().zip(&results[1].0).enumerate() {
        if lhs != rhs {
            let diff = (lhs - rhs).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            println!("y mismatch at {}: {} vs {}", idx, lhs, rhs);
            break;
        }
    }
    for (idx, (lhs, rhs)) in results[0].1.iter().zip(&results[1].1).enumerate() {
        if lhs != rhs {
            let diff = (lhs - rhs).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            println!("state mismatch at {}: {} vs {}", idx, lhs, rhs);
            break;
        }
    }
    assert_eq!(results[0].0, results[1].0, "Prefill outputs differ (max diff {max_diff})");
    assert_eq!(results[0].1, results[1].1, "Prefill states differ (max diff {max_diff})");
}

#[test]
fn conv1d_scan_is_deterministic() {
    let Some(ctx) = create_context() else {
        eprintln!("Skipping conv1d scan determinism test: no Metal device");
        return;
    };
    let activation = Activation::Identity;
    let kernel =
        Conv1dScanKernel::new(&ctx, KernelDataType::Float32, &activation)
            .unwrap();

    let suffix_len = 192usize;
    let channels = 8usize;
    let kernel_size = 5i32;
    let tap_count = (kernel_size - 1) as usize;

    let total_x = suffix_len * channels;
    let total_state = channels * tap_count;

    let x_data: Vec<f32> = (0..total_x)
        .map(|i| ((i % 31) as f32) * 0.02 - 0.3)
        .collect();
    let w_data: Vec<f32> = (0..(channels * kernel_size as usize))
        .map(|i| ((i % 17) as f32) * 0.01 - 0.04)
        .collect();
    let b_data: Vec<f32> = (0..channels)
        .map(|i| ((i % 5) as f32) * 0.03 - 0.07)
        .collect();
    let state_init: Vec<f32> = (0..total_state)
        .map(|i| ((i % 23) as f32) * 0.02 - 0.1)
        .collect();

    let device = &ctx.device;
    let x_buf = device.new_buffer_with_data(
        x_data.as_ptr() as *const _,
        (total_x * 4) as u64,
        STORAGE_MODE,
    );
    let w_buf = device.new_buffer_with_data(
        w_data.as_ptr() as *const _,
        (channels * kernel_size as usize * 4) as u64,
        STORAGE_MODE,
    );
    let b_buf = device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        (channels * 4) as u64,
        STORAGE_MODE,
    );
    let state_buf =
        device.new_buffer((total_state * 4) as u64, STORAGE_MODE);
    let y_buf = device.new_buffer((total_x * 4) as u64, STORAGE_MODE);

    let row_stride = channels;
    let state_stride = tap_count;

    let mut results = Vec::new();

    for _run in 0..2 {
        write_buffer(&state_buf, &state_init);
        zero_buffer(&y_buf);

        let args = Conv1dScanArguments {
            x: &x_buf,
            w: &w_buf,
            b: Some(&b_buf),
            state: &state_buf,
            y: &y_buf,
            suffix_len,
            kernel_size,
            row_stride,
            state_stride,
            channels,
        };

        let command_buffer = ctx.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        kernel.encode(encoder, args).unwrap();
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let y_vec = read_buffer(&y_buf, total_x);
        let state_vec = read_buffer(&state_buf, total_state);
        results.push((y_vec, state_vec));
    }

    assert_eq!(results[0].0, results[1].0, "Conv outputs differ");
    assert_eq!(results[0].1, results[1].1, "Conv states differ");
}
