#![cfg(target_os = "macos")]

use std::mem::size_of;

use bytemuck;
use metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt,
    MTLResourceOptions,
};
use objc2::runtime::ProtocolObject;
use uzu::{
    DataType,
    backends::{
        common::{
            Context,
            kernel::{
                Conv1dScanKernel,
                ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode},
            },
        },
        metal::{Metal, MetalContext, kernel::dsl::Conv1dScanMetalKernel},
    },
};

const STORAGE_MODE: MTLResourceOptions = MTLResourceOptions::STORAGE_MODE_SHARED;

fn silu_scalar(x: f32) -> f32 {
    let y = 1.0 / (1.0 + (-x).exp());
    x * y
}

fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn write_buffer(
    buf: &ProtocolObject<dyn MTLBuffer>,
    data: &[f32],
) {
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents().as_ptr() as *mut f32, data.len());
    }
}

fn read_buffer(
    buf: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.contents().as_ptr() as *const f32, out.as_mut_ptr(), len);
    }
    out
}

fn zero_buffer(buf: &ProtocolObject<dyn MTLBuffer>) {
    unsafe {
        std::ptr::write_bytes(buf.contents().as_ptr(), 0, buf.length() as usize);
    }
}

fn ssd_prefill_cpu_reference(
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: i32,
    x_data: &[f32],
    dt_raw_data: &[f32],
    b_data: &[f32],
    c_data: &[f32],
    d_data: &[f32],
    z_data: &[f32],
    state_init: &[f32],
    x_strides: [usize; 3],
    dt_strides: [usize; 2],
    cb_strides: [usize; 3],
    state_strides: [usize; 3],
) -> (Vec<f32>, Vec<f32>) {
    let total_pairs = suffix_len * num_heads * head_dim;
    let mut y_out = vec![0.0f32; total_pairs];
    let mut state = state_init.to_vec();
    let safe_group = group_size.max(1) as usize;

    for h in 0..num_heads {
        let group_idx = h / safe_group;
        for dh in 0..head_dim {
            let state_base = h * state_strides[0] + dh * state_strides[1];
            for token in 0..suffix_len {
                let x_idx = token * x_strides[0] + h * x_strides[1] + dh * x_strides[2];
                let dt_idx = token * dt_strides[0] + h * dt_strides[1];
                let cb_base = token * cb_strides[0] + group_idx * cb_strides[1];

                let x_val = x_data[x_idx];
                let dt_raw = dt_raw_data[dt_idx];
                let dt_val = softplus_f32(dt_raw);
                let decay_val = (-dt_val).exp();
                let dt_scaled_input = x_val;
                let gate = silu_scalar(z_data[x_idx]);
                let mut acc = d_data[h] * x_val;

                for s in 0..state_dim {
                    let state_idx = state_base + s * state_strides[2];
                    let cb_idx = cb_base + s * cb_strides[2];
                    let b_coeff = b_data[cb_idx];
                    let c_coeff = c_data[cb_idx];
                    let new_state = decay_val * state[state_idx] + dt_scaled_input * b_coeff;
                    state[state_idx] = new_state;
                    acc += new_state * c_coeff;
                }

                y_out[x_idx] = acc * gate;
            }
        }
    }

    (y_out, state)
}

#[allow(dead_code)]
struct SSDPrefillFixture {
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: i32,
    total_x: usize,
    total_dt: usize,
    total_cb: usize,
    total_state: usize,
    x_strides: [usize; 3],
    dt_strides: [usize; 2],
    cb_strides: [usize; 3],
    state_strides: [usize; 3],
    x_data: Vec<f32>,
    dt_data: Vec<f32>,
    b_data: Vec<f32>,
    c_data: Vec<f32>,
    d_data: Vec<f32>,
    z_data: Vec<f32>,
    state_init: Vec<f32>,
}

impl SSDPrefillFixture {
    fn new() -> Self {
        let suffix_len = 512usize;
        let num_heads = 32usize;
        let head_dim = 64usize;
        let state_dim = 64usize;
        let group_size = 1i32;
        let group_count = num_heads / (group_size as usize);
        let total_pairs = num_heads * head_dim;
        let total_x = suffix_len * total_pairs;
        let total_dt = suffix_len * num_heads;
        let total_cb = suffix_len * group_count * state_dim;
        let total_state = num_heads * head_dim * state_dim;

        let x_data: Vec<f32> = (0..total_x).map(|i| ((i % 17) as f32) * 0.01 - 0.05).collect();
        let dt_data: Vec<f32> = (0..total_dt).map(|i| ((i % 13) as f32) * 0.2 - 1.5).collect();
        let b_data: Vec<f32> = (0..total_cb).map(|i| ((i % 11) as f32) * 0.02 - 0.05).collect();
        let c_data: Vec<f32> = (0..total_cb).map(|i| ((i % 19) as f32) * 0.01 - 0.02).collect();
        let d_data: Vec<f32> = (0..num_heads).map(|i| ((i % 3) as f32) * 0.05 - 0.05).collect();
        let z_data: Vec<f32> = (0..total_x).map(|i| ((i % 23) as f32) * 0.02 - 0.1).collect();
        let state_init: Vec<f32> = (0..total_state).map(|i| ((i % 29) as f32) * 0.03 - 0.4).collect();

        let x_strides = [num_heads * head_dim, head_dim, 1usize];
        let dt_strides = [num_heads, 1usize];
        let cb_strides = [group_count * state_dim, state_dim, 1usize];
        let state_strides = [head_dim * state_dim, state_dim, 1usize];

        Self {
            suffix_len,
            num_heads,
            head_dim,
            state_dim,
            group_size,
            total_x,
            total_dt,
            total_cb,
            total_state,
            x_strides,
            dt_strides,
            cb_strides,
            state_strides,
            x_data,
            dt_data,
            b_data,
            c_data,
            d_data,
            z_data,
            state_init,
        }
    }
}

fn run_prefill_kernel_mode(
    ctx: &MetalContext,
    kernel: &SSDPrefillKernels<Metal>,
    fixture: &SSDPrefillFixture,
    mode: SSDPrefillMode,
) -> (Vec<f32>, Vec<f32>, Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>) {
    let device = &ctx.device;
    let x_buf = device
        .new_buffer_with_data(bytemuck::cast_slice(&fixture.x_data), STORAGE_MODE)
        .expect("Failed to create buffer");
    let dt_buf = device
        .new_buffer_with_data(bytemuck::cast_slice(&fixture.dt_data), STORAGE_MODE)
        .expect("Failed to create buffer");
    let b_buf = device
        .new_buffer_with_data(bytemuck::cast_slice(&fixture.b_data), STORAGE_MODE)
        .expect("Failed to create buffer");
    let c_buf = device
        .new_buffer_with_data(bytemuck::cast_slice(&fixture.c_data), STORAGE_MODE)
        .expect("Failed to create buffer");
    let d_buf = device
        .new_buffer_with_data(bytemuck::cast_slice(&fixture.d_data), STORAGE_MODE)
        .expect("Failed to create buffer");
    let z_buf = device
        .new_buffer_with_data(bytemuck::cast_slice(&fixture.z_data), STORAGE_MODE)
        .expect("Failed to create buffer");
    let state_buf = device.new_buffer(fixture.total_state * 4, STORAGE_MODE).expect("Failed to create buffer");
    let y_buf = device.new_buffer(fixture.total_x * 4, STORAGE_MODE).expect("Failed to create buffer");

    write_buffer(&state_buf, &fixture.state_init);
    zero_buffer(&y_buf);

    let args = SSDPrefillArguments::<Metal> {
        x: &x_buf,
        dt: &dt_buf,
        b: &b_buf,
        c: &c_buf,
        d: &d_buf,
        z: &z_buf,
        state: &state_buf,
        y: &y_buf,
        suffix_len: fixture.suffix_len,
        group_size: fixture.group_size,
        state_size: fixture.state_dim as i32,
        x_strides: fixture.x_strides,
        dt_strides: fixture.dt_strides,
        cb_strides: fixture.cb_strides,
        state_strides: fixture.state_strides,
        channels: fixture.num_heads,
        head_dim: fixture.head_dim,
    };

    let command_buffer = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");
    kernel.encode(&encoder, args, mode);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let y_vec = read_buffer(&y_buf, fixture.total_x);
    let state_vec = read_buffer(&state_buf, fixture.total_state);
    (y_vec, state_vec, None)
}

fn run_conv_scan_once(
    ctx: &MetalContext,
    kernel: &Conv1dScanMetalKernel,
    suffix_len: usize,
    channels: usize,
    kernel_size: i32,
    tap_count: usize,
    x_data: &[f32],
    w_data: &[f32],
    b_data: &[f32],
    state_init: &[f32],
    use_scratch: bool,
    alias_io: bool,
) -> (Vec<f32>, Vec<f32>) {
    let device = &ctx.device;
    let total_x = suffix_len * channels;
    let _total_w = channels * kernel_size as usize;
    let total_state = channels * tap_count;

    let y_buf = if alias_io {
        device.new_buffer_with_data(bytemuck::cast_slice(x_data), STORAGE_MODE).expect("Failed to create buffer")
    } else {
        let buf = device.new_buffer(total_x * size_of::<f32>(), STORAGE_MODE).expect("Failed to create buffer");
        zero_buffer(&buf);
        buf
    };
    let w_buf =
        device.new_buffer_with_data(bytemuck::cast_slice(w_data), STORAGE_MODE).expect("Failed to create buffer");
    let b_buf =
        device.new_buffer_with_data(bytemuck::cast_slice(b_data), STORAGE_MODE).expect("Failed to create buffer");
    let state_buf = device.new_buffer(total_state * size_of::<f32>(), STORAGE_MODE).expect("Failed to create buffer");
    let scratch_buf = if use_scratch && tap_count > 0 {
        Some(device.new_buffer(total_state * size_of::<f32>(), STORAGE_MODE).expect("Failed to create buffer"))
    } else {
        None
    };

    write_buffer(&state_buf, state_init);
    if let Some(ref scratch) = scratch_buf {
        zero_buffer(scratch);
    }

    let padded_len = tap_count + suffix_len;
    let padded_buf =
        device.new_buffer(padded_len * channels * size_of::<f32>(), STORAGE_MODE).expect("Failed to create buffer");
    {
        let mut host = vec![0.0f32; padded_len * channels];
        for tap in 0..tap_count {
            for ch in 0..channels {
                host[tap * channels + ch] = state_init[ch * tap_count + tap];
            }
        }
        for token in 0..suffix_len {
            for ch in 0..channels {
                host[(tap_count + token) * channels + ch] = x_data[token * channels + ch];
            }
        }
        write_buffer(&padded_buf, &host);
    }

    let command_buffer = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
    let encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");
    kernel.encode(
        &padded_buf,
        &w_buf,
        Some(&b_buf),
        &y_buf,
        &y_buf,
        &y_buf,
        scratch_buf.as_ref().unwrap_or(&state_buf),
        suffix_len as u32,
        kernel_size as u32,
        channels as u32,
        tap_count as u32,
        channels as u32,
        channels as u32,
        0u32,
        &encoder,
    );
    encoder.end_encoding();

    if let Some(ref scratch) = scratch_buf {
        let bytes = channels * tap_count * size_of::<f32>();
        if bytes > 0 {
            let blit = command_buffer.new_blit_command_encoder().expect("Failed to create blit encoder");
            blit.copy_buffer_to_buffer(scratch, 0, &state_buf, 0, bytes);
            blit.end_encoding();
        }
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let y_vec = read_buffer(&y_buf, total_x);
    let state_vec = read_buffer(&state_buf, total_state);
    (y_vec, state_vec)
}

fn assert_deterministic_for_mode(mode: SSDPrefillMode) {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("Skipping SSD prefill determinism test: no Metal device");
        return;
    };
    let kernel = SSDPrefillKernels::<Metal>::new(&ctx, DataType::F32).unwrap();
    let fixture = SSDPrefillFixture::new();

    let (y_a, state_a, _) = run_prefill_kernel_mode(&ctx, &kernel, &fixture, mode);
    let (y_b, state_b, _) = run_prefill_kernel_mode(&ctx, &kernel, &fixture, mode);

    assert_eq!(y_a, y_b, "Prefill outputs differ in {:?} mode", mode);
    assert_eq!(state_a, state_b, "Prefill states differ in {:?} mode", mode);
}

fn assert_matches_cpu_reference(mode: SSDPrefillMode) {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("Skipping SSD prefill reference test: no Metal device");
        return;
    };
    let kernel = SSDPrefillKernels::<Metal>::new(&ctx, DataType::F32).unwrap();
    let fixture = SSDPrefillFixture::new();

    let (y_ref, state_ref) = ssd_prefill_cpu_reference(
        fixture.suffix_len,
        fixture.num_heads,
        fixture.head_dim,
        fixture.state_dim,
        fixture.group_size,
        &fixture.x_data,
        &fixture.dt_data,
        &fixture.b_data,
        &fixture.c_data,
        &fixture.d_data,
        &fixture.z_data,
        &fixture.state_init,
        fixture.x_strides,
        fixture.dt_strides,
        fixture.cb_strides,
        fixture.state_strides,
    );

    let (y_gpu, state_gpu, _) = run_prefill_kernel_mode(&ctx, &kernel, &fixture, mode);

    let tolerance = 5e-5f32;
    let mut max_y_diff = 0.0f32;
    let mut max_y_idx = 0usize;
    for (idx, (&lhs, &rhs)) in y_gpu.iter().zip(&y_ref).enumerate() {
        let diff = (lhs - rhs).abs();
        if diff > max_y_diff {
            max_y_diff = diff;
            max_y_idx = idx;
        }
    }
    assert!(
        max_y_diff <= tolerance,
        "Prefill outputs diverge in {:?} mode at idx {max_y_idx}: metal={} cpu={} (diff {max_y_diff})",
        mode,
        y_gpu[max_y_idx],
        y_ref[max_y_idx]
    );

    let mut max_state_diff = 0.0f32;
    let mut max_state_idx = 0usize;
    for (idx, (&lhs, &rhs)) in state_gpu.iter().zip(&state_ref).enumerate() {
        let diff = (lhs - rhs).abs();
        if diff > max_state_diff {
            max_state_diff = diff;
            max_state_idx = idx;
        }
    }
    assert!(
        max_state_diff <= tolerance,
        "Prefill states diverge in {:?} mode at idx {max_state_idx}: metal={} cpu={} (diff {max_state_diff})",
        mode,
        state_gpu[max_state_idx],
        state_ref[max_state_idx]
    );
}

#[test]
fn ssd_prefill_sequential_is_deterministic() {
    assert_deterministic_for_mode(SSDPrefillMode::Sequential);
}

#[test]
fn ssd_prefill_single_pass_is_deterministic() {
    assert_deterministic_for_mode(SSDPrefillMode::SinglePass);
}

#[test]
fn ssd_prefill_sequential_matches_cpu_reference() {
    assert_matches_cpu_reference(SSDPrefillMode::Sequential);
}

#[test]
fn ssd_prefill_single_pass_matches_cpu_reference() {
    assert_matches_cpu_reference(SSDPrefillMode::SinglePass);
}

#[test]
fn conv1d_scan_is_deterministic() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("Skipping conv1d scan determinism test: no Metal device");
        return;
    };
    let kernel = Conv1dScanMetalKernel::new(&ctx, DataType::F32, 0u32, true).unwrap();

    let suffix_len = 192usize;
    let channels = 8usize;
    let kernel_size = 5i32;
    let tap_count = (kernel_size - 1) as usize;

    let total_x = suffix_len * channels;
    let total_state = channels * tap_count;

    let x_data: Vec<f32> = (0..total_x).map(|i| ((i % 31) as f32) * 0.02 - 0.3).collect();
    let w_data: Vec<f32> = (0..(channels * kernel_size as usize)).map(|i| ((i % 17) as f32) * 0.01 - 0.04).collect();
    let b_data: Vec<f32> = (0..channels).map(|i| ((i % 5) as f32) * 0.03 - 0.07).collect();
    let state_init: Vec<f32> = (0..total_state).map(|i| ((i % 23) as f32) * 0.02 - 0.1).collect();

    let use_scratch = tap_count > 0 && suffix_len > 1;

    for &alias_io in &[false, true] {
        let first = run_conv_scan_once(
            &ctx,
            &kernel,
            suffix_len,
            channels,
            kernel_size,
            tap_count,
            &x_data,
            &w_data,
            &b_data,
            &state_init,
            use_scratch,
            alias_io,
        );
        let second = run_conv_scan_once(
            &ctx,
            &kernel,
            suffix_len,
            channels,
            kernel_size,
            tap_count,
            &x_data,
            &w_data,
            &b_data,
            &state_init,
            use_scratch,
            alias_io,
        );

        assert_eq!(first.0, second.0, "Conv outputs differ (alias_io={alias_io})");
        assert_eq!(first.1, second.1, "Conv states differ (alias_io={alias_io})");
    }
}
