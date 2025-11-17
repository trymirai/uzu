#![cfg(target_os = "macos")]

use std::mem::size_of;

use metal::MTLResourceOptions;
use uzu::backends::metal::kernel::ssm::conv1d_scan::{
    Conv1dScanArguments, Conv1dScanKernel,
};
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::ssm::{SSDPrefillArguments, SSDPrefillKernel, SSDPrefillMode},
};
use uzu::config::Activation;

const STORAGE_MODE: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

fn silu_scalar(x: f32) -> f32 {
    let y = 1.0 / (1.0 + (-x).exp());
    x * y
}

fn create_context() -> Option<MTLContext> {
    let device = metal::Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn write_buffer(
    buf: &metal::BufferRef,
    data: &[f32],
) {
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents() as *mut f32,
            data.len(),
        );
    }
}

fn read_buffer(
    buf: &metal::BufferRef,
    len: usize,
) -> Vec<f32> {
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
        std::ptr::write_bytes(buf.contents(), 0, buf.length() as usize);
    }
}

fn ssd_prefill_cpu_reference(
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: i32,
    x_data: &[f32],
    dt_data: &[f32],
    decay_data: &[f32],
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
                let x_idx =
                    token * x_strides[0] + h * x_strides[1] + dh * x_strides[2];
                let dt_idx = token * dt_strides[0] + h * dt_strides[1];
                let cb_base = token * cb_strides[0] + group_idx * cb_strides[1];

                let x_val = x_data[x_idx];
                let dt_val = dt_data[dt_idx];
                let decay_val = decay_data[dt_idx];
                let dt_safe = dt_val.max(1e-6);
                let dt_scaled_input = (x_val / dt_safe) * dt_val;
                let gate = silu_scalar(z_data[x_idx]);
                let mut acc = d_data[h] * x_val;

                for s in 0..state_dim {
                    let state_idx = state_base + s * state_strides[2];
                    let cb_idx = cb_base + s * cb_strides[2];
                    let b_coeff = b_data[cb_idx];
                    let c_coeff = c_data[cb_idx];
                    let new_state = decay_val * state[state_idx]
                        + dt_scaled_input * b_coeff;
                    state[state_idx] = new_state;
                    acc += new_state * c_coeff;
                }

                y_out[x_idx] = acc * gate;
            }
        }
    }

    (y_out, state)
}

struct SSDPrefillFixture {
    suffix_len: usize,
    num_heads: usize,
    head_dim: usize,
    state_dim: usize,
    group_size: i32,
    chunk_count: usize,
    total_pairs: usize,
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
    decay_data: Vec<f32>,
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
        let chunk_size =
            uzu::backends::metal::kernel::ssm::ssd_prefill::SSD_PREFILL_CHUNK;
        let chunk_count = (suffix_len + chunk_size - 1) / chunk_size;
        let group_count = num_heads / (group_size as usize);
        let total_pairs = num_heads * head_dim;
        let total_x = suffix_len * total_pairs;
        let total_dt = suffix_len * num_heads;
        let total_cb = suffix_len * group_count * state_dim;
        let total_state = num_heads * head_dim * state_dim;

        let x_data: Vec<f32> =
            (0..total_x).map(|i| ((i % 17) as f32) * 0.01 - 0.05).collect();
        let dt_data: Vec<f32> =
            (0..total_dt).map(|i| 0.01 + ((i % 13) as f32) * 0.005).collect();
        let decay_data: Vec<f32> =
            (0..total_dt).map(|i| 0.5 + ((i % 7) as f32) * 0.01).collect();
        let b_data: Vec<f32> =
            (0..total_cb).map(|i| ((i % 11) as f32) * 0.02 - 0.05).collect();
        let c_data: Vec<f32> =
            (0..total_cb).map(|i| ((i % 19) as f32) * 0.01 - 0.02).collect();
        let d_data: Vec<f32> =
            (0..num_heads).map(|i| ((i % 3) as f32) * 0.05 - 0.05).collect();
        let z_data: Vec<f32> =
            (0..total_x).map(|i| ((i % 23) as f32) * 0.02 - 0.1).collect();
        let state_init: Vec<f32> =
            (0..total_state).map(|i| ((i % 29) as f32) * 0.03 - 0.4).collect();

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
            chunk_count,
            total_pairs,
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
            decay_data,
            b_data,
            c_data,
            d_data,
            z_data,
            state_init,
        }
    }
}

fn run_prefill_kernel_mode(
    ctx: &MTLContext,
    kernel: &SSDPrefillKernel,
    fixture: &SSDPrefillFixture,
    mode: SSDPrefillMode,
) -> (Vec<f32>, Vec<f32>) {
    let device = &ctx.device;
    let x_buf = device.new_buffer_with_data(
        fixture.x_data.as_ptr() as *const _,
        (fixture.total_x * 4) as u64,
        STORAGE_MODE,
    );
    let dt_buf = device.new_buffer_with_data(
        fixture.dt_data.as_ptr() as *const _,
        (fixture.total_dt * 4) as u64,
        STORAGE_MODE,
    );
    let decay_buf = device.new_buffer_with_data(
        fixture.decay_data.as_ptr() as *const _,
        (fixture.total_dt * 4) as u64,
        STORAGE_MODE,
    );
    let b_buf = device.new_buffer_with_data(
        fixture.b_data.as_ptr() as *const _,
        (fixture.total_cb * 4) as u64,
        STORAGE_MODE,
    );
    let c_buf = device.new_buffer_with_data(
        fixture.c_data.as_ptr() as *const _,
        (fixture.total_cb * 4) as u64,
        STORAGE_MODE,
    );
    let d_buf = device.new_buffer_with_data(
        fixture.d_data.as_ptr() as *const _,
        (fixture.num_heads * 4) as u64,
        STORAGE_MODE,
    );
    let z_buf = device.new_buffer_with_data(
        fixture.z_data.as_ptr() as *const _,
        (fixture.total_x * 4) as u64,
        STORAGE_MODE,
    );
    let state_buf =
        device.new_buffer((fixture.total_state * 4) as u64, STORAGE_MODE);
    let y_buf = device.new_buffer((fixture.total_x * 4) as u64, STORAGE_MODE);

    let chunk_a_len = fixture.chunk_count * fixture.total_pairs;
    let chunk_b_len = chunk_a_len * fixture.state_dim;
    let chunk_prefix_len =
        fixture.chunk_count * fixture.total_pairs * fixture.state_dim;
    let chunk_a_buf = device.new_buffer((chunk_a_len * 4) as u64, STORAGE_MODE);
    let chunk_b_buf = device.new_buffer((chunk_b_len * 4) as u64, STORAGE_MODE);
    let chunk_prefix_buf =
        device.new_buffer((chunk_prefix_len * 4) as u64, STORAGE_MODE);

    write_buffer(&state_buf, &fixture.state_init);
    zero_buffer(&y_buf);
    zero_buffer(&chunk_a_buf);
    zero_buffer(&chunk_b_buf);
    zero_buffer(&chunk_prefix_buf);

    let matrix_args = if matches!(mode, SSDPrefillMode::Matrix) {
        let dt_total = fixture.total_dt;
        let chunk_total = fixture.chunk_count * fixture.num_heads;
        let group_count = fixture.num_heads / (fixture.group_size as usize);
        let square_heads =
            fixture.num_heads * fixture.suffix_len * fixture.suffix_len;
        let square_groups =
            group_count * fixture.suffix_len * fixture.suffix_len;
        let pack_group = group_count * fixture.suffix_len * fixture.state_dim;
        let pack_group_t = group_count * fixture.state_dim * fixture.suffix_len;
        let dtx_total =
            fixture.num_heads * fixture.suffix_len * fixture.head_dim;
        let dtx_decay_total =
            fixture.num_heads * fixture.head_dim * fixture.suffix_len;
        let b_head_total =
            fixture.num_heads * fixture.suffix_len * fixture.state_dim;
        let c_head_total =
            fixture.num_heads * fixture.state_dim * fixture.suffix_len;

        let make_zero = |len: usize| -> metal::Buffer {
            let buf = device.new_buffer((len * 4) as u64, STORAGE_MODE);
            zero_buffer(&buf);
            buf
        };

        Some(uzu::backends::metal::kernel::ssm::ssd_prefill::SSDPrefillMatrixArguments {
            dt_a: make_zero(dt_total),
            prefix: make_zero(dt_total),
            chunk_sums: make_zero(chunk_total),
            chunk_offsets: make_zero(chunk_total),
            decay_matrix: make_zero(square_heads),
            decay_last: make_zero(dt_total),
            c_packed: make_zero(pack_group),
            b_packed: make_zero(pack_group_t),
            cb_groups: make_zero(square_groups),
            cb_heads: make_zero(square_heads),
            attn: make_zero(square_heads),
            dtx: make_zero(dtx_total),
            y_tmp: make_zero(dtx_total),
            dtxdecay: make_zero(dtx_decay_total),
            b_head: make_zero(b_head_total),
            c_head_transposed: make_zero(c_head_total),
        })
    } else {
        None
    };

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
        suffix_len: fixture.suffix_len,
        group_size: fixture.group_size,
        state_size: fixture.state_dim as i32,
        x_strides: fixture.x_strides,
        dt_strides: fixture.dt_strides,
        cb_strides: fixture.cb_strides,
        state_strides: fixture.state_strides,
        channels: fixture.num_heads,
        head_dim: fixture.head_dim,
        matrix: matrix_args,
    };

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel.encode(encoder, args, mode).unwrap();
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let y_vec = read_buffer(&y_buf, fixture.total_x);
    let state_vec = read_buffer(&state_buf, fixture.total_state);
    (y_vec, state_vec)
}

fn run_conv_scan_once(
    ctx: &MTLContext,
    kernel: &Conv1dScanKernel,
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
    let total_w = channels * kernel_size as usize;
    let total_state = channels * tap_count;

    let y_buf = if alias_io {
        device.new_buffer_with_data(
            x_data.as_ptr() as *const _,
            (total_x * size_of::<f32>()) as u64,
            STORAGE_MODE,
        )
    } else {
        let buf = device
            .new_buffer((total_x * size_of::<f32>()) as u64, STORAGE_MODE);
        zero_buffer(&buf);
        buf
    };
    let w_buf = device.new_buffer_with_data(
        w_data.as_ptr() as *const _,
        (total_w * size_of::<f32>()) as u64,
        STORAGE_MODE,
    );
    let b_buf = device.new_buffer_with_data(
        b_data.as_ptr() as *const _,
        (channels * size_of::<f32>()) as u64,
        STORAGE_MODE,
    );
    let state_buf = device
        .new_buffer((total_state * size_of::<f32>()) as u64, STORAGE_MODE);
    let scratch_buf =
        if use_scratch && tap_count > 0 {
            Some(device.new_buffer(
                (total_state * size_of::<f32>()) as u64,
                STORAGE_MODE,
            ))
        } else {
            None
        };

    write_buffer(&state_buf, state_init);
    if let Some(ref scratch) = scratch_buf {
        zero_buffer(scratch);
    }

    let padded_len = tap_count + suffix_len;
    let padded_buf = device.new_buffer(
        (padded_len * channels * size_of::<f32>()) as u64,
        STORAGE_MODE,
    );
    {
        let mut host = vec![0.0f32; padded_len * channels];
        for tap in 0..tap_count {
            for ch in 0..channels {
                host[tap * channels + ch] = state_init[ch * tap_count + tap];
            }
        }
        for token in 0..suffix_len {
            for ch in 0..channels {
                host[(tap_count + token) * channels + ch] =
                    x_data[token * channels + ch];
            }
        }
        write_buffer(&padded_buf, &host);
    }

    let args = Conv1dScanArguments {
        padded: &padded_buf,
        w: &w_buf,
        b: Some(&b_buf),
        y: &y_buf,
        state_out: scratch_buf.as_ref().unwrap_or(&state_buf),
        suffix_len,
        kernel_size,
        row_stride: channels,
        state_stride: tap_count,
        channels,
    };

    let command_buffer = ctx.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    kernel.encode(encoder, args).unwrap();
    encoder.end_encoding();

    if let Some(ref scratch) = scratch_buf {
        let bytes = (channels * tap_count * size_of::<f32>()) as u64;
        if bytes > 0 {
            let blit = command_buffer.new_blit_command_encoder();
            blit.copy_from_buffer(scratch, 0, &state_buf, 0, bytes);
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
    let Some(ctx) = create_context() else {
        eprintln!("Skipping SSD prefill determinism test: no Metal device");
        return;
    };
    let kernel = SSDPrefillKernel::new(&ctx, KernelDataType::Float32).unwrap();
    let fixture = SSDPrefillFixture::new();

    let run_a = run_prefill_kernel_mode(&ctx, &kernel, &fixture, mode);
    let run_b = run_prefill_kernel_mode(&ctx, &kernel, &fixture, mode);

    assert_eq!(run_a.0, run_b.0, "Prefill outputs differ in {:?} mode", mode);
    assert_eq!(run_a.1, run_b.1, "Prefill states differ in {:?} mode", mode);
}

fn assert_matches_cpu_reference(mode: SSDPrefillMode) {
    let Some(ctx) = create_context() else {
        eprintln!("Skipping SSD prefill reference test: no Metal device");
        return;
    };
    let kernel = SSDPrefillKernel::new(&ctx, KernelDataType::Float32).unwrap();
    let fixture = SSDPrefillFixture::new();

    let (y_ref, state_ref) = ssd_prefill_cpu_reference(
        fixture.suffix_len,
        fixture.num_heads,
        fixture.head_dim,
        fixture.state_dim,
        fixture.group_size,
        &fixture.x_data,
        &fixture.dt_data,
        &fixture.decay_data,
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

    let (y_gpu, state_gpu) =
        run_prefill_kernel_mode(&ctx, &kernel, &fixture, mode);

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
#[ignore = "Multi-pass kernels still under investigation for determinism"]
fn ssd_prefill_multi_pass_is_deterministic() {
    assert_deterministic_for_mode(SSDPrefillMode::MultiPass);
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
#[ignore = "Multi-pass kernels still under investigation for determinism"]
fn ssd_prefill_multi_pass_matches_cpu_reference() {
    assert_matches_cpu_reference(SSDPrefillMode::MultiPass);
}

#[test]
fn ssd_prefill_matrix_is_deterministic() {
    assert_deterministic_for_mode(SSDPrefillMode::Matrix);
}

#[test]
fn ssd_prefill_matrix_matches_cpu_reference() {
    assert_matches_cpu_reference(SSDPrefillMode::Matrix);
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

    let x_data: Vec<f32> =
        (0..total_x).map(|i| ((i % 31) as f32) * 0.02 - 0.3).collect();
    let w_data: Vec<f32> = (0..(channels * kernel_size as usize))
        .map(|i| ((i % 17) as f32) * 0.01 - 0.04)
        .collect();
    let b_data: Vec<f32> =
        (0..channels).map(|i| ((i % 5) as f32) * 0.03 - 0.07).collect();
    let state_init: Vec<f32> =
        (0..total_state).map(|i| ((i % 23) as f32) * 0.02 - 0.1).collect();

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

        assert_eq!(
            first.0, second.0,
            "Conv outputs differ (alias_io={alias_io})"
        );
        assert_eq!(
            first.1, second.1,
            "Conv states differ (alias_io={alias_io})"
        );
    }
}
