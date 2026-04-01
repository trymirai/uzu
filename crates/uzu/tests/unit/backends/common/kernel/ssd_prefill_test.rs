use std::{
    mem::size_of,
    ops::{Deref, DerefMut},
};

use uzu::backends::common::Buffer;

use crate::{
    ArrayContextExt, DataType,
    backends::common::{
        Backend, Context, Encoder, Kernels,
        gpu_types::ActivationType,
        kernel::{
            Conv1dScanKernel,
            ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode},
        },
    },
    for_each_non_cpu_backend,
};

fn write_buffer<B: Backend>(
    buf: &B::Buffer,
    data: &[f32],
) {
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.cpu_ptr().as_ptr() as *mut f32, data.len());
    }
}

fn read_buffer<B: Backend>(
    buf: &B::Buffer,
    len: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    unsafe {
        std::ptr::copy_nonoverlapping(buf.cpu_ptr().as_ptr() as *const f32, out.as_mut_ptr(), len);
    }
    out
}

fn zero_buffer<B: Backend>(buf: &B::Buffer) {
    unsafe {
        std::ptr::write_bytes(buf.cpu_ptr().as_ptr(), 0, buf.length());
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
                let dt_val = ActivationType::SOFTPLUS.activate(dt_raw);
                let decay_val = (-dt_val).exp();
                let dt_scaled_input = x_val;
                let gate = ActivationType::SILU.activate(z_data[x_idx]);
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

fn run_prefill_kernel_mode<B: Backend>(
    ctx: &B::Context,
    kernel: &SSDPrefillKernels<B>,
    fixture: &SSDPrefillFixture,
    mode: SSDPrefillMode,
) -> (Vec<f32>, Vec<f32>, Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)>) {
    let x_array = ctx.create_array_from(&[fixture.x_data.len()], &fixture.x_data, "");
    let x_array_buf = x_array.buffer();
    let x_array_buf_borrow = x_array_buf.borrow();
    let x_buf = x_array_buf_borrow.deref();

    let dt_array = ctx.create_array_from(&[fixture.dt_data.len()], &fixture.dt_data, "");
    let dt_array_buf = dt_array.buffer();
    let dt_array_buf_borrow = dt_array_buf.borrow();
    let dt_buf = dt_array_buf_borrow.deref();

    let b_array = ctx.create_array_from(&[fixture.b_data.len()], &fixture.b_data, "");
    let b_array_buf = b_array.buffer();
    let b_array_buf_borrow = b_array_buf.borrow();
    let b_buf = b_array_buf_borrow.deref();

    let c_array = ctx.create_array_from(&[fixture.c_data.len()], &fixture.c_data, "");
    let c_array_buf = c_array.buffer();
    let c_array_buf_borrow = c_array_buf.borrow();
    let c_buf = c_array_buf_borrow.deref();

    let d_array = ctx.create_array_from(&[fixture.d_data.len()], &fixture.d_data, "");
    let d_array_buf = d_array.buffer();
    let d_array_buf_borrow = d_array_buf.borrow();
    let d_buf = d_array_buf_borrow.deref();

    let z_array = ctx.create_array_from(&[fixture.z_data.len()], &fixture.z_data, "");
    let z_array_buf = z_array.buffer();
    let z_array_buf_borrow = z_array_buf.borrow();
    let z_buf = z_array_buf_borrow.deref();

    let state_array = ctx.create_array_from(&[fixture.state_init.len()], &fixture.state_init, "");
    let state_array_buf = state_array.buffer();
    let mut state_array_buf_borrow = state_array_buf.borrow_mut();
    let state_buf = state_array_buf_borrow.deref_mut();

    let y = vec![0f32; fixture.total_x];
    let y_array = ctx.create_array_from(&[y.len()], &y, "");
    let y_array_buf = y_array.buffer();
    let mut y_array_buf_borrow = y_array_buf.borrow_mut();
    let y_buf = y_array_buf_borrow.deref_mut();

    let args = SSDPrefillArguments::<B> {
        x: x_buf,
        dt: dt_buf,
        b: b_buf,
        c: c_buf,
        d: d_buf,
        z: z_buf,
        state: state_buf,
        y: y_buf,
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

    let mut encoder = Encoder::new(ctx).unwrap();
    kernel.encode(&mut encoder, args, mode);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let y_vec = read_buffer::<B>(y_buf, fixture.total_x);
    let state_vec = read_buffer::<B>(state_buf, fixture.total_state);
    (y_vec, state_vec, None)
}

fn run_conv_scan_once<B: Backend>(
    ctx: &<B as Backend>::Context,
    kernel: &<<B as Backend>::Kernels as Kernels>::Conv1dScanKernel,
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
    let total_x = suffix_len * channels;
    let _total_w = channels * kernel_size as usize;
    let total_state = channels * tap_count;

    let y_array = if alias_io {
        ctx.create_array_from(&[x_data.len()], x_data, "")
    } else {
        ctx.create_array_zeros(&[total_x], DataType::F32, "")
    };
    let b_out_array = ctx.create_array_zeros(&[total_x], DataType::F32, "");
    let c_out_array = ctx.create_array_zeros(&[total_x], DataType::F32, "");

    let w_array = ctx.create_array_from(&[w_data.len()], w_data, "");
    let b_array = ctx.create_array_from(&[b_data.len()], b_data, "");

    let state_array = ctx.create_array_uninitialized(&[total_state], DataType::F32, "");
    let scratch_array = ctx.create_array_uninitialized(&[total_state], DataType::F32, "");

    {
        let state_buf = state_array.buffer();
        let state_borrow = state_buf.borrow();
        write_buffer::<B>(state_borrow.deref(), state_init);
        if use_scratch && tap_count > 0 {
            let scratch_buf = scratch_array.buffer();
            let scratch_borrow = scratch_buf.borrow();
            zero_buffer::<B>(scratch_borrow.deref());
        }
    }

    let padded_len = tap_count + suffix_len;
    let padded_array = ctx.create_array_uninitialized(&[padded_len * channels], DataType::F32, "");
    {
        let padded_buf = padded_array.buffer();
        let padded_borrow = padded_buf.borrow();
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
        write_buffer::<B>(padded_borrow.deref(), &host);
    }

    let mut encoder = Encoder::new(ctx).unwrap();
    {
        let padded_buf = padded_array.buffer();
        let padded_borrow = padded_buf.borrow();
        let w_buf = w_array.buffer();
        let w_borrow = w_buf.borrow();
        let b_buf = b_array.buffer();
        let b_borrow = b_buf.borrow();
        let y_buf = y_array.buffer();
        let mut y_borrow = y_buf.borrow_mut();
        let b_out_buf = b_out_array.buffer();
        let mut b_out_borrow = b_out_buf.borrow_mut();
        let c_out_buf = c_out_array.buffer();
        let mut c_out_borrow = c_out_buf.borrow_mut();
        let state_buf = state_array.buffer();
        let mut state_borrow = state_buf.borrow_mut();

        kernel.encode(
            padded_borrow.deref(),
            w_borrow.deref(),
            Some(b_borrow.deref()),
            y_borrow.deref_mut(),
            b_out_borrow.deref_mut(),
            c_out_borrow.deref_mut(),
            state_borrow.deref_mut(),
            suffix_len as u32,
            kernel_size as u32,
            channels as u32,
            tap_count as u32,
            channels as u32,
            channels as u32,
            0u32,
            ActivationType::SILU,
            &mut encoder,
        );
    }

    if use_scratch && tap_count > 0 {
        let bytes = channels * tap_count * size_of::<f32>();
        if bytes > 0 {
            let scratch_buf = scratch_array.buffer();
            let scratch_borrow = scratch_buf.borrow();
            let state_buf = state_array.buffer();
            let mut state_borrow = state_buf.borrow_mut();
            encoder.encode_copy(scratch_borrow.deref(), 0..bytes, state_borrow.deref_mut(), 0..bytes);
        }
    }

    encoder.end_encoding().submit().wait_until_completed().unwrap();

    let y_buf = y_array.buffer();
    let y_borrow = y_buf.borrow();
    let state_buf = state_array.buffer();
    let state_borrow = state_buf.borrow();
    let y_vec = read_buffer::<B>(y_borrow.deref(), total_x);
    let state_vec = read_buffer::<B>(state_borrow.deref(), total_state);
    (y_vec, state_vec)
}

fn assert_deterministic_for_mode<B: Backend>(mode: SSDPrefillMode) {
    let Some(ctx) = <B as Backend>::Context::new().ok() else {
        eprintln!("Skipping SSD prefill determinism test: no Metal device");
        return;
    };
    let kernel = SSDPrefillKernels::<B>::new(&ctx, DataType::F32).unwrap();
    let fixture = SSDPrefillFixture::new();

    let (y_a, state_a, _) = run_prefill_kernel_mode::<B>(&ctx, &kernel, &fixture, mode);
    let (y_b, state_b, _) = run_prefill_kernel_mode::<B>(&ctx, &kernel, &fixture, mode);

    assert_eq!(y_a, y_b, "Prefill outputs differ in {:?} mode", mode);
    assert_eq!(state_a, state_b, "Prefill states differ in {:?} mode", mode);
}

fn assert_matches_cpu_reference<B: Backend>(mode: SSDPrefillMode) {
    let Some(ctx) = <B as Backend>::Context::new().ok() else {
        eprintln!("Skipping SSD prefill reference test: no Metal device");
        return;
    };
    let kernel = SSDPrefillKernels::<B>::new(&ctx, DataType::F32).unwrap();
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

    let (y_gpu, state_gpu, _) = run_prefill_kernel_mode::<B>(&ctx, &kernel, &fixture, mode);

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

fn conv1d_scan_deterministic_internal<B: Backend>() {
    let Some(ctx) = <B as Backend>::Context::new().ok() else {
        eprintln!("Skipping conv1d scan determinism test: no Metal device");
        return;
    };
    let kernel = <<B as Backend>::Kernels as Kernels>::Conv1dScanKernel::new(&ctx, DataType::F32, true).unwrap();

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
        let first = run_conv_scan_once::<B>(
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
        let second = run_conv_scan_once::<B>(
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

#[test]
fn ssd_prefill_sequential_is_deterministic() {
    for_each_non_cpu_backend!(|B| {
        assert_deterministic_for_mode::<B>(SSDPrefillMode::Sequential);
    });
}

#[test]
fn ssd_prefill_single_pass_is_deterministic() {
    for_each_non_cpu_backend!(|B| {
        assert_deterministic_for_mode::<B>(SSDPrefillMode::SinglePass);
    });
}

#[test]
fn ssd_prefill_sequential_matches_cpu_reference() {
    for_each_non_cpu_backend!(|B| {
        assert_matches_cpu_reference::<B>(SSDPrefillMode::Sequential);
    });
}

#[test]
fn ssd_prefill_single_pass_matches_cpu_reference() {
    for_each_non_cpu_backend!(|B| {
        assert_matches_cpu_reference::<B>(SSDPrefillMode::SinglePass);
    });
}

#[test]
fn conv1d_scan_is_deterministic() {
    for_each_non_cpu_backend!(|B| {
        conv1d_scan_deterministic_internal::<B>();
    });
}
