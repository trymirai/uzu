#![cfg(any(target_os = "macos"))]

use bytemuck;
use half::bf16;
use metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLDeviceExt, MTLResourceOptions,
};
use uzu::backends::{
    common::Context,
    metal::{KernelDataType, MTLContext, kernel::dsl::SSDUpdateKernel},
};

#[allow(dead_code)]
fn ssm_update_ref_bf16(
    x: &[bf16],
    dt: &[bf16],
    a: &[bf16],
    b: &[bf16],
    c: &[bf16],
    d: &[bf16],
    z: &[bf16],
    state: &[bf16],
    bsz: usize,
    h: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let n: usize = 16;
    let mut y = vec![bf16::from_f32(0.0); bsz * h];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * h * n];
    for bb in 0..bsz {
        for ch in 0..h {
            let idx = bb * h + ch;
            let vx = x[idx].to_f32();
            let sig = 1.0 / (1.0 + (-vx.abs()).exp());
            let this_x = if vx < 0.0 {
                (1.0 - sig) * vx
            } else {
                sig * vx
            };
            let vz = z[idx].to_f32();
            let sigz = 1.0 / (1.0 + (-vz.abs()).exp());
            let this_z = if vz < 0.0 {
                (1.0 - sigz) * vz
            } else {
                sigz * vz
            };
            let delta = (1.0 + dt[idx].to_f32().exp()).ln();
            let cb_offset = bb * n;
            let state_offset = idx * n;
            let mut acc = 0.0f32;
            for i in 0..n {
                let new_s = state[state_offset + i].to_f32()
                    * (a[i].to_f32() * delta).exp()
                    + b[cb_offset + i].to_f32() * delta * this_x;
                next_state[state_offset + i] = bf16::from_f32(new_s);
                acc += new_s * c[cb_offset + i].to_f32();
            }
            acc += d[ch].to_f32() * this_x;
            acc *= this_z;
            y[idx] = bf16::from_f32(acc);
        }
    }
    (y, next_state)
}

fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[allow(dead_code)]
fn ssd_update_no_z_ref_bf16(
    x: &[bf16],
    dt_raw: &[bf16],
    b: &[bf16],
    c: &[bf16],
    d: &[bf16],
    state: &[bf16],
    bsz: usize,
    h: usize,
    dh: usize,
    g: usize,
    n: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let group_size = h / g;
    let mut y = vec![bf16::from_f32(0.0); bsz * h * dh];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * h * dh * n];
    for bb in 0..bsz {
        for hh in 0..h {
            for ddh in 0..dh {
                let x_idx = bb * h * dh + hh * dh + ddh;
                let dt_idx = bb * h + hh;
                let cb_idx0 = bb * g * n + (hh / group_size) * n;
                let state_idx0 = (bb * h * dh + hh * dh + ddh) * n;
                let this_x = x[x_idx].to_f32();
                let dt_raw_val = dt_raw[dt_idx].to_f32();
                let this_dt = softplus_f32(dt_raw_val);
                let this_decay = (-this_dt).exp();
                let this_d = d[hh].to_f32();
                let dt_safe = this_dt.max(1e-6);
                let normalized_x = this_x / dt_safe;
                let dt_scaled_input = normalized_x * this_dt;
                let mut acc = 0.0f32;
                for i in 0..n {
                    let s_old = state[state_idx0 + i].to_f32();
                    let new_s = s_old * this_decay
                        + b[cb_idx0 + i].to_f32() * dt_scaled_input;
                    next_state[state_idx0 + i] = bf16::from_f32(new_s);
                    acc += new_s * c[cb_idx0 + i].to_f32();
                }
                acc += this_d * this_x;
                y[x_idx] = bf16::from_f32(acc);
            }
        }
    }
    (y, next_state)
}

fn ssd_update_ref_bf16(
    x: &[bf16],
    dt_raw: &[bf16],
    b: &[bf16],
    c: &[bf16],
    d: &[bf16],
    z: &[bf16],
    state: &[bf16],
    bsz: usize,
    h: usize,
    dh: usize,
    g: usize,
    n: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let group_size = h / g;
    let mut y = vec![bf16::from_f32(0.0); bsz * h * dh];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * h * dh * n];
    for bb in 0..bsz {
        for hh in 0..h {
            for ddh in 0..dh {
                let x_idx = bb * h * dh + hh * dh + ddh;
                let dt_idx = bb * h + hh;
                let cb_idx0 = bb * g * n + (hh / group_size) * n;
                let state_idx0 = (bb * h * dh + hh * dh + ddh) * n;
                let this_x = x[x_idx].to_f32();
                let dt_raw_val = dt_raw[dt_idx].to_f32();
                let this_dt = softplus_f32(dt_raw_val);
                let this_decay = (-this_dt).exp();
                let this_d = d[hh].to_f32();
                let dt_safe = this_dt.max(1e-6);
                let normalized_x = this_x / dt_safe;
                let dt_scaled_input = normalized_x * this_dt;
                let vz = z[x_idx].to_f32();
                let sigz = 1.0 / (1.0 + (-vz.abs()).exp());
                let this_z = if vz < 0.0 {
                    (1.0 - sigz) * vz
                } else {
                    sigz * vz
                };
                let mut acc = 0.0f32;
                for i in 0..n {
                    let s_old = state[state_idx0 + i].to_f32();
                    let new_s = s_old * this_decay
                        + b[cb_idx0 + i].to_f32() * dt_scaled_input;
                    next_state[state_idx0 + i] = bf16::from_f32(new_s);
                    acc += new_s * c[cb_idx0 + i].to_f32();
                }
                acc += this_d * this_x;
                acc *= this_z;
                y[x_idx] = bf16::from_f32(acc);
            }
        }
    }
    (y, next_state)
}

#[test]
fn ssd_update_with_z_bf16() {
    let Some(ctx) = MTLContext::new().ok() else {
        eprintln!("Skipping: no Metal device");
        return;
    };
    let bsz = 1usize;
    let h = 4usize;
    let dh = 3usize;
    let g = 2usize;
    let n = 8usize;

    let x: Vec<bf16> = (0..bsz * h * dh)
        .map(|i| bf16::from_f32(((i % 7) as f32) * 0.1 - 0.2))
        .collect();
    let z: Vec<bf16> = (0..bsz * h * dh)
        .map(|i| bf16::from_f32(((i % 5) as f32) * 0.1 - 0.1))
        .collect();
    let dt: Vec<bf16> = (0..bsz * h)
        .map(|i| bf16::from_f32(((i % 5) as f32) * 0.3 - 1.0))
        .collect();
    let b: Vec<bf16> = (0..bsz * g * n)
        .map(|i| bf16::from_f32(((i % 11) as f32) * 0.02 - 0.05))
        .collect();
    let c: Vec<bf16> = (0..bsz * g * n)
        .map(|i| bf16::from_f32(((i % 13) as f32) * 0.015))
        .collect();
    let d: Vec<bf16> =
        (0..h).map(|i| bf16::from_f32(((i % 3) as f32) * 0.05)).collect();
    let state: Vec<bf16> = (0..bsz * h * dh * n)
        .map(|i| bf16::from_f32(((i % 23) as f32) * 0.01 - 0.05))
        .collect();

    let (y_exp, ns_exp) =
        ssd_update_ref_bf16(&x, &dt, &b, &c, &d, &z, &state, bsz, h, dh, g, n);
    let y_exp_f32: Vec<f32> = y_exp.iter().map(|&v| v.to_f32()).collect();
    let ns_exp_f32: Vec<f32> = ns_exp.iter().map(|&v| v.to_f32()).collect();

    let x_strides = [(h * dh) as u32, dh as u32, 1u32];
    let dt_strides = [h as u32, 1u32];
    let cb_strides = [(g * n) as u32, n as u32, 1u32];
    let state_strides = [(h * dh * n) as u32, (dh * n) as u32, n as u32, 1u32];

    let x_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&x),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let dt_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&dt),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let b_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&b),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let c_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&c),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let d_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&d),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let z_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&z),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let state_buf = ctx
        .device
        .new_buffer_with_data(
            bytemuck::cast_slice(&state),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let y_buf = ctx
        .device
        .new_buffer(
            bsz * h * dh * std::mem::size_of::<bf16>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");
    let ns_buf = ctx
        .device
        .new_buffer(
            bsz * h * dh * n * std::mem::size_of::<bf16>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");

    let kernel = SSDUpdateKernel::new(&ctx, KernelDataType::BFloat16).unwrap();
    let cb_ref = ctx
        .command_queue
        .command_buffer()
        .expect("Failed to create command buffer");
    let cb = cb_ref.to_owned();
    let enc = cb
        .new_compute_command_encoder()
        .expect("Failed to create compute encoder");
    kernel.encode(
        &x_buf,
        &dt_buf,
        &b_buf,
        &c_buf,
        &d_buf,
        &z_buf,
        &state_buf,
        &y_buf,
        &ns_buf,
        (h / g) as u32,
        n as u32,
        x_strides.as_slice(),
        dt_strides.as_slice(),
        cb_strides.as_slice(),
        state_strides.as_slice(),
        bsz as u32,
        h as u32,
        dh as u32,
        &enc,
    );
    enc.end_encoding();
    cb_ref.commit();
    cb_ref.wait_until_completed();

    let y_ptr = y_buf.contents().as_ptr() as *const bf16;
    let y_out = unsafe { std::slice::from_raw_parts(y_ptr, bsz * h * dh) };
    let y_out_f32: Vec<f32> = y_out.iter().map(|&v| v.to_f32()).collect();
    let ns_ptr = ns_buf.contents().as_ptr() as *const bf16;
    let ns_out =
        unsafe { std::slice::from_raw_parts(ns_ptr, bsz * h * dh * n) };
    let ns_out_f32: Vec<f32> = ns_out.iter().map(|&v| v.to_f32()).collect();
    let tol = 2e-2;
    for (i, (a, b)) in y_out_f32.iter().zip(y_exp_f32.iter()).enumerate() {
        assert!((a - b).abs() < tol, "y {} {} {}", i, a, b);
    }
    for (i, (a, b)) in ns_out_f32.iter().zip(ns_exp_f32.iter()).enumerate() {
        assert!((a - b).abs() < tol, "ns {} {} {}", i, a, b);
    }
}
