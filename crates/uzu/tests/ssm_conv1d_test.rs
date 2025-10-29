#![cfg(any(target_os = "macos"))]

use half::bf16;
use metal::MTLResourceOptions;
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::ssm::{
        Conv1dForwardArguments, Conv1dForwardKernel,
        Conv1dSwishForwardArguments, Conv1dSwishForwardKernel,
        Conv1dUpdateArguments, Conv1dUpdateKernel,
    },
};

fn create_context() -> Option<MTLContext> {
    let device = metal::Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn conv1d_forward_ref_bf16(
    x: &[bf16],
    w: &[bf16],
    b: &[bf16],
    bsz: usize,
    d: usize,
    l: usize,
    k: usize,
) -> Vec<bf16> {
    let mut y = vec![bf16::from_f32(0.0); bsz * d * l];
    for bb in 0..bsz {
        for ch in 0..d {
            for t in 0..l {
                let y_off = bb * d * l + ch * l + t;
                if t + k - 1 < l {
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += x[bb * d * l + ch * l + t + i].to_f32()
                            * w[ch * k + i].to_f32();
                    }
                    acc += b[ch].to_f32();
                    y[y_off] = bf16::from_f32(acc);
                } else {
                    y[y_off] = bf16::from_f32(0.0);
                }
            }
        }
    }
    y
}

fn conv1d_update_ref_bf16(
    x: &[bf16],
    w: &[bf16],
    b: &[bf16],
    state: &[bf16],
    bsz: usize,
    d: usize,
    k: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let mut y = vec![bf16::from_f32(0.0); bsz * d];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * d * (k - 1)];
    for bb in 0..bsz {
        for ch in 0..d {
            let x_idx = bb * d + ch;
            let w_start = ch * k;
            let state_start = bb * d * (k - 1) + ch * (k - 1);
            let mut acc = 0.0f32;
            for i in 0..(k - 1) {
                acc +=
                    state[state_start + i].to_f32() * w[w_start + i].to_f32();
            }
            acc += x[x_idx].to_f32() * w[w_start + k - 1].to_f32();
            acc += b[ch].to_f32();
            y[x_idx] = bf16::from_f32(acc);
            for i in 0..(k - 2) {
                next_state[state_start + i] = state[state_start + i + 1];
            }
            next_state[state_start + (k - 2)] = x[x_idx];
        }
    }
    (y, next_state)
}

#[test]
fn conv1d_forward_bf16() {
    let Some(ctx) = create_context() else {
        eprintln!("Skipping: no Metal device");
        return;
    };

    let bsz = 2usize;
    let d = 3usize;
    let l = 16usize;
    let k = 3usize;

    let mut x_f32 = vec![0.0f32; bsz * d * l];
    for i in 0..x_f32.len() {
        x_f32[i] = ((i % 7) as f32) * 0.1 - 0.2;
    }
    let mut w_f32 = vec![0.0f32; d * k];
    for i in 0..w_f32.len() {
        w_f32[i] = ((i % 5) as f32) * 0.05;
    }
    let mut b_f32 = vec![0.0f32; d];
    for i in 0..d {
        b_f32[i] = (i as f32) * 0.01;
    }

    let x: Vec<bf16> = x_f32.iter().copied().map(bf16::from_f32).collect();
    let w: Vec<bf16> = w_f32.iter().copied().map(bf16::from_f32).collect();
    let b: Vec<bf16> = b_f32.iter().copied().map(bf16::from_f32).collect();

    let expected = conv1d_forward_ref_bf16(&x, &w, &b, bsz, d, l, k);
    let expected_f32: Vec<f32> =
        expected.into_iter().map(|v| v.to_f32()).collect();

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w_buf = ctx.device.new_buffer_with_data(
        w.as_ptr() as *const _,
        (w.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = ctx.device.new_buffer(
        (bsz * d * l * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel = Conv1dForwardKernel::new(&ctx, KernelDataType::BFloat16)
        .expect("kernel");

    let command_buffer_ref = ctx.command_queue.new_command_buffer();
    let command_buffer = command_buffer_ref.to_owned();
    let compute_encoder = command_buffer.new_compute_command_encoder();

    kernel
        .encode(
            &compute_encoder,
            Conv1dForwardArguments {
                x: &x_buf,
                w: &w_buf,
                b: &b_buf,
                y: &y_buf,
                x_strides: [d * l, l, 1],
                kernel_size: k as i32,
                batch_size: bsz,
                channels: d,
                seq_len: l,
            },
        )
        .unwrap();

    compute_encoder.end_encoding();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let out_ptr = y_buf.contents() as *const bf16;
    let out = unsafe { std::slice::from_raw_parts(out_ptr, bsz * d * l) };
    let out_f32: Vec<f32> = out.iter().map(|&v| v.to_f32()).collect();

    let tol = 1e-2;
    for (i, (a, b)) in out_f32.iter().zip(expected_f32.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(diff < tol, "conv1d_forward mismatch at {}: {} vs {}", i, a, b);
    }
}

#[test]
fn conv1d_update_bf16() {
    let Some(ctx) = create_context() else {
        eprintln!("Skipping: no Metal device");
        return;
    };
    let bsz = 2usize;
    let d = 4usize;
    let k = 5usize;
    let mut x_f32 = vec![0.0f32; bsz * d];
    for i in 0..x_f32.len() {
        x_f32[i] = (i as f32) * 0.01 - 0.2;
    }
    let mut w_f32 = vec![0.0f32; d * k];
    for i in 0..w_f32.len() {
        w_f32[i] = ((i % 7) as f32) * 0.03;
    }
    let mut b_f32 = vec![0.0f32; d];
    for i in 0..d {
        b_f32[i] = (i as f32) * 0.02;
    }
    let mut state_f32 = vec![0.0f32; bsz * d * (k - 1)];
    for i in 0..state_f32.len() {
        state_f32[i] = ((i % 11) as f32) * 0.01 - 0.05;
    }

    let x: Vec<bf16> = x_f32.iter().copied().map(bf16::from_f32).collect();
    let w: Vec<bf16> = w_f32.iter().copied().map(bf16::from_f32).collect();
    let b: Vec<bf16> = b_f32.iter().copied().map(bf16::from_f32).collect();
    let state: Vec<bf16> =
        state_f32.iter().copied().map(bf16::from_f32).collect();

    let (y_exp, ns_exp) = conv1d_update_ref_bf16(&x, &w, &b, &state, bsz, d, k);
    let y_exp_f32: Vec<f32> = y_exp.iter().map(|&v| v.to_f32()).collect();
    let ns_exp_f32: Vec<f32> = ns_exp.iter().map(|&v| v.to_f32()).collect();

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w_buf = ctx.device.new_buffer_with_data(
        w.as_ptr() as *const _,
        (w.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let s_buf = ctx.device.new_buffer_with_data(
        state.as_ptr() as *const _,
        (state.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = ctx.device.new_buffer(
        (bsz * d * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let ns_buf = ctx.device.new_buffer(
        (bsz * d * (k - 1) * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel =
        Conv1dUpdateKernel::new(&ctx, KernelDataType::BFloat16).unwrap();
    let cb_ref = ctx.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            &enc,
            Conv1dUpdateArguments {
                x: &x_buf,
                w: &w_buf,
                b: &b_buf,
                state: &s_buf,
                y: &y_buf,
                next_state: &ns_buf,
                kernel_size: k as i32,
                x_strides: [d, 1],
                state_strides: [d * (k - 1), k - 1, 1],
                batch_size: bsz,
                channels: d,
            },
        )
        .unwrap();
    enc.end_encoding();
    cb_ref.commit();
    cb_ref.wait_until_completed();

    let y_ptr = y_buf.contents() as *const bf16;
    let y_out = unsafe { std::slice::from_raw_parts(y_ptr, bsz * d) };
    let y_out_f32: Vec<f32> = y_out.iter().map(|&v| v.to_f32()).collect();

    let ns_ptr = ns_buf.contents() as *const bf16;
    let ns_out =
        unsafe { std::slice::from_raw_parts(ns_ptr, bsz * d * (k - 1)) };
    let ns_out_f32: Vec<f32> = ns_out.iter().map(|&v| v.to_f32()).collect();

    let tol = 1e-2;
    for (i, (a, b)) in y_out_f32.iter().zip(y_exp_f32.iter()).enumerate() {
        assert!((a - b).abs() < tol, "y mismatch {} {} {}", i, a, b);
    }
    for (i, (a, b)) in ns_out_f32.iter().zip(ns_exp_f32.iter()).enumerate() {
        assert!((a - b).abs() < tol, "ns mismatch {} {} {}", i, a, b);
    }
}

#[test]
fn conv1d_swish_forward_bf16() {
    let Some(ctx) = create_context() else {
        eprintln!("Skipping: no Metal device");
        return;
    };
    let bsz = 1usize;
    let d = 2usize;
    let l = 8usize;
    let k = 3usize;
    let x: Vec<bf16> = (0..bsz * d * l)
        .map(|i| bf16::from_f32(((i % 5) as f32) * 0.2 - 0.3))
        .collect();
    let w: Vec<bf16> =
        (0..d * k).map(|i| bf16::from_f32(((i % 3) as f32) * 0.1)).collect();
    let b: Vec<bf16> =
        (0..d).map(|i| bf16::from_f32((i as f32) * 0.05)).collect();

    // CPU reference (apply conv then SILU per element)
    let mut exp = conv1d_forward_ref_bf16(&x, &w, &b, bsz, d, l, k);
    // Apply SILU to expected where valid positions
    for val in exp.iter_mut() {
        let v = val.to_f32();
        let sig = 1.0 / (1.0 + (-v.abs()).exp());
        let out = if v < 0.0 {
            (1.0 - sig) * v
        } else {
            sig * v
        };
        *val = bf16::from_f32(out);
    }
    let exp_f32: Vec<f32> = exp.iter().map(|&v| v.to_f32()).collect();

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let w_buf = ctx.device.new_buffer_with_data(
        w.as_ptr() as *const _,
        (w.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let y_buf = ctx.device.new_buffer(
        (bsz * d * l * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let kernel =
        Conv1dSwishForwardKernel::new(&ctx, KernelDataType::BFloat16).unwrap();
    let cb_ref = ctx.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            &enc,
            Conv1dSwishForwardArguments {
                x: &x_buf,
                w: &w_buf,
                b: &b_buf,
                y: &y_buf,
                x_strides: [d * l, l, 1],
                kernel_size: k as i32,
                batch_size: bsz,
                channels: d,
                seq_len: l,
            },
        )
        .unwrap();
    enc.end_encoding();
    cb_ref.commit();
    cb_ref.wait_until_completed();

    let out_ptr = y_buf.contents() as *const bf16;
    let out = unsafe { std::slice::from_raw_parts(out_ptr, bsz * d * l) };
    let out_f32: Vec<f32> = out.iter().map(|&v| v.to_f32()).collect();
    let tol = 1e-2;
    for (i, (a, b)) in out_f32.iter().zip(exp_f32.iter()).enumerate() {
        assert!((a - b).abs() < tol, "mismatch {} {} {}", i, a, b);
    }
}
