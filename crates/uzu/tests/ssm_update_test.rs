#![cfg(any(target_os = "macos"))]

use half::bf16;
use metal::MTLResourceOptions;
use uzu::backends::metal::{
    KernelDataType, MTLContext,
    kernel::ssm::{
        SSMUpdateArguments, SSMUpdateKernel,
        SSDUpdateArguments, SSDUpdateKernel,
        SSDUpdateNoZArguments, SSDUpdateNoZKernel,
    },
};
fn ssm_update_ref_bf16(
    x: &[bf16], dt: &[bf16], a: &[bf16], b: &[bf16], c: &[bf16], d: &[bf16], z: &[bf16], state: &[bf16],
    bsz: usize, h: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let n: usize = 16;
    let mut y = vec![bf16::from_f32(0.0); bsz * h];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * h * n];
    for bb in 0..bsz { for ch in 0..h {
        let idx = bb * h + ch;
        let vx = x[idx].to_f32();
        let sig = 1.0 / (1.0 + (-vx.abs()).exp());
        let this_x = if vx < 0.0 { (1.0 - sig) * vx } else { sig * vx };
        let vz = z[idx].to_f32();
        let sigz = 1.0 / (1.0 + (-vz.abs()).exp());
        let this_z = if vz < 0.0 { (1.0 - sigz) * vz } else { sigz * vz };
        let delta = (1.0 + dt[idx].to_f32().exp()).ln();
        let cb_offset = bb * n;
        let state_offset = idx * n;
        let mut acc = 0.0f32;
        for i in 0..n {
            let new_s = state[state_offset + i].to_f32() * (a[i].to_f32() * delta).exp()
                + b[cb_offset + i].to_f32() * delta * this_x;
            next_state[state_offset + i] = bf16::from_f32(new_s);
            acc += new_s * c[cb_offset + i].to_f32();
        }
        acc += d[ch].to_f32() * this_x;
        acc *= this_z;
        y[idx] = bf16::from_f32(acc);
    }}
    (y, next_state)
}

fn ssd_update_no_z_ref_bf16(
    x: &[bf16], dt: &[bf16], decay: &[bf16], b: &[bf16], c: &[bf16], d: &[bf16], state: &[bf16],
    bsz: usize, h: usize, dh: usize, g: usize, n: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let group_size = h / g;
    let mut y = vec![bf16::from_f32(0.0); bsz * h * dh];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * h * dh * n];
    for bb in 0..bsz { for hh in 0..h { for ddh in 0..dh {
        let x_idx = bb * h * dh + hh * dh + ddh;
        let dt_idx = bb * h + hh;
        let cb_idx0 = bb * g * n + (hh / group_size) * n;
        let state_idx0 = (bb * h * dh + hh * dh + ddh) * n;
        let this_x = x[x_idx].to_f32();
        let this_dt = dt[dt_idx].to_f32();
        let this_decay = decay[dt_idx].to_f32();
        let this_d = d[hh].to_f32();
        let mut acc = 0.0f32;
        for i in 0..n {
            let s_old = state[state_idx0 + i].to_f32();
            let new_s = s_old * this_decay + b[cb_idx0 + i].to_f32() * this_dt * this_x;
            next_state[state_idx0 + i] = bf16::from_f32(new_s);
            acc += new_s * c[cb_idx0 + i].to_f32();
        }
        acc += this_d * this_x;
        y[x_idx] = bf16::from_f32(acc);
    }}}
    (y, next_state)
}

fn ssd_update_ref_bf16(
    x: &[bf16], dt: &[bf16], decay: &[bf16], b: &[bf16], c: &[bf16], d: &[bf16], z: &[bf16], state: &[bf16],
    bsz: usize, h: usize, dh: usize, g: usize, n: usize,
) -> (Vec<bf16>, Vec<bf16>) {
    let group_size = h / g;
    let mut y = vec![bf16::from_f32(0.0); bsz * h * dh];
    let mut next_state = vec![bf16::from_f32(0.0); bsz * h * dh * n];
    for bb in 0..bsz { for hh in 0..h { for ddh in 0..dh {
        let x_idx = bb * h * dh + hh * dh + ddh;
        let dt_idx = bb * h + hh;
        let cb_idx0 = bb * g * n + (hh / group_size) * n;
        let state_idx0 = (bb * h * dh + hh * dh + ddh) * n;
        let this_x = x[x_idx].to_f32();
        let this_dt = dt[dt_idx].to_f32();
        let this_decay = decay[dt_idx].to_f32();
        let this_d = d[hh].to_f32();
        let vz = z[x_idx].to_f32();
        let sigz = 1.0 / (1.0 + (-vz.abs()).exp());
        let this_z = if vz < 0.0 { (1.0 - sigz) * vz } else { sigz * vz };
        let mut acc = 0.0f32;
        for i in 0..n {
            let s_old = state[state_idx0 + i].to_f32();
            let new_s = s_old * this_decay + b[cb_idx0 + i].to_f32() * this_dt * this_x;
            next_state[state_idx0 + i] = bf16::from_f32(new_s);
            acc += new_s * c[cb_idx0 + i].to_f32();
        }
        acc += this_d * this_x;
        acc *= this_z;
        y[x_idx] = bf16::from_f32(acc);
    }}}
    (y, next_state)
}

fn create_context() -> Option<MTLContext> {
    let device = metal::Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

#[test]
fn ssm_update_bf16() {
    let Some(ctx) = create_context() else { eprintln!("Skipping: no Metal device"); return; };
    let bsz=2usize; let h=4usize; let n=16usize;

    let x: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(((i%7) as f32)*0.1 - 0.2)).collect();
    let dt: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(((i%5) as f32)*0.03)).collect();
    let a: Vec<bf16> = (0..n).map(|i| bf16::from_f32(((i%11) as f32)*0.02 - 0.1)).collect();
    let b: Vec<bf16> = (0..bsz*n).map(|i| bf16::from_f32(((i%13) as f32)*0.01)).collect();
    let c: Vec<bf16> = (0..bsz*n).map(|i| bf16::from_f32(((i%17) as f32)*0.015)).collect();
    let d: Vec<bf16> = (0..h).map(|i| bf16::from_f32(((i%3) as f32)*0.05)).collect();
    let z: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(((i%7) as f32)*0.1 - 0.1)).collect();
    let state: Vec<bf16> = (0..bsz*h*n).map(|i| bf16::from_f32(((i%23) as f32)*0.01 - 0.05)).collect();

    let (y_exp, ns_exp) = ssm_update_ref_bf16(&x, &dt, &a, &b, &c, &d, &z, &state, bsz, h);
    let y_exp_f32: Vec<f32> = y_exp.iter().map(|&v| v.to_f32()).collect();
    let ns_exp_f32: Vec<f32> = ns_exp.iter().map(|&v| v.to_f32()).collect();

    let x_buf = ctx.device.new_buffer_with_data(x.as_ptr() as *const _, (x.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let dt_buf = ctx.device.new_buffer_with_data(dt.as_ptr() as *const _, (dt.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let a_buf = ctx.device.new_buffer_with_data(a.as_ptr() as *const _, (a.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let b_buf = ctx.device.new_buffer_with_data(b.as_ptr() as *const _, (b.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let c_buf = ctx.device.new_buffer_with_data(c.as_ptr() as *const _, (c.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let d_buf = ctx.device.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let z_buf = ctx.device.new_buffer_with_data(z.as_ptr() as *const _, (z.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let s_buf = ctx.device.new_buffer_with_data(state.as_ptr() as *const _, (state.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let y_buf = ctx.device.new_buffer((bsz*h*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let ns_buf = ctx.device.new_buffer((bsz*h*n*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);

    let kernel = SSMUpdateKernel::new(&ctx, KernelDataType::BFloat16).unwrap();
    let cb_ref = ctx.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel.encode(&enc, SSMUpdateArguments {
        x: &x_buf, dt: &dt_buf, a: &a_buf, b: &b_buf, c: &c_buf, d: &d_buf, z: &z_buf, state: &s_buf,
        y: &y_buf, next_state: &ns_buf, batch_size: bsz, channels: h,
    }).unwrap();
    enc.end_encoding(); cb_ref.commit(); cb_ref.wait_until_completed();

    let y_ptr = y_buf.contents() as *const bf16;
    let y_out = unsafe { std::slice::from_raw_parts(y_ptr, bsz*h) };
    let y_out_f32: Vec<f32> = y_out.iter().map(|&v| v.to_f32()).collect();
    let ns_ptr = ns_buf.contents() as *const bf16;
    let ns_out = unsafe { std::slice::from_raw_parts(ns_ptr, bsz*h*n) };
    let ns_out_f32: Vec<f32> = ns_out.iter().map(|&v| v.to_f32()).collect();
    let tol = 2e-2;
    for (i,(a,b)) in y_out_f32.iter().zip(y_exp_f32.iter()).enumerate(){ assert!((a-b).abs()<tol, "y {} {} {}", i,a,b); }
    for (i,(a,b)) in ns_out_f32.iter().zip(ns_exp_f32.iter()).enumerate(){ assert!((a-b).abs()<tol, "ns {} {} {}", i,a,b); }
}

#[test]
fn ssd_update_no_z_bf16() {
    let Some(ctx) = create_context() else { eprintln!("Skipping: no Metal device"); return; };
    let bsz=1usize; let h=4usize; let dh=3usize; let g=2usize; let n=8usize;

    let x: Vec<bf16> = (0..bsz*h*dh).map(|i| bf16::from_f32(((i%7) as f32)*0.1 - 0.2)).collect();
    let dt: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(((i%5) as f32)*0.03)).collect();
    let decay: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(0.8 + ((i%3) as f32)*0.01)).collect();
    let b: Vec<bf16> = (0..bsz*g*n).map(|i| bf16::from_f32(((i%11) as f32)*0.02 - 0.05)).collect();
    let c: Vec<bf16> = (0..bsz*g*n).map(|i| bf16::from_f32(((i%13) as f32)*0.015)).collect();
    let d: Vec<bf16> = (0..h).map(|i| bf16::from_f32(((i%3) as f32)*0.05)).collect();
    let state: Vec<bf16> = (0..bsz*h*dh*n).map(|i| bf16::from_f32(((i%23) as f32)*0.01 - 0.05)).collect();

    let (y_exp, ns_exp) = ssd_update_no_z_ref_bf16(&x, &dt, &decay, &b, &c, &d, &state, bsz, h, dh, g, n);
    let y_exp_f32: Vec<f32> = y_exp.iter().map(|&v| v.to_f32()).collect();
    let ns_exp_f32: Vec<f32> = ns_exp.iter().map(|&v| v.to_f32()).collect();

    let x_strides = [h*dh, dh, 1usize];
    let dt_strides = [h, 1usize];
    let cb_strides = [g*n, n, 1usize];
    let state_strides = [h*dh*n, dh*n, n, 1usize];

    let x_buf = ctx.device.new_buffer_with_data(x.as_ptr() as *const _, (x.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let dt_buf = ctx.device.new_buffer_with_data(dt.as_ptr() as *const _, (dt.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let decay_buf = ctx.device.new_buffer_with_data(decay.as_ptr() as *const _, (decay.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let b_buf = ctx.device.new_buffer_with_data(b.as_ptr() as *const _, (b.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let c_buf = ctx.device.new_buffer_with_data(c.as_ptr() as *const _, (c.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let d_buf = ctx.device.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let state_buf = ctx.device.new_buffer_with_data(state.as_ptr() as *const _, (state.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let y_buf = ctx.device.new_buffer((bsz*h*dh*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let ns_buf = ctx.device.new_buffer((bsz*h*dh*n*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);

    let kernel = SSDUpdateNoZKernel::new(&ctx, KernelDataType::BFloat16).unwrap();
    let cb_ref = ctx.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel.encode(&enc, SSDUpdateNoZArguments {
        x: &x_buf, dt: &dt_buf, decay: &decay_buf, b: &b_buf, c: &c_buf, d: &d_buf, state: &state_buf,
        y: &y_buf, next_state: &ns_buf,
        group_size: (h/g) as i32, state_size: n as i32,
        x_strides, dt_strides, cb_strides, state_strides,
        b_size: bsz, h_size: h, dh_size: dh,
    }).unwrap();
    enc.end_encoding(); cb_ref.commit(); cb_ref.wait_until_completed();

    let y_ptr = y_buf.contents() as *const bf16;
    let y_out = unsafe { std::slice::from_raw_parts(y_ptr, bsz*h*dh) };
    let y_out_f32: Vec<f32> = y_out.iter().map(|&v| v.to_f32()).collect();
    let ns_ptr = ns_buf.contents() as *const bf16;
    let ns_out = unsafe { std::slice::from_raw_parts(ns_ptr, bsz*h*dh*n) };
    let ns_out_f32: Vec<f32> = ns_out.iter().map(|&v| v.to_f32()).collect();
    let tol = 2e-2;
    for (i,(a,b)) in y_out_f32.iter().zip(y_exp_f32.iter()).enumerate(){ assert!((a-b).abs()<tol, "y {} {} {}", i,a,b); }
    for (i,(a,b)) in ns_out_f32.iter().zip(ns_exp_f32.iter()).enumerate(){ assert!((a-b).abs()<tol, "ns {} {} {}", i,a,b); }
}

#[test]
fn ssd_update_with_z_bf16() {
    let Some(ctx) = create_context() else { eprintln!("Skipping: no Metal device"); return; };
    let bsz=1usize; let h=4usize; let dh=3usize; let g=2usize; let n=8usize;

    let x: Vec<bf16> = (0..bsz*h*dh).map(|i| bf16::from_f32(((i%7) as f32)*0.1 - 0.2)).collect();
    let z: Vec<bf16> = (0..bsz*h*dh).map(|i| bf16::from_f32(((i%5) as f32)*0.1 - 0.1)).collect();
    let dt: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(((i%5) as f32)*0.03)).collect();
    let decay: Vec<bf16> = (0..bsz*h).map(|i| bf16::from_f32(0.8 + ((i%3) as f32)*0.01)).collect();
    let b: Vec<bf16> = (0..bsz*g*n).map(|i| bf16::from_f32(((i%11) as f32)*0.02 - 0.05)).collect();
    let c: Vec<bf16> = (0..bsz*g*n).map(|i| bf16::from_f32(((i%13) as f32)*0.015)).collect();
    let d: Vec<bf16> = (0..h).map(|i| bf16::from_f32(((i%3) as f32)*0.05)).collect();
    let state: Vec<bf16> = (0..bsz*h*dh*n).map(|i| bf16::from_f32(((i%23) as f32)*0.01 - 0.05)).collect();

    let (y_exp, ns_exp) = ssd_update_ref_bf16(&x, &dt, &decay, &b, &c, &d, &z, &state, bsz, h, dh, g, n);
    let y_exp_f32: Vec<f32> = y_exp.iter().map(|&v| v.to_f32()).collect();
    let ns_exp_f32: Vec<f32> = ns_exp.iter().map(|&v| v.to_f32()).collect();

    let x_strides = [h*dh, dh, 1usize];
    let dt_strides = [h, 1usize];
    let cb_strides = [g*n, n, 1usize];
    let state_strides = [h*dh*n, dh*n, n, 1usize];

    let x_buf = ctx.device.new_buffer_with_data(x.as_ptr() as *const _, (x.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let dt_buf = ctx.device.new_buffer_with_data(dt.as_ptr() as *const _, (dt.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let decay_buf = ctx.device.new_buffer_with_data(decay.as_ptr() as *const _, (decay.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let b_buf = ctx.device.new_buffer_with_data(b.as_ptr() as *const _, (b.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let c_buf = ctx.device.new_buffer_with_data(c.as_ptr() as *const _, (c.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let d_buf = ctx.device.new_buffer_with_data(d.as_ptr() as *const _, (d.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let z_buf = ctx.device.new_buffer_with_data(z.as_ptr() as *const _, (z.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let state_buf = ctx.device.new_buffer_with_data(state.as_ptr() as *const _, (state.len()*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let y_buf = ctx.device.new_buffer((bsz*h*dh*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);
    let ns_buf = ctx.device.new_buffer((bsz*h*dh*n*std::mem::size_of::<bf16>()) as u64, MTLResourceOptions::StorageModeShared);

    let kernel = SSDUpdateKernel::new(&ctx, KernelDataType::BFloat16).unwrap();
    let cb_ref = ctx.command_queue.new_command_buffer();
    let cb = cb_ref.to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel.encode(&enc, SSDUpdateArguments {
        x: &x_buf, dt: &dt_buf, decay: &decay_buf, b: &b_buf, c: &c_buf, d: &d_buf, z: &z_buf, state: &state_buf,
        y: &y_buf, next_state: &ns_buf,
        group_size: (h/g) as i32, state_size: n as i32,
        x_strides, dt_strides, cb_strides, state_strides,
        b_size: bsz, h_size: h, dh_size: dh,
    }).unwrap();
    enc.end_encoding(); cb_ref.commit(); cb_ref.wait_until_completed();

    let y_ptr = y_buf.contents() as *const bf16;
    let y_out = unsafe { std::slice::from_raw_parts(y_ptr, bsz*h*dh) };
    let y_out_f32: Vec<f32> = y_out.iter().map(|&v| v.to_f32()).collect();
    let ns_ptr = ns_buf.contents() as *const bf16;
    let ns_out = unsafe { std::slice::from_raw_parts(ns_ptr, bsz*h*dh*n) };
    let ns_out_f32: Vec<f32> = ns_out.iter().map(|&v| v.to_f32()).collect();
    let tol = 2e-2;
    for (i,(a,b)) in y_out_f32.iter().zip(y_exp_f32.iter()).enumerate(){ assert!((a-b).abs()<tol, "y {} {} {}", i,a,b); }
    for (i,(a,b)) in ns_out_f32.iter().zip(ns_exp_f32.iter()).enumerate(){ assert!((a-b).abs()<tol, "ns {} {} {}", i,a,b); }
}


