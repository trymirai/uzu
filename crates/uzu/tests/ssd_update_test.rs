#![cfg(any(target_os = "macos", target_os = "ios"))]

use metal::Device;
use mpsgraph::CommandBuffer;
use uzu::{
    Array, DataType, DeviceContext,
    backends::metal::{KernelDataType, MTLContext},
};

use uzu::backends::metal::kernel::{SSDUpdateNoZArguments, SSDUpdateNoZKernel};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    match MTLContext::new(device, command_queue) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!(
                "Skipping SSD tests: failed to create Metal context: {:?}",
                e
            );
            None
        },
    }
}

fn ssd_update_cpu_no_z(
    x: &[f32],     // (b,h,dh)
    dt: &[f32],    // (b,h)
    decay: &[f32], // (b,h)
    b: &[f32],     // (b,g,n)
    c: &[f32],     // (b,g,n)
    d: &[f32],     // (h)
    state: &[f32], // (b,h,dh,n)
    bsz: usize,
    h: usize,
    dh: usize,
    g: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    let group_size = h / g;
    let mut y = vec![0.0f32; bsz * h * dh];
    let mut next_state = vec![0.0f32; bsz * h * dh * n];

    for b_i in 0..bsz {
        for h_i in 0..h {
            let g_i = h_i / group_size;
            let this_d = d[h_i];
            let this_dt = dt[b_i * h + h_i];
            let this_decay = decay[b_i * h + h_i];

            for dh_i in 0..dh {
                let x_idx = b_i * (h * dh) + h_i * dh + dh_i;
                let this_x = x[x_idx];

                let mut acc = 0.0f32;
                for s in 0..n {
                    let cb_idx = b_i * (g * n) + g_i * n + s;
                    let state_idx =
                        b_i * (h * dh * n) + h_i * (dh * n) + dh_i * n + s;
                    let new_state = state[state_idx] * this_decay
                        + b[cb_idx] * this_dt * this_x;
                    next_state[state_idx] = new_state;
                    acc += new_state * c[cb_idx];
                }
                y[x_idx] = acc + this_d * this_x;
            }
        }
    }

    (y, next_state)
}

#[test]
fn ssd_update_no_z_matches_cpu() {
    let Some(context) = create_test_context() else {
        return;
    };

    // Dimensions
    let bsz = 2usize;
    let h = 4usize;
    let dh = 8usize;
    let g = 2usize; // must divide h
    let n = 5usize;

    let act_dtype = DataType::F32;
    let kdt: KernelDataType = act_dtype.into();

    // Allocate device arrays
    let mut x = context.array(&[bsz, h, dh], act_dtype);
    let mut dt = context.array(&[bsz, h], act_dtype);
    let mut decay = context.array(&[bsz, h], act_dtype);
    let mut bb = context.array(&[bsz, g, n], act_dtype);
    let mut cc = context.array(&[bsz, g, n], act_dtype);
    let mut dd = context.array(&[h], act_dtype);
    let mut state = context.array(&[bsz, h, dh, n], act_dtype);
    let mut y = context.array(&[bsz, h, dh], act_dtype);
    let mut next_state = context.array(&[bsz, h, dh, n], act_dtype);

    // Initialize with deterministic values
    {
        let xs = x.as_slice_mut::<f32>().unwrap();
        for (i, v) in xs.iter_mut().enumerate() {
            *v = (i as f32 % 13.0) / 13.0;
        }
    }
    {
        let dts = dt.as_slice_mut::<f32>().unwrap();
        for (i, v) in dts.iter_mut().enumerate() {
            *v = 0.1 + (i as f32 % 7.0) / 10.0;
        }
    }
    {
        let dec = decay.as_slice_mut::<f32>().unwrap();
        for (i, v) in dec.iter_mut().enumerate() {
            *v = 0.8 + (i as f32 % 5.0) / 50.0;
        }
    }
    {
        let bs = bb.as_slice_mut::<f32>().unwrap();
        for (i, v) in bs.iter_mut().enumerate() {
            *v = ((i * 7 % 17) as f32) / 50.0;
        }
    }
    {
        let cs = cc.as_slice_mut::<f32>().unwrap();
        for (i, v) in cs.iter_mut().enumerate() {
            *v = ((i * 5 % 19) as f32) / 40.0;
        }
    }
    {
        let ds = dd.as_slice_mut::<f32>().unwrap();
        for (i, v) in ds.iter_mut().enumerate() {
            *v = (i as f32 % 3.0) / 7.0;
        }
    }
    {
        let st = state.as_slice_mut::<f32>().unwrap();
        for (i, v) in st.iter_mut().enumerate() {
            *v = (i as f32 % 11.0) / 17.0;
        }
    }

    // CPU reference
    let (y_cpu, next_state_cpu) = ssd_update_cpu_no_z(
        x.as_slice::<f32>().unwrap(),
        dt.as_slice::<f32>().unwrap(),
        decay.as_slice::<f32>().unwrap(),
        bb.as_slice::<f32>().unwrap(),
        cc.as_slice::<f32>().unwrap(),
        dd.as_slice::<f32>().unwrap(),
        state.as_slice::<f32>().unwrap(),
        bsz,
        h,
        dh,
        g,
        n,
    );

    // GPU invoke
    let kernel = SSDUpdateNoZKernel::new(&context, kdt).expect("kernel new");
    let command_buffer =
        CommandBuffer::from_command_queue(&context.command_queue);
    let root = command_buffer.root_command_buffer().to_owned();
    let compute = root.new_compute_command_encoder();

    let x_buf = unsafe { x.mtl_buffer() };
    let dt_buf = unsafe { dt.mtl_buffer() };
    let decay_buf = unsafe { decay.mtl_buffer() };
    let b_buf = unsafe { bb.mtl_buffer() };
    let c_buf = unsafe { cc.mtl_buffer() };
    let d_buf = unsafe { dd.mtl_buffer() };
    let state_buf = unsafe { state.mtl_buffer() };
    let y_buf = unsafe { y.mtl_buffer() };
    let next_state_buf = unsafe { next_state.mtl_buffer() };

    // Compute strides/constants like the kernel expects
    let x_strides = [h * dh, dh, 1usize];
    let dt_strides = [h, 1usize];
    let cb_strides = [g * n, n, 1usize];
    let state_strides = [h * dh * n, dh * n, n, 1usize];
    let group_size = (h / g) as i32;
    let state_size = n as i32;

    kernel
        .encode(
            &compute,
            SSDUpdateNoZArguments {
                x: &x_buf,
                dt: &dt_buf,
                decay: &decay_buf,
                b: &b_buf,
                c: &c_buf,
                d: &d_buf,
                state: &state_buf,
                y: &y_buf,
                next_state: &next_state_buf,
                group_size,
                state_size,
                x_strides,
                dt_strides,
                cb_strides,
                state_strides,
                b_size: bsz,
                h_size: h,
                dh_size: dh,
            },
        )
        .expect("encode no_z");

    compute.end_encoding();
    command_buffer.commit();
    root.wait_until_completed();

    // Validate
    let y_gpu = y.as_slice::<f32>().unwrap().to_vec();
    let next_state_gpu = next_state.as_slice::<f32>().unwrap().to_vec();

    let tol = 1e-3f32;
    for (i, (a, b)) in y_gpu.iter().zip(y_cpu.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= tol,
            "y mismatch at {}: gpu={} cpu={} diff={}",
            i,
            a,
            b,
            diff
        );
    }
    for (i, (a, b)) in
        next_state_gpu.iter().zip(next_state_cpu.iter()).enumerate()
    {
        let diff = (a - b).abs();
        assert!(
            diff <= tol,
            "state mismatch at {}: gpu={} cpu={} diff={}",
            i,
            a,
            b,
            diff
        );
    }
}
