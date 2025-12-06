use half::{bf16, f16};
use metal::{Device, MTLResourceOptions};
use uzu::{
    DataType,
    backends::metal::{
        MTLContext,
        kernel::{MatmulArguments, MatmulKernel},
    },
};

fn create_test_context() -> Option<MTLContext> {
    let device = Device::system_default()?;
    let command_queue = device.new_command_queue();
    MTLContext::new(device, command_queue).ok()
}

fn matmul_ref(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[test]
fn gemm_f16_64x64() {
    let Some(ctx) = create_test_context() else {
        return;
    }; // Skip if no Metal

    let m = 64;
    let k = 64;
    let n = 64;
    let a_f32: Vec<f32> =
        (0..(m * k)).map(|i| ((i % 13) as f32) * 0.01).collect();
    let b_f32: Vec<f32> =
        (0..(k * n)).map(|i| ((i % 17) as f32) * 0.02 - 0.1).collect();
    let exp = matmul_ref(&a_f32, &b_f32, m, k, n);

    let a: Vec<f16> = a_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let b: Vec<f16> = b_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<f16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut kernel = MatmulKernel::new(&ctx, DataType::F16, false, false)
        .expect("kernel new");

    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            &ctx,
            &enc,
            MatmulArguments {
                a: &a_buf,
                b: &b_buf,
                d: &d_buf,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: n as i32,
                ldd: n as i32,
                batch_count: 1,
            },
        )
        .expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let got: Vec<f32> = unsafe {
        let ptr = d_buf.contents() as *const f16;
        let slice = core::slice::from_raw_parts(ptr, m * n);
        slice.iter().map(|&h| h.to_f32()).collect()
    };

    for (i, (&e, &g)) in exp.iter().zip(got.iter()).enumerate() {
        let diff = (e - g).abs();
        assert!(diff <= 0.1, "idx {i}: exp {e} got {g} diff {diff}");
    }
}

#[test]
fn gemm_bf16_32x48() {
    let Some(ctx) = create_test_context() else {
        return;
    }; // Skip if no Metal
    let m = 32;
    let k = 48;
    let n = 32;
    let a_f32: Vec<f32> =
        (0..(m * k)).map(|i| ((i % 7) as f32) * 0.03).collect();
    let b_f32: Vec<f32> =
        (0..(k * n)).map(|i| ((i % 5) as f32) * 0.04 - 0.2).collect();
    let exp = matmul_ref(&a_f32, &b_f32, m, k, n);

    let a: Vec<bf16> = a_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let b: Vec<bf16> = b_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut kernel = MatmulKernel::new(&ctx, DataType::BF16, false, false)
        .expect("kernel new");

    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            &ctx,
            &enc,
            MatmulArguments {
                a: &a_buf,
                b: &b_buf,
                d: &d_buf,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: n as i32,
                ldd: n as i32,
                batch_count: 1,
            },
        )
        .expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let got: Vec<f32> = unsafe {
        let ptr = d_buf.contents() as *const bf16;
        let slice = core::slice::from_raw_parts(ptr, m * n);
        slice.iter().map(|&h| h.to_f32()).collect()
    };

    for (i, (&e, &g)) in exp.iter().zip(got.iter()).enumerate() {
        let diff = (e - g).abs();
        assert!(diff <= 0.2, "idx {i}: exp {e} got {g} diff {diff}");
    }
}

#[test]
fn gemm_f32_16x20() {
    let Some(ctx) = create_test_context() else {
        return;
    }; // Skip if no Metal
    let m = 16;
    let k = 20;
    let n = 16;
    let a: Vec<f32> = (0..(m * k)).map(|i| ((i % 11) as f32) * 0.05).collect();
    let b: Vec<f32> =
        (0..(k * n)).map(|i| ((i % 9) as f32) * 0.02 - 0.3).collect();
    let exp = matmul_ref(&a, &b, m, k, n);

    let a_buf = ctx.device.new_buffer_with_data(
        a.as_ptr() as *const _,
        (a.len() * core::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let b_buf = ctx.device.new_buffer_with_data(
        b.as_ptr() as *const _,
        (b.len() * core::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let d_buf = ctx.device.new_buffer(
        (m * n * core::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut kernel = MatmulKernel::new(&ctx, DataType::F32, false, false)
        .expect("kernel new");
    let cb = ctx.command_queue.new_command_buffer().to_owned();
    let enc = cb.new_compute_command_encoder();
    kernel
        .encode(
            &ctx,
            &enc,
            MatmulArguments {
                a: &a_buf,
                b: &b_buf,
                d: &d_buf,
                batch: m as i32,
                input_dim: k as i32,
                output_dim: n as i32,
                lda: k as i32,
                ldb: n as i32,
                ldd: n as i32,
                batch_count: 1,
            },
        )
        .expect("encode");
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let got: Vec<f32> = unsafe {
        let ptr = d_buf.contents() as *const f32;
        let slice = core::slice::from_raw_parts(ptr, m * n);
        slice.to_vec()
    };

    for (i, (&e, &g)) in exp.iter().zip(got.iter()).enumerate() {
        let diff = (e - g).abs();
        assert!(diff <= 1e-3, "idx {i}: exp {e} got {g} diff {diff}");
    }
}
