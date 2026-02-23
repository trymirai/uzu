#![cfg(target_os = "macos")]

use bytemuck;
use half::bf16;
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use ndarray::Array2;
use uzu::{
    DataType,
    backends::{
        common::{
            Context,
            kernel::matmul::{MatmulArguments, MatmulKernel},
        },
        metal::{Metal, MetalContext, kernel::matmul::choose_dispatch_descriptor},
    },
};

fn run_metal_matmul_i8_i8_i32(
    ctx: &MetalContext,
    a_data: &[i8],
    b_data: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<i32> {
    let a_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(a_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let b_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(b_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let d_buf = ctx
        .device
        .new_buffer(m * n * core::mem::size_of::<i32>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut kernel = MatmulKernel::<Metal>::new_mixed(DataType::I8, DataType::I8, DataType::I32).expect("kernel");

    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer").to_owned();
    let enc = cb.new_compute_command_encoder().expect("Failed to create compute encoder");
    let mut arguments = MatmulArguments {
        a: &a_buf,
        a_offset: 0,
        b: &b_buf,
        d: &d_buf,
        bias: None,
        batch: m as i32,
        input_dim: k as i32,
        output_dim: n as i32,
        lda: k as i32,
        ldb: k as i32,
        ldd: n as i32,
        batch_count: 1,
        transpose_b: true,
    };
    MatmulKernel::<Metal>::apply_batch_collapse(&mut arguments);
    let descriptor = choose_dispatch_descriptor(ctx, DataType::I8, &arguments).expect("dispatch descriptor");
    let encode_result = kernel.encode_with_descriptor(ctx, arguments, &descriptor, &enc);
    enc.end_encoding();
    encode_result.expect("encode");
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = d_buf.contents().as_ptr() as *const i32;
        std::slice::from_raw_parts(ptr, m * n).to_vec()
    }
}

fn run_metal_matmul_i8_bf16_bf16(
    ctx: &MetalContext,
    a_data: &[i8],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<bf16> {
    let a_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(a_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let b_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(b_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let d_buf = ctx
        .device
        .new_buffer(m * n * core::mem::size_of::<bf16>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut kernel = MatmulKernel::<Metal>::new_mixed(DataType::I8, DataType::BF16, DataType::BF16).expect("kernel");

    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer").to_owned();
    let enc = cb.new_compute_command_encoder().expect("Failed to create compute encoder");
    let mut arguments = MatmulArguments {
        a: &a_buf,
        a_offset: 0,
        b: &b_buf,
        d: &d_buf,
        bias: None,
        batch: m as i32,
        input_dim: k as i32,
        output_dim: n as i32,
        lda: k as i32,
        ldb: k as i32,
        ldd: n as i32,
        batch_count: 1,
        transpose_b: true,
    };
    MatmulKernel::<Metal>::apply_batch_collapse(&mut arguments);
    let descriptor = choose_dispatch_descriptor(ctx, DataType::I8, &arguments).expect("dispatch descriptor");
    let encode_result = kernel.encode_with_descriptor(ctx, arguments, &descriptor, &enc);
    enc.end_encoding();
    encode_result.expect("encode");
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = d_buf.contents().as_ptr() as *const bf16;
        std::slice::from_raw_parts(ptr, m * n).to_vec()
    }
}

fn run_metal_matmul_bf16_bf16_bf16(
    ctx: &MetalContext,
    a_data: &[bf16],
    b_data: &[bf16],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<bf16> {
    let a_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(a_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let b_buf = ctx
        .device
        .new_buffer_with_data(bytemuck::cast_slice(b_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");
    let d_buf = ctx
        .device
        .new_buffer(m * n * core::mem::size_of::<bf16>(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let mut kernel = MatmulKernel::<Metal>::new_mixed(DataType::BF16, DataType::BF16, DataType::BF16).expect("kernel");

    let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer").to_owned();
    let enc = cb.new_compute_command_encoder().expect("Failed to create compute encoder");
    let mut arguments = MatmulArguments {
        a: &a_buf,
        a_offset: 0,
        b: &b_buf,
        d: &d_buf,
        bias: None,
        batch: m as i32,
        input_dim: k as i32,
        output_dim: n as i32,
        lda: k as i32,
        ldb: k as i32,
        ldd: n as i32,
        batch_count: 1,
        transpose_b: true,
    };
    MatmulKernel::<Metal>::apply_batch_collapse(&mut arguments);
    let descriptor = choose_dispatch_descriptor(ctx, DataType::BF16, &arguments).expect("dispatch descriptor");
    let encode_result = kernel.encode_with_descriptor(ctx, arguments, &descriptor, &enc);
    enc.end_encoding();
    encode_result.expect("encode");
    cb.commit();
    cb.wait_until_completed();

    unsafe {
        let ptr = d_buf.contents().as_ptr() as *const bf16;
        std::slice::from_raw_parts(ptr, m * n).to_vec()
    }
}

fn run_ndarray_matmul_i8_i8_i32(a_data: &[i8], b_data: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let a_f64: Vec<f64> = a_data.iter().map(|&x| x as f64).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|&x| x as f64).collect();

    let a_arr = Array2::from_shape_vec((m, k), a_f64).expect("A shape");
    let b_arr = Array2::from_shape_vec((n, k), b_f64).expect("B shape");
    let result = a_arr.dot(&b_arr.t());

    result.iter().map(|&x| x as i32).collect()
}

fn run_ndarray_matmul_i8_bf16_bf16(a_data: &[i8], b_data: &[bf16], m: usize, k: usize, n: usize) -> Vec<bf16> {
    let a_f64: Vec<f64> = a_data.iter().map(|&x| x as f64).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|x| x.to_f64()).collect();

    let a_arr = Array2::from_shape_vec((m, k), a_f64).expect("A shape");
    let b_arr = Array2::from_shape_vec((n, k), b_f64).expect("B shape");
    let result = a_arr.dot(&b_arr.t());

    result.iter().map(|&x| bf16::from_f64(x)).collect()
}

fn run_ndarray_matmul_bf16_bf16_bf16(a_data: &[bf16], b_data: &[bf16], m: usize, k: usize, n: usize) -> Vec<bf16> {
    let a_f64: Vec<f64> = a_data.iter().map(|x| x.to_f64()).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|x| x.to_f64()).collect();

    let a_arr = Array2::from_shape_vec((m, k), a_f64).expect("A shape");
    let b_arr = Array2::from_shape_vec((n, k), b_f64).expect("B shape");
    let result = a_arr.dot(&b_arr.t());

    result.iter().map(|&x| bf16::from_f64(x)).collect()
}

const INT_MATMUL_CORRECTNESS_FULL_ENV: &str = "UZU_INT_MATMUL_CORRECTNESS_FULL";

const QUICK_BATCH_SIZES: [usize; 2] = [1, 16];
const FULL_BATCH_SIZES: [usize; 12] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];

const QUICK_MODEL_SHAPES: [(usize, usize); 8] =
    [(896, 896), (896, 4864), (1024, 4096), (1152, 6912), (1536, 8960), (2048, 8192), (4096, 14336), (5120, 17408)];

const FULL_MODEL_SHAPES: [(usize, usize); 10] = [
    (896, 896),
    (1024, 4096),
    (2048, 2048),
    (2048, 8192),
    (3072, 8192),
    (4096, 4096),
    (4096, 14336),
    (5120, 5120),
    (5120, 17408),
    (128, 128),
];

fn run_full_case_matrix() -> bool {
    std::env::var(INT_MATMUL_CORRECTNESS_FULL_ENV)
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes")
        })
        .unwrap_or(false)
}

#[test]
#[ignore]
fn int_matmul_correctness_i8_i8_i32() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("No Metal device available, skipping test");
        return;
    };

    if !ctx.is_mpp_available() {
        eprintln!("MPP not available on this device, skipping test");
        return;
    }

    let full_case_matrix = run_full_case_matrix();
    let (batch_sizes, model_shapes): (&[usize], &[(usize, usize)]) = if full_case_matrix {
        (&FULL_BATCH_SIZES, &FULL_MODEL_SHAPES)
    } else {
        (&QUICK_BATCH_SIZES, &QUICK_MODEL_SHAPES)
    };

    let mut passed = 0;
    let mut failed = Vec::new();

    for &(k, n) in model_shapes {
        for &m in batch_sizes {
            let a: Vec<i8> = (0..(m * k)).map(|i| ((i % 13) as i8) - 6).collect();
            let b: Vec<i8> = (0..(n * k)).map(|i| ((i % 17) as i8) - 8).collect();

            let metal_result = run_metal_matmul_i8_i8_i32(&ctx, &a, &b, m, k, n);
            let reference = run_ndarray_matmul_i8_i8_i32(&a, &b, m, k, n);

            let max_diff = metal_result
                .iter()
                .zip(reference.iter())
                .map(|(&m_val, &r_val)| (m_val - r_val).unsigned_abs())
                .max()
                .unwrap_or(0);

            if max_diff == 0 {
                passed += 1;
                eprintln!("  i8*i8->i32 m={m} k={k} n={n} PASS");
            } else {
                failed.push(format!("i8*i8->i32 m={m} k={k} n={n} max_diff={max_diff}"));
                eprintln!("  i8*i8->i32 m={m} k={k} n={n} FAIL max_diff={max_diff}");
            }
        }
    }

    eprintln!("\ni8*i8->i32: {passed}/{} passed", passed + failed.len());
    if !failed.is_empty() {
        panic!("{} tests failed:\n{}", failed.len(), failed.join("\n"));
    }
}

#[test]
#[ignore]
fn int_matmul_correctness_i8_bf16_bf16() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("No Metal device available, skipping test");
        return;
    };

    if !ctx.is_mpp_available() {
        eprintln!("MPP not available on this device, skipping test");
        return;
    }

    let full_case_matrix = run_full_case_matrix();
    let (batch_sizes, model_shapes): (&[usize], &[(usize, usize)]) = if full_case_matrix {
        (&FULL_BATCH_SIZES, &FULL_MODEL_SHAPES)
    } else {
        (&QUICK_BATCH_SIZES, &QUICK_MODEL_SHAPES)
    };

    let base_tolerance = 0.05;
    let mut passed = 0;
    let mut failed = Vec::new();

    for &(k, n) in model_shapes {
        for &m in batch_sizes {
            let a: Vec<i8> = (0..(m * k)).map(|i| ((i % 13) as i8) - 6).collect();
            let b: Vec<bf16> = (0..(n * k)).map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15)).collect();

            let metal_result = run_metal_matmul_i8_bf16_bf16(&ctx, &a, &b, m, k, n);
            let reference = run_ndarray_matmul_i8_bf16_bf16(&a, &b, m, k, n);

            let tolerance = base_tolerance * (k as f32 / 1024.0).sqrt();

            let max_diff = metal_result
                .iter()
                .zip(reference.iter())
                .map(|(&m_val, &r_val)| (m_val.to_f32() - r_val.to_f32()).abs())
                .fold(0.0f32, f32::max);

            if max_diff <= tolerance {
                passed += 1;
                eprintln!("  i8*bf16->bf16 m={m} k={k} n={n} PASS (max_diff={max_diff:.6})");
            } else {
                failed.push(format!("i8*bf16->bf16 m={m} k={k} n={n} max_diff={max_diff:.6} > tol={tolerance:.6}"));
                eprintln!("  i8*bf16->bf16 m={m} k={k} n={n} FAIL max_diff={max_diff:.6} > tol={tolerance:.6}");
            }
        }
    }

    eprintln!("\ni8*bf16->bf16: {passed}/{} passed", passed + failed.len());
    if !failed.is_empty() {
        panic!("{} tests failed:\n{}", failed.len(), failed.join("\n"));
    }
}

#[test]
#[ignore]
fn int_matmul_correctness_bf16_bf16_bf16() {
    let Some(ctx) = MetalContext::new().ok() else {
        eprintln!("No Metal device available, skipping test");
        return;
    };

    if !ctx.is_mpp_available() {
        eprintln!("MPP not available on this device, skipping test");
        return;
    }

    let full_case_matrix = run_full_case_matrix();
    let (batch_sizes, model_shapes): (&[usize], &[(usize, usize)]) = if full_case_matrix {
        (&FULL_BATCH_SIZES, &FULL_MODEL_SHAPES)
    } else {
        (&QUICK_BATCH_SIZES, &QUICK_MODEL_SHAPES)
    };

    let base_tolerance = 0.01;
    let mut passed = 0;
    let mut failed = Vec::new();

    for &(k, n) in model_shapes {
        for &m in batch_sizes {
            let a: Vec<bf16> = (0..(m * k)).map(|i| bf16::from_f32(((i % 13) as f32) * 0.01 - 0.06)).collect();
            let b: Vec<bf16> = (0..(n * k)).map(|i| bf16::from_f32(((i % 17) as f32) * 0.02 - 0.15)).collect();

            let metal_result = run_metal_matmul_bf16_bf16_bf16(&ctx, &a, &b, m, k, n);
            let reference = run_ndarray_matmul_bf16_bf16_bf16(&a, &b, m, k, n);

            let tolerance = base_tolerance * (k as f32 / 1024.0).sqrt() * (1.0 + (m as f32).log2() * 0.02);

            let max_diff = metal_result
                .iter()
                .zip(reference.iter())
                .map(|(&m_val, &r_val)| (m_val.to_f32() - r_val.to_f32()).abs())
                .fold(0.0f32, f32::max);

            if max_diff <= tolerance {
                passed += 1;
                eprintln!("  bf16*bf16->bf16 m={m} k={k} n={n} PASS (max_diff={max_diff:.6})");
            } else {
                failed.push(format!(
                    "bf16*bf16->bf16 m={m} k={k} n={n} max_diff={max_diff:.6} > tol={tolerance:.6}"
                ));
                eprintln!("  bf16*bf16->bf16 m={m} k={k} n={n} FAIL max_diff={max_diff:.6} > tol={tolerance:.6}");
            }
        }
    }

    eprintln!("\nbf16*bf16->bf16: {passed}/{} passed", passed + failed.len());
    if !failed.is_empty() {
        panic!("{} tests failed:\n{}", failed.len(), failed.join("\n"));
    }
}
