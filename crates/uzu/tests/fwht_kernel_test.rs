#![cfg(any(target_os = "macos"))]

use bytemuck;
use half::{bf16, f16};
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use uzu::{
    DataType,
    backends::{
        common::{Backend, Context, Kernels, kernel::{FwhtKernel, FwhtBlockKernel}},
        metal::Metal,
    },
};

// --- CPU reference implementation ---

fn cpu_fwht_inplace(data: &mut [f32], scale: f32) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FWHT requires power-of-2 length");
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let a = data[j];
                let b = data[j + h];
                data[j] = a + b;
                data[j + h] = a - b;
            }
        }
        h <<= 1;
    }
    for x in data.iter_mut() {
        *x *= scale;
    }
}

fn cpu_fwht_block_inplace(data: &mut [f32], block_size: usize, scale: f32) {
    let n = data.len();
    assert!(n % block_size == 0, "Row length must be divisible by block_size");
    for chunk in data.chunks_exact_mut(block_size) {
        cpu_fwht_inplace(chunk, scale);
    }
}

// --- Helper traits for multi-type testing ---

trait TestFloat: Copy + Clone + std::fmt::Debug + PartialEq + bytemuck::NoUninit {
    fn from_f32(val: f32) -> Self;
    fn to_f32(self) -> f32;
    fn size_of() -> usize;
    fn data_type() -> DataType;
}

impl TestFloat for f32 {
    fn from_f32(val: f32) -> Self { val }
    fn to_f32(self) -> f32 { self }
    fn size_of() -> usize { std::mem::size_of::<f32>() }
    fn data_type() -> DataType { DataType::F32 }
}

impl TestFloat for f16 {
    fn from_f32(val: f32) -> Self { f16::from_f32(val) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn size_of() -> usize { std::mem::size_of::<f16>() }
    fn data_type() -> DataType { DataType::F16 }
}

impl TestFloat for bf16 {
    fn from_f32(val: f32) -> Self { bf16::from_f32(val) }
    fn to_f32(self) -> f32 { self.to_f32() }
    fn size_of() -> usize { std::mem::size_of::<bf16>() }
    fn data_type() -> DataType { DataType::BF16 }
}

// --- Full-vector FWHT tests ---

fn test_fwht_full<T: TestFloat>(n: usize, batch_size: usize) {
    let mtl_context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping FWHT test: {}", e);
            return;
        },
    };

    let scale = 1.0f32 / (n as f32).sqrt();
    let total = batch_size * n;

    // Generate test data: sine wave pattern
    let input_f32: Vec<f32> = (0..total)
        .map(|i| 2.0 * f32::sin(i as f32 * 0.037) + 0.5 * f32::cos(i as f32 * 0.013))
        .collect();

    // Compute CPU reference
    let mut expected_f32 = input_f32.clone();
    for row in expected_f32.chunks_exact_mut(n) {
        cpu_fwht_inplace(row, scale);
    }

    // Convert to target type for GPU
    let input_typed: Vec<T> = input_f32.iter().map(|&x| T::from_f32(x)).collect();

    // Create Metal buffer
    let buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&input_typed), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    // Create kernel
    let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtKernel::new(
        &mtl_context,
        T::data_type(),
        n as i32,
    )
    .expect("Failed to create FWHT kernel");

    // Encode and execute
    let command_buffer_ref = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
    let command_buffer = command_buffer_ref.to_owned();
    let mut encoder = command_buffer.new_compute_command_encoder().expect("Failed to create encoder");

    kernel.encode(&buffer, batch_size as u32, scale, &mut encoder);

    encoder.end_encoding();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    // Read back results
    let output_ptr = buffer.contents().as_ptr() as *const T;
    let output_typed = unsafe { std::slice::from_raw_parts(output_ptr, total) };
    let output_f32: Vec<f32> = output_typed.iter().map(|x| x.to_f32()).collect();

    // Compare with CPU reference.
    // bf16 has only 7 mantissa bits; after log2(N) butterfly stages the
    // cumulative rounding error grows significantly for large N.
    let tolerance = match T::data_type() {
        DataType::BF16 => 0.2,
        DataType::F16 => 0.1,
        _ => 1e-3,
    };

    let mut max_diff = 0.0f32;
    for (i, (&expected, &actual)) in expected_f32.iter().zip(output_f32.iter()).enumerate() {
        let diff = (expected - actual).abs();
        max_diff = max_diff.max(diff);
        let relative = if expected.abs() > 1e-6 { diff / expected.abs() } else { diff };
        assert!(
            relative < tolerance || diff < tolerance,
            "FWHT mismatch at index {} (N={}, batch={}, type={:?}): expected {}, got {}, diff {}, rel {}",
            i, n, batch_size, T::data_type(), expected, actual, diff, relative
        );
    }

    println!(
        "  FWHT Full N={}, batch={}, type={:?} passed (max_diff={:.6})",
        n, batch_size, T::data_type(), max_diff
    );
}

// --- Block-wise FWHT tests ---

fn test_fwht_block<T: TestFloat>(n: usize, block_size: usize, batch_size: usize) {
    let mtl_context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping FWHT block test: {}", e);
            return;
        },
    };

    let scale = 1.0f32 / (block_size as f32).sqrt();
    let total = batch_size * n;

    // Generate test data
    let input_f32: Vec<f32> = (0..total)
        .map(|i| 2.0 * f32::sin(i as f32 * 0.037) + 0.5 * f32::cos(i as f32 * 0.013))
        .collect();

    // Compute CPU reference (block-wise)
    let mut expected_f32 = input_f32.clone();
    for row in expected_f32.chunks_exact_mut(n) {
        cpu_fwht_block_inplace(row, block_size, scale);
    }

    // Convert to target type
    let input_typed: Vec<T> = input_f32.iter().map(|&x| T::from_f32(x)).collect();

    let buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&input_typed), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtBlockKernel::new(
        &mtl_context,
        T::data_type(),
        block_size as i32,
    )
    .expect("Failed to create FwhtBlock kernel");

    let command_buffer_ref = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
    let command_buffer = command_buffer_ref.to_owned();
    let mut encoder = command_buffer.new_compute_command_encoder().expect("Failed to create encoder");

    kernel.encode(&buffer, batch_size as u32, n as u32, scale, &mut encoder);

    encoder.end_encoding();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let output_ptr = buffer.contents().as_ptr() as *const T;
    let output_typed = unsafe { std::slice::from_raw_parts(output_ptr, total) };
    let output_f32: Vec<f32> = output_typed.iter().map(|x| x.to_f32()).collect();

    let tolerance = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.1
    } else {
        1e-3
    };

    for (i, (&expected, &actual)) in expected_f32.iter().zip(output_f32.iter()).enumerate() {
        let diff = (expected - actual).abs();
        let relative = if expected.abs() > 1e-6 { diff / expected.abs() } else { diff };
        assert!(
            relative < tolerance || diff < tolerance,
            "FwhtBlock mismatch at index {} (n={}, block={}, batch={}, type={:?}): expected {}, got {}, diff {}",
            i, n, block_size, batch_size, T::data_type(), expected, actual, diff
        );
    }

    println!(
        "  FWHT Block n={}, block_size={}, batch={}, type={:?} passed",
        n, block_size, batch_size, T::data_type()
    );
}

// --- Involution test: applying FWHT twice should give back the original ---

fn test_fwht_involution(n: usize) {
    let mtl_context = match <Metal as Backend>::Context::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping FWHT involution test: {}", e);
            return;
        },
    };

    let scale = 1.0f32 / (n as f32).sqrt();

    let input_f32: Vec<f32> = (0..n)
        .map(|i| 3.0 * f32::sin(i as f32 * 0.1) + 1.0)
        .collect();

    let buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&input_f32), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtKernel::new(
        &mtl_context,
        DataType::F32,
        n as i32,
    )
    .expect("Failed to create FWHT kernel");

    // Apply FWHT twice
    let command_buffer_ref = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
    let command_buffer = command_buffer_ref.to_owned();
    let mut encoder = command_buffer.new_compute_command_encoder().expect("Failed to create encoder");

    kernel.encode(&buffer, 1u32, scale, &mut encoder);
    kernel.encode(&buffer, 1u32, scale, &mut encoder);

    encoder.end_encoding();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    // Should be back to original
    let output_ptr = buffer.contents().as_ptr() as *const f32;
    let output_f32 = unsafe { std::slice::from_raw_parts(output_ptr, n) };

    for (i, (&original, &roundtrip)) in input_f32.iter().zip(output_f32.iter()).enumerate() {
        let diff = (original - roundtrip).abs();
        assert!(
            diff < 1e-3,
            "Involution mismatch at index {} (N={}): original {}, roundtrip {}, diff {}",
            i, n, original, roundtrip, diff
        );
    }

    println!("  FWHT involution N={} passed", n);
}

// --- Test entries ---

#[test]
fn test_fwht_full_f32() {
    println!("Testing FWHT Full (f32):");
    for &n in &[64, 128, 256, 512, 1024, 2048, 4096] {
        test_fwht_full::<f32>(n, 1);
        test_fwht_full::<f32>(n, 4);
    }
}

#[test]
fn test_fwht_full_f16() {
    println!("Testing FWHT Full (f16):");
    for &n in &[64, 256, 1024, 2048] {
        test_fwht_full::<f16>(n, 1);
        test_fwht_full::<f16>(n, 4);
    }
}

#[test]
fn test_fwht_full_bf16() {
    println!("Testing FWHT Full (bf16):");
    for &n in &[64, 256, 1024, 2048] {
        test_fwht_full::<bf16>(n, 1);
        test_fwht_full::<bf16>(n, 4);
    }
}

#[test]
fn test_fwht_block_simd_f32() {
    println!("Testing FWHT Block simd path (block_size=32, f32):");
    test_fwht_block::<f32>(2048, 32, 1);
    test_fwht_block::<f32>(2048, 32, 4);
    test_fwht_block::<f32>(1024, 32, 1);
}

#[test]
fn test_fwht_block_threadgroup_f32() {
    println!("Testing FWHT Block threadgroup path (f32):");
    test_fwht_block::<f32>(2048, 64, 1);
    test_fwht_block::<f32>(2048, 128, 1);
    test_fwht_block::<f32>(2048, 256, 1);
}

#[test]
fn test_fwht_block_f16() {
    println!("Testing FWHT Block (f16):");
    test_fwht_block::<f16>(2048, 32, 1);
    test_fwht_block::<f16>(2048, 64, 1);
}

#[test]
fn test_fwht_involution_various_sizes() {
    println!("Testing FWHT involution (H * H = I):");
    for &n in &[64, 128, 256, 512, 1024, 2048, 4096] {
        test_fwht_involution(n);
    }
}
