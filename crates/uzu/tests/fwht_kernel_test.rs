#![cfg(any(target_os = "macos"))]

use std::rc::Rc;

use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use thiserror::Error;
use uzu::{
    DataType,
    backends::{
        common::{Backend, Context, Kernels, kernel::{FwhtKernel, FwhtSimdBlockKernel}},
        metal::Metal,
    },
};

type ComputeEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
type MetalContext = <Metal as Backend>::Context;

#[derive(Debug, Error)]
enum FwhtTestError {
    #[error("Metal context creation failed: {0}")]
    ContextCreation(String),
    #[error("Buffer creation failed")]
    BufferCreation,
    #[error("Kernel creation failed: {0}")]
    KernelCreation(String),
    #[error("Command buffer creation failed")]
    CommandBufferCreation,
    #[error("Encoder creation failed")]
    EncoderCreation,
    #[error("{label}: mismatch at {index}: expected {expected}, got {actual}, diff {difference}")]
    Mismatch { label: String, index: usize, expected: f32, actual: f32, difference: f32 },
}

fn cpu_fwht_inplace(data: &mut [f32], scale: f32) {
    let length = data.len();
    assert!(length.is_power_of_two());
    let mut stride = 1;
    while stride < length {
        for block_start in (0..length).step_by(stride * 2) {
            for index in block_start..block_start + stride {
                let sum = data[index] + data[index + stride];
                let difference = data[index] - data[index + stride];
                data[index] = sum;
                data[index + stride] = difference;
            }
        }
        stride <<= 1;
    }
    for element in data.iter_mut() {
        *element *= scale;
    }
}

fn cpu_fwht_block_inplace(data: &mut [f32], block_size: usize, scale: f32) {
    assert!(data.len() % block_size == 0);
    for block in data.chunks_exact_mut(block_size) {
        cpu_fwht_inplace(block, scale);
    }
}

fn make_test_data(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| 2.0 * f32::sin(index as f32 * 0.037) + 0.5 * f32::cos(index as f32 * 0.013))
        .collect()
}

fn create_context() -> Result<Rc<MetalContext>, FwhtTestError> {
    MetalContext::new().map_err(|error| FwhtTestError::ContextCreation(error.to_string()))
}

fn run_on_gpu(context: &MetalContext, encode: impl FnOnce(&mut ComputeEncoder)) -> Result<(), FwhtTestError> {
    let command_buffer = context.command_queue.command_buffer().ok_or(FwhtTestError::CommandBufferCreation)?;
    let mut encoder = command_buffer.new_compute_command_encoder().ok_or(FwhtTestError::EncoderCreation)?;
    encode(&mut encoder);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}

fn read_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, length: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, length) }.to_vec()
}

fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32, label: &str) -> Result<(), FwhtTestError> {
    let mut max_difference = 0.0f32;
    for (index, (&expected_value, &actual_value)) in expected.iter().zip(actual.iter()).enumerate() {
        let difference = (expected_value - actual_value).abs();
        max_difference = max_difference.max(difference);
        if difference >= tolerance {
            return Err(FwhtTestError::Mismatch {
                label: label.to_string(),
                index,
                expected: expected_value,
                actual: actual_value,
                difference,
            });
        }
    }
    println!("  {label} passed (max_diff={max_difference:.6})");
    Ok(())
}

#[test]
fn test_fwht_full() -> Result<(), FwhtTestError> {
    let context = create_context()?;
    for &row_length in &[32, 64, 128, 256, 512, 1024, 2048, 4096] {
        for &batch_size in &[1, 4] {
            let total_elements = batch_size * row_length;
            let scale = 1.0 / (row_length as f32).sqrt();
            let input = make_test_data(total_elements);

            let mut expected = input.clone();
            for row in expected.chunks_exact_mut(row_length) {
                cpu_fwht_inplace(row, scale);
            }

            let buffer = context.device
                .new_buffer_with_data(bytemuck::cast_slice(&input), MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(FwhtTestError::BufferCreation)?;

            let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtKernel::new(&context, DataType::F32, row_length as i32)
                .map_err(|error| FwhtTestError::KernelCreation(error.to_string()))?;

            run_on_gpu(&context, |encoder| kernel.encode(&buffer, batch_size as u32, scale, encoder))?;

            assert_close(&expected, &read_buffer(&buffer, total_elements), 1e-3, &format!("Full N={row_length} batch={batch_size}"))?;
        }
    }
    Ok(())
}

#[test]
fn test_fwht_involution() -> Result<(), FwhtTestError> {
    let context = create_context()?;
    for &row_length in &[64, 128, 256, 512, 1024, 2048, 4096] {
        let scale = 1.0 / (row_length as f32).sqrt();
        let input = make_test_data(row_length);

        let buffer = context.device
            .new_buffer_with_data(bytemuck::cast_slice(&input), MTLResourceOptions::STORAGE_MODE_SHARED)
            .ok_or(FwhtTestError::BufferCreation)?;

        let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtKernel::new(&context, DataType::F32, row_length as i32)
            .map_err(|error| FwhtTestError::KernelCreation(error.to_string()))?;

        run_on_gpu(&context, |encoder| {
            kernel.encode(&buffer, 1u32, scale, encoder);
            kernel.encode(&buffer, 1u32, scale, encoder);
        })?;

        assert_close(&input, &read_buffer(&buffer, row_length), 1e-3, &format!("Involution N={row_length}"))?;
    }
    Ok(())
}

#[test]
fn test_fwht_simd_shuffle() -> Result<(), FwhtTestError> {
    let context = create_context()?;
    let preload_size = 128;
    let simd_group_size = 32;
    for &row_length in &[1024, 2048] {
        for &batch_size in &[1, 4] {
            let total_elements = batch_size * row_length;
            let scale = 1.0 / (simd_group_size as f32).sqrt();
            let input = make_test_data(total_elements);

            let mut expected = input.clone();
            for row in expected.chunks_exact_mut(row_length) {
                cpu_fwht_block_inplace(row, simd_group_size, scale);
            }

            let buffer = context.device
                .new_buffer_with_data(bytemuck::cast_slice(&input), MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(FwhtTestError::BufferCreation)?;

            let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtSimdBlockKernel::new(
                &context, DataType::F32, preload_size as i32,
            ).map_err(|error| FwhtTestError::KernelCreation(error.to_string()))?;

            let num_groups = (row_length / preload_size) as u32;
            run_on_gpu(&context, |encoder| kernel.encode(&buffer, batch_size as u32 * num_groups, scale, encoder))?;

            assert_close(&expected, &read_buffer(&buffer, total_elements), 1e-3, &format!("SimdShuffle n={row_length} batch={batch_size}"))?;
        }
    }
    Ok(())
}

#[test]
fn test_fwht_block() -> Result<(), FwhtTestError> {
    let context = create_context()?;
    for &block_size in &[32, 64, 128] {
        for &row_length in &[1024, 2048] {
            let scale = 1.0 / (block_size as f32).sqrt();
            let input = make_test_data(row_length);

            let mut expected = input.clone();
            cpu_fwht_block_inplace(&mut expected, block_size, scale);

            let buffer = context.device
                .new_buffer_with_data(bytemuck::cast_slice(&input), MTLResourceOptions::STORAGE_MODE_SHARED)
                .ok_or(FwhtTestError::BufferCreation)?;

            let kernel = <<Metal as Backend>::Kernels as Kernels>::FwhtKernel::new(&context, DataType::F32, block_size as i32)
                .map_err(|error| FwhtTestError::KernelCreation(error.to_string()))?;

            let num_blocks = (row_length / block_size) as u32;
            run_on_gpu(&context, |encoder| kernel.encode(&buffer, num_blocks, scale, encoder))?;

            assert_close(&expected, &read_buffer(&buffer, row_length), 1e-3, &format!("Block block={block_size} n={row_length}"))?;
        }
    }
    Ok(())
}
