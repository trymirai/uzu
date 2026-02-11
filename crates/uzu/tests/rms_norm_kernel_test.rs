#![cfg(any(target_os = "macos"))]
#![allow(dead_code)]

use bytemuck;
use half::{bf16, f16};
use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDeviceExt, MTLResourceOptions};
use uzu::backends::common::kernel::RMSNormKernel;
use uzu::backends::metal::kernel::dsl::RMSNormMetalKernel;
use uzu::backends::metal::kernel::rms_norm;
use uzu::{
    DataType,
    backends::{
        common::Context,
        metal::{
            MTLContext,
            kernel::rms_norm::{QKNormArguments, QKNormTarget, RMSNormKernelType},
            metal_extensions::CommandBufferTimingExt,
        },
    },
};

// Helper trait to unify different float types for testing
trait TestFloat: Copy + Clone + std::fmt::Debug + PartialEq + bytemuck::NoUninit {
    fn from_f32(val: f32) -> Self;
    fn to_f32(self) -> f32;
    fn size_of() -> usize;
    fn data_type() -> DataType;
}

impl TestFloat for f32 {
    fn from_f32(val: f32) -> Self {
        val
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn size_of() -> usize {
        std::mem::size_of::<f32>()
    }
    fn data_type() -> DataType {
        DataType::F32
    }
}

impl TestFloat for f16 {
    fn from_f32(val: f32) -> Self {
        f16::from_f32(val)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn size_of() -> usize {
        std::mem::size_of::<f16>()
    }
    fn data_type() -> DataType {
        DataType::F16
    }
}

impl TestFloat for bf16 {
    fn from_f32(val: f32) -> Self {
        bf16::from_f32(val)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn size_of() -> usize {
        std::mem::size_of::<bf16>()
    }
    fn data_type() -> DataType {
        DataType::BF16
    }
}

// Helper function to compute expected RMS norm values using accumulation type precision
fn compute_expected_rms_norm(
    input_data_f32: &[f32], // Original f32 values for reference
    scale_data_f32: &[f32], // Original f32 values for reference
    epsilon: f32,
    input_type: DataType,
    scale_type: DataType,
    accumulation_type: DataType,
    scale_offset: f32,
    full_layer_mode: bool, // Corresponds to UpcastMode::FullLayer
) -> Vec<f32> {
    match accumulation_type {
        DataType::F16 => {
            // Convert inputs to their actual types, then to F16 for accumulation
            let input_f16: Vec<f16> = match input_type {
                DataType::F32 => input_data_f32.iter().map(|&x| f16::from_f32(x)).collect(),
                DataType::F16 => input_data_f32.iter().map(|&x| f16::from_f32(f16::from_f32(x).to_f32())).collect(),
                DataType::BF16 => input_data_f32.iter().map(|&x| f16::from_f32(bf16::from_f32(x).to_f32())).collect(),
                _ => panic!("Unsupported input type: {:?}", input_type),
            };
            let scale_f16: Vec<f16> = match scale_type {
                DataType::F32 => scale_data_f32.iter().map(|&x| f16::from_f32(x)).collect(),
                DataType::F16 => scale_data_f32.iter().map(|&x| f16::from_f32(f16::from_f32(x).to_f32())).collect(),
                DataType::BF16 => scale_data_f32.iter().map(|&x| f16::from_f32(bf16::from_f32(x).to_f32())).collect(),
                _ => panic!("Unsupported scale type: {:?}", scale_type),
            };
            let epsilon_f16 = f16::from_f32(epsilon);
            let scale_offset_f16 = f16::from_f32(scale_offset);

            // Compute mean square in F16 (matches Metal kernel logic)
            let mean_square_f16 = input_f16.iter().map(|&x| x * x).fold(f16::ZERO, |acc, x| acc + x)
                / f16::from_f32(input_f16.len() as f32);

            // Use rsqrt like the Metal kernel (inverse square root)
            let rms_norm_f16 = f16::from_f32((mean_square_f16 + epsilon_f16).to_f32().sqrt().recip());

            // Apply normalization and scaling based on mode (matches Metal kernel exactly)
            if full_layer_mode {
                // Full-layer: keep everything in accumulation precision
                input_f16
                    .iter()
                    .zip(scale_f16.iter())
                    .map(|(&input, &scale)| {
                        let normalized_high = input * rms_norm_f16;
                        let scale_value_high = scale + scale_offset_f16;
                        (normalized_high * scale_value_high).to_f32()
                    })
                    .collect()
            } else {
                // Only-normalization: cast down for the scale multiply
                input_f16
                    .iter()
                    .zip(scale_f16.iter())
                    .map(|(&input, &scale)| {
                        let normalized_high = input * rms_norm_f16;
                        let normalized_low = normalized_high; // Cast to output type (F16 in this case)
                        let scale_value_low = scale + scale_offset_f16; // Cast scale to output type
                        let product_low = normalized_low * scale_value_low;
                        product_low.to_f32() // Cast back to f32 for comparison
                    })
                    .collect()
            }
        },
        DataType::BF16 => {
            // Convert inputs to their actual types, then to BF16 for accumulation
            let input_bf16: Vec<bf16> = match input_type {
                DataType::F32 => input_data_f32.iter().map(|&x| bf16::from_f32(x)).collect(),
                DataType::F16 => input_data_f32.iter().map(|&x| bf16::from_f32(f16::from_f32(x).to_f32())).collect(),
                DataType::BF16 => input_data_f32.iter().map(|&x| bf16::from_f32(bf16::from_f32(x).to_f32())).collect(),
                _ => panic!("Unsupported input type: {:?}", input_type),
            };
            let scale_bf16: Vec<bf16> = match scale_type {
                DataType::F32 => scale_data_f32.iter().map(|&x| bf16::from_f32(x)).collect(),
                DataType::F16 => scale_data_f32.iter().map(|&x| bf16::from_f32(f16::from_f32(x).to_f32())).collect(),
                DataType::BF16 => scale_data_f32.iter().map(|&x| bf16::from_f32(bf16::from_f32(x).to_f32())).collect(),
                _ => panic!("Unsupported scale type: {:?}", scale_type),
            };
            let epsilon_bf16 = bf16::from_f32(epsilon);
            let scale_offset_bf16 = bf16::from_f32(scale_offset);

            // Compute mean square in BF16 (matches Metal kernel logic)
            let mean_square_bf16 = input_bf16.iter().map(|&x| x * x).fold(bf16::ZERO, |acc, x| acc + x)
                / bf16::from_f32(input_bf16.len() as f32);

            // Use rsqrt like the Metal kernel (inverse square root)
            let rms_norm_bf16 = bf16::from_f32((mean_square_bf16 + epsilon_bf16).to_f32().sqrt().recip());

            // Apply normalization and scaling based on mode (matches Metal kernel exactly)
            if full_layer_mode {
                // Full-layer: keep everything in accumulation precision
                input_bf16
                    .iter()
                    .zip(scale_bf16.iter())
                    .map(|(&input, &scale)| {
                        let normalized_high = input * rms_norm_bf16;
                        let scale_value_high = scale + scale_offset_bf16;
                        (normalized_high * scale_value_high).to_f32()
                    })
                    .collect()
            } else {
                // Only-normalization: cast down for the scale multiply
                input_bf16
                    .iter()
                    .zip(scale_bf16.iter())
                    .map(|(&input, &scale)| {
                        let normalized_high = input * rms_norm_bf16;
                        let normalized_low = normalized_high; // Cast to output type (BF16 in this case)
                        let scale_value_low = scale + scale_offset_bf16; // Cast scale to output type
                        let product_low = normalized_low * scale_value_low;
                        product_low.to_f32() // Cast back to f32 for comparison
                    })
                    .collect()
            }
        },
        DataType::F32 => {
            // Convert inputs to their actual types, then to F32 for accumulation
            let input_f32: Vec<f32> = match input_type {
                DataType::F32 => input_data_f32.to_vec(),
                DataType::F16 => input_data_f32.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
                DataType::BF16 => input_data_f32.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
                _ => panic!("Unsupported input type: {:?}", input_type),
            };
            let scale_f32: Vec<f32> = match scale_type {
                DataType::F32 => scale_data_f32.to_vec(),
                DataType::F16 => scale_data_f32.iter().map(|&x| f16::from_f32(x).to_f32()).collect(),
                DataType::BF16 => scale_data_f32.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
                _ => panic!("Unsupported scale type: {:?}", scale_type),
            };

            // Compute mean square in F32 (matches Metal kernel logic)
            let mean_square: f32 = input_f32.iter().map(|&x| x * x).sum::<f32>() / (input_f32.len() as f32);
            // Use rsqrt like the Metal kernel (inverse square root)
            let rms_norm = (mean_square + epsilon).sqrt().recip();

            // Apply normalization and scaling based on mode (matches Metal kernel exactly)
            if full_layer_mode {
                // Full-layer: keep everything in accumulation precision
                input_f32
                    .iter()
                    .zip(scale_f32.iter())
                    .map(|(&input, &scale)| {
                        let normalized_high = input * rms_norm;
                        let scale_value_high = scale + scale_offset;
                        normalized_high * scale_value_high
                    })
                    .collect()
            } else {
                // Only-normalization: cast down for the scale multiply
                input_f32
                    .iter()
                    .zip(scale_f32.iter())
                    .map(|(&input, &scale)| {
                        let normalized_high = input * rms_norm;
                        let normalized_low = normalized_high; // Cast to output type (F32 in this case)
                        let scale_value_low = scale + scale_offset; // Cast scale to output type
                        let product_low = normalized_low * scale_value_low;
                        product_low // Already f32
                    })
                    .collect()
            }
        },
        _ => panic!("Unsupported accumulation type: {:?}", accumulation_type),
    }
}

fn test_rms_norm_basic_typed<InputT, ScaleT, OutputT>(
    input_type: DataType,
    scale_type: DataType,
    output_type: DataType,
    accumulation_type: DataType,
) where
    InputT: TestFloat,
    ScaleT: TestFloat,
    OutputT: TestFloat,
{
    // Create Metal context
    let mtl_context = match MTLContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping RMS norm test: {}", e);
            return;
        },
    };

    // Test parameters
    let batch_size = 1i32;
    let model_dim = 4096i32;
    let epsilon = 1e-5f32;

    // Create simple test data with at least 4096 values
    let input_data_f32: Vec<f32> = (0..model_dim)
        .map(|i| {
            // Create a sine wave pattern with values roughly in [-2, 2] range
            // This gives us reasonable values that work well with F16 precision
            2.0 * f32::sin(i as f32 * 0.01) + 0.1 * (i as f32 / model_dim as f32)
        })
        .collect();
    let scale_data_f32: Vec<f32> = vec![1.0; model_dim as usize]; // Unit scales

    // Convert to target types
    let input_data: Vec<InputT> = input_data_f32.iter().map(|&x| InputT::from_f32(x)).collect();
    let scale_data: Vec<ScaleT> = scale_data_f32.iter().map(|&x| ScaleT::from_f32(x)).collect();

    // Calculate expected output
    let expected_output_f32 = compute_expected_rms_norm(
        &input_data_f32,
        &scale_data_f32,
        epsilon,
        input_type,
        scale_type,
        accumulation_type,
        0.0,   // scale_offset (matches the test's scale_offset parameter)
        false, // full_layer_mode (Standard kernel uses OnlyNormalization mode)
    );

    // Create Metal buffers
    let input_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&input_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let scale_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&scale_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let output_buffer = mtl_context
        .device
        .new_buffer(model_dim as usize * OutputT::size_of(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    // Create RMS norm kernel
    let kernel = RMSNormMetalKernel::new(&mtl_context, input_type, scale_type, output_type, accumulation_type)
        .expect("Failed to create RMS norm kernel");
    // Create command buffer and encode
    let command_buffer_ref = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
    let command_buffer = command_buffer_ref.to_owned();
    let compute_encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");

    kernel.encode(
        &input_buffer,
        &scale_buffer,
        &output_buffer,
        batch_size as u32,
        model_dim as u32,
        epsilon,
        0.0,
        false,
        &compute_encoder,
    );

    compute_encoder.end_encoding();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    // Read results
    let output_ptr = output_buffer.contents().as_ptr() as *const OutputT;
    let output_data = unsafe { std::slice::from_raw_parts(output_ptr, model_dim as usize) };
    let output_data_f32: Vec<f32> = output_data.iter().map(|&x| x.to_f32()).collect();

    println!(
        "Types: Input={:?}, Scale={:?}, Output={:?}, Accum={:?}",
        input_type, scale_type, output_type, accumulation_type
    );
    println!("Input (first 8): {:?}", &input_data_f32[..8]);
    println!("Expected (first 8): {:?}", &expected_output_f32[..8]);
    println!("Output (first 8): {:?}", &output_data_f32[..8]);
    println!(
        "RMS value: {}",
        f32::sqrt(input_data_f32.iter().map(|&x| x * x).sum::<f32>() / (model_dim as f32) + epsilon)
    );

    // Verify results with tolerance for different precisions
    let tolerance = if matches!(input_type, DataType::F16 | DataType::BF16)
        || matches!(scale_type, DataType::F16 | DataType::BF16)
        || matches!(output_type, DataType::F16 | DataType::BF16)
        || matches!(accumulation_type, DataType::F16 | DataType::BF16)
    {
        1e-2
    } else {
        1e-5
    };

    for (i, (&expected, &actual)) in expected_output_f32.iter().zip(output_data_f32.iter()).enumerate() {
        let diff = (expected - actual).abs();
        assert!(
            diff < tolerance,
            "Mismatch at index {}: expected {}, got {}, diff {} (tolerance {})",
            i,
            expected,
            actual,
            diff,
            tolerance
        );
    }

    println!("✅ RMS norm test passed!");
}

fn test_rms_norm_edge_cases_typed<InputT, ScaleT, OutputT>(
    input_type: DataType,
    scale_type: DataType,
    output_type: DataType,
    accumulation_type: DataType,
) where
    InputT: TestFloat,
    ScaleT: TestFloat,
    OutputT: TestFloat,
{
    let mtl_context = match MTLContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping RMS norm edge case test: {}", e);
            return;
        },
    };

    let batch_size = 1i32;
    let model_dim = 4i32;
    let epsilon = 1e-6f32;

    // Test with very small values (near epsilon)
    let input_data_f32: Vec<f32> = vec![1e-4, 2e-4, 3e-4, 4e-4]; // Scaled up for F16 precision
    let scale_data_f32: Vec<f32> = vec![2.0, 0.5, 1.5, 0.8]; // Non-unit scales

    let input_data: Vec<InputT> = input_data_f32.iter().map(|&x| InputT::from_f32(x)).collect();
    let scale_data: Vec<ScaleT> = scale_data_f32.iter().map(|&x| ScaleT::from_f32(x)).collect();

    let input_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&input_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let scale_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&scale_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let output_buffer = mtl_context
        .device
        .new_buffer(4 * OutputT::size_of(), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let kernel = RMSNormMetalKernel::new(&mtl_context, input_type, scale_type, output_type, accumulation_type)
        .expect("Failed to create RMS norm kernel");

    let command_buffer_ref = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
    let command_buffer = command_buffer_ref.to_owned();
    let compute_encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");

    kernel.encode(
        &input_buffer,
        &scale_buffer,
        &output_buffer,
        batch_size as u32,
        model_dim as u32,
        epsilon,
        0.0,
        false,
        &compute_encoder,
    );

    compute_encoder.end_encoding();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();

    let output_ptr = output_buffer.contents().as_ptr() as *const OutputT;
    let output_data = unsafe { std::slice::from_raw_parts(output_ptr, 4) };

    // Check that all values are finite (no NaN or infinity)
    for (i, &value) in output_data.iter().enumerate() {
        let f32_value = value.to_f32();
        assert!(f32_value.is_finite(), "Output at index {} is not finite: {} (type {:?})", i, f32_value, output_type);
    }

    println!(
        "✅ Edge case RMS norm test passed for types: Input={:?}, Scale={:?}, Output={:?}, Accum={:?}",
        input_type, scale_type, output_type, accumulation_type
    );
}

// All F32 accumulation tests
#[test]
fn test_rms_norm_f32_f32_f32_f32() {
    test_rms_norm_basic_typed::<f32, f32, f32>(DataType::F32, DataType::F32, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_f32_f16_f32_f32() {
    test_rms_norm_basic_typed::<f32, f16, f32>(DataType::F32, DataType::F16, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_f32_f32_f16_f32() {
    test_rms_norm_basic_typed::<f32, f32, f16>(DataType::F32, DataType::F32, DataType::F16, DataType::F32);
}
#[test]
fn test_rms_norm_f32_f16_f16_f32() {
    test_rms_norm_basic_typed::<f32, f16, f16>(DataType::F32, DataType::F16, DataType::F16, DataType::F32);
}
#[test]
fn test_rms_norm_f16_f32_f32_f32() {
    test_rms_norm_basic_typed::<f16, f32, f32>(DataType::F16, DataType::F32, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_f16_f16_f32_f32() {
    test_rms_norm_basic_typed::<f16, f16, f32>(DataType::F16, DataType::F16, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_f16_f32_f16_f32() {
    test_rms_norm_basic_typed::<f16, f32, f16>(DataType::F16, DataType::F32, DataType::F16, DataType::F32);
}
#[test]
fn test_rms_norm_f16_f16_f16_f32() {
    test_rms_norm_basic_typed::<f16, f16, f16>(DataType::F16, DataType::F16, DataType::F16, DataType::F32);
}

// F16 accumulation tests
#[test]
fn test_rms_norm_f32_f32_f32_f16() {
    test_rms_norm_basic_typed::<f32, f32, f32>(DataType::F32, DataType::F32, DataType::F32, DataType::F16);
}
#[test]
fn test_rms_norm_f32_f16_f32_f16() {
    test_rms_norm_basic_typed::<f32, f16, f32>(DataType::F32, DataType::F16, DataType::F32, DataType::F16);
}
#[test]
fn test_rms_norm_f32_f32_f16_f16() {
    test_rms_norm_basic_typed::<f32, f32, f16>(DataType::F32, DataType::F32, DataType::F16, DataType::F16);
}
#[test]
fn test_rms_norm_f32_f16_f16_f16() {
    test_rms_norm_basic_typed::<f32, f16, f16>(DataType::F32, DataType::F16, DataType::F16, DataType::F16);
}
#[test]
fn test_rms_norm_f16_f32_f32_f16() {
    test_rms_norm_basic_typed::<f16, f32, f32>(DataType::F16, DataType::F32, DataType::F32, DataType::F16);
}
#[test]
fn test_rms_norm_f16_f16_f32_f16() {
    test_rms_norm_basic_typed::<f16, f16, f32>(DataType::F16, DataType::F16, DataType::F32, DataType::F16);
}
#[test]
fn test_rms_norm_f16_f32_f16_f16() {
    test_rms_norm_basic_typed::<f16, f32, f16>(DataType::F16, DataType::F32, DataType::F16, DataType::F16);
}
#[test]
fn test_rms_norm_f16_f16_f16_f16() {
    test_rms_norm_basic_typed::<f16, f16, f16>(DataType::F16, DataType::F16, DataType::F16, DataType::F16);
}

// BFloat16 tests (using f32 storage)
#[test]
fn test_rms_norm_bf16_bf16_f32_f32() {
    test_rms_norm_basic_typed::<bf16, bf16, f32>(DataType::BF16, DataType::BF16, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_bf16_f32_f32_f32() {
    test_rms_norm_basic_typed::<bf16, f32, f32>(DataType::BF16, DataType::F32, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_bf16_f16_f32_f32() {
    test_rms_norm_basic_typed::<bf16, f16, f32>(DataType::BF16, DataType::F16, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_f32_bf16_f32_f32() {
    test_rms_norm_basic_typed::<f32, bf16, f32>(DataType::F32, DataType::BF16, DataType::F32, DataType::F32);
}

// Edge case tests for critical combinations
#[test]
fn test_rms_norm_edge_f32_f32_f32_f32() {
    test_rms_norm_edge_cases_typed::<f32, f32, f32>(DataType::F32, DataType::F32, DataType::F32, DataType::F32);
}
#[test]
fn test_rms_norm_edge_f16_f16_f16_f16() {
    test_rms_norm_edge_cases_typed::<f16, f16, f16>(DataType::F16, DataType::F16, DataType::F16, DataType::F16);
}
#[test]
fn test_rms_norm_edge_f32_f16_f16_f32() {
    test_rms_norm_edge_cases_typed::<f32, f16, f16>(DataType::F32, DataType::F16, DataType::F16, DataType::F32);
}

// Legacy test wrappers for compatibility
#[test]
fn test_rms_norm_kernel_basic() {
    test_rms_norm_basic_typed::<f32, f32, f32>(DataType::F32, DataType::F32, DataType::F32, DataType::F32);
}

#[test]
fn test_rms_norm_edge_cases() {
    test_rms_norm_edge_cases_typed::<f32, f32, f32>(DataType::F32, DataType::F32, DataType::F32, DataType::F32);
}

fn perf_rms_norm_with_size(
    batch_size: i32,
    model_dim: i32,
) {
    use std::time::Instant;

    use rand::{RngExt, SeedableRng, rngs::StdRng};

    // ---- Metal context ----
    let mtl_context = match MTLContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping RMS norm perf test: {}", e);
            return;
        },
    };

    const EPSILON: f32 = 1e-6;

    // ---- Create kernel ----
    let kernel = RMSNormMetalKernel::new(&mtl_context, DataType::F32, DataType::F32, DataType::F32, DataType::F32)
        .expect("Failed to create RMS norm kernel");

    // ---- Generate random data ----
    let mut rng = StdRng::seed_from_u64(42);
    let mut input_data = vec![0.0f32; (batch_size * model_dim) as usize];
    let mut scale_data = vec![0.0f32; model_dim as usize];

    for x in input_data.iter_mut() {
        *x = rng.random_range(-2.0f32..2.0f32);
    }
    for x in scale_data.iter_mut() {
        *x = rng.random_range(0.1f32..3.0f32);
    }

    // ---- Create Metal buffers ----
    let input_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&input_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let scale_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&scale_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let output_buffer = mtl_context
        .device
        .new_buffer(
            (batch_size * model_dim) as usize * std::mem::size_of::<f32>(),
            MTLResourceOptions::STORAGE_MODE_SHARED,
        )
        .expect("Failed to create buffer");

    // ---- Launch and time ----
    let command_buffer_ref = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
    let command_buffer = command_buffer_ref.to_owned();
    let compute_encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");

    kernel.encode(
        &input_buffer,
        &scale_buffer,
        &output_buffer,
        batch_size as u32,
        model_dim as u32,
        EPSILON,
        0.0,
        false,
        &compute_encoder,
    );

    compute_encoder.end_encoding();

    // Time both host-side and GPU execution
    let host_timer = Instant::now();
    command_buffer_ref.commit();
    command_buffer_ref.wait_until_completed();
    let host_elapsed_ms = host_timer.elapsed().as_secs_f64() * 1e3;

    // Get actual GPU execution time
    let gpu_elapsed_ms = command_buffer.gpu_execution_time_ms();

    match gpu_elapsed_ms {
        Some(gpu_time) => {
            println!(
                "RMS norm perf (batch={}, model_dim={}): GPU={:.2} ms, Host-side={:.2} ms",
                batch_size, model_dim, gpu_time, host_elapsed_ms
            );
        },
        None => {
            println!(
                "RMS norm perf (batch={}, model_dim={}): Host-side={:.2} ms (GPU timing unavailable)",
                batch_size, model_dim, host_elapsed_ms
            );
        },
    }

    // ---- Sanity check results ----
    let output_ptr = output_buffer.contents().as_ptr() as *const f32;
    let output_data = unsafe { std::slice::from_raw_parts(output_ptr, (batch_size * model_dim) as usize) };

    // Sample check for large outputs
    let sample_size = std::cmp::min(1000, output_data.len());
    for i in (0..output_data.len()).step_by(std::cmp::max(1, output_data.len() / sample_size)) {
        assert!(output_data[i].is_finite(), "Output at index {} is not finite: {}", i, output_data[i]);
    }

    // Check that normalization is working
    let mean_abs = output_data.iter().take(sample_size).map(|x| x.abs()).sum::<f32>() / sample_size as f32;
    assert!(
        mean_abs > 0.01 && mean_abs < 100.0,
        "Mean absolute value {} seems unreasonable - normalization may not be working",
        mean_abs
    );

    println!("✅ RMS norm performance test passed with reasonable output values");
}

#[test]
fn perf_rms_norm_8k() {
    perf_rms_norm_with_size(8, 8192); // Large model like LLaMA-70B
}

#[test]
fn perf_rms_norm_16k() {
    perf_rms_norm_with_size(16, 16384); // Huge model dimension
}

// ============================================================================
// QK Normalization Tests
// ============================================================================

#[test]
fn qk_norm_test() {
    // Test to verify that the QK norm kernel now accesses the correct data ranges
    let mtl_context = match MTLContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            println!("Skipping QK norm buffer addressing test: {}", e);
            return;
        },
    };

    let batch_size = 1i32;
    let num_q_heads = 4i32;
    let num_kv_heads = 2i32;
    let head_dim = 8i32;
    let epsilon = 1e-6f32;

    // QKV layout: [Q0, Q1, Q2, Q3, K0, K1, V0, V1] where each head has head_dim elements
    let qkv_width = ((num_q_heads + 2 * num_kv_heads) * head_dim) as usize;
    let total_size = (batch_size as usize) * qkv_width;

    // Create test data with distinct patterns for each head type
    let mut qkv_data = vec![0.0f32; total_size];
    let batch_offset = 0; // Only one batch

    // Q heads: values 1.0, 1.1, 1.2, 1.3, ...
    for head in 0..num_q_heads {
        for dim in 0..head_dim {
            let idx = batch_offset + (head * head_dim + dim) as usize;
            qkv_data[idx] = 1.0 + (head as f32) * 0.1 + (dim as f32) * 0.01;
        }
    }

    // K heads: values 2.0, 2.1, 2.2, 2.3, ... (start after Q heads)
    let k_offset = (num_q_heads * head_dim) as usize;
    for head in 0..num_kv_heads {
        for dim in 0..head_dim {
            let idx = batch_offset + k_offset + (head * head_dim + dim) as usize;
            qkv_data[idx] = 2.0 + (head as f32) * 0.1 + (dim as f32) * 0.01;
        }
    }

    // V heads: values 3.0, 3.1, 3.2, 3.3, ... (start after K heads)
    let v_offset = ((num_q_heads + num_kv_heads) * head_dim) as usize;
    for head in 0..num_kv_heads {
        for dim in 0..head_dim {
            let idx = batch_offset + v_offset + (head * head_dim + dim) as usize;
            qkv_data[idx] = 3.0 + (head as f32) * 0.1 + (dim as f32) * 0.01;
        }
    }

    // Create scale data (identity scaling for simplicity)
    let scale_data = vec![1.0f32; head_dim as usize];

    // Create buffers
    let qkv_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&qkv_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let q_scales_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&scale_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    let k_scales_buffer = mtl_context
        .device
        .new_buffer_with_data(bytemuck::cast_slice(&scale_data), MTLResourceOptions::STORAGE_MODE_SHARED)
        .expect("Failed to create buffer");

    // Create QK norm kernels
    let q_kernel = rms_norm::RMSNormKernel::new_with_mode(
        &mtl_context,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        RMSNormKernelType::QueryKey,
        false,
    )
    .expect("Failed to create Q norm kernel");

    let k_kernel = rms_norm::RMSNormKernel::new_with_mode(
        &mtl_context,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        DataType::F32,
        RMSNormKernelType::QueryKey,
        false,
    )
    .expect("Failed to create K norm kernel");

    // Test Q head normalization
    {
        let command_buffer = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
        let compute_encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");

        let _ = q_kernel.encode_qk_norm(
            &compute_encoder,
            QKNormArguments {
                qkv_input_buffer: &qkv_buffer,
                scales_buffer: &q_scales_buffer,
                qkv_output_buffer: &qkv_buffer,
                batch_size,
                num_q_heads,  // Now correctly passes actual head count
                num_kv_heads, // Now correctly passes actual head count
                head_dim,
                epsilon,
                scale_offset: 0.0,
                target: QKNormTarget::QueryHeads, // Only process Q heads
            },
        );

        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Verify Q heads were normalized (should have different values now)
    let output_data =
        unsafe { std::slice::from_raw_parts(qkv_buffer.contents().as_ptr() as *const f32, qkv_data.len()) };

    // Check that Q heads were processed (values should be different from original)
    for head in 0..num_q_heads as usize {
        let head_start = head * head_dim as usize;
        let original_value = 1.0 + (head as f32) * 0.1;
        let processed_value = output_data[head_start];

        // The processed value should be different (RMS normalized)
        assert!(
            (processed_value - original_value).abs() > 0.01,
            "Q head {} was not processed. Original: {}, Processed: {}",
            head,
            original_value,
            processed_value
        );
    }

    // Check that K heads were NOT processed (should still have original values)
    let k_start = (num_q_heads * head_dim) as usize;
    for head in 0..num_kv_heads as usize {
        let head_start = k_start + head * head_dim as usize;
        let expected_value = 2.0 + (head as f32) * 0.1;
        let actual_value = output_data[head_start];

        assert!(
            (actual_value - expected_value).abs() < 0.001,
            "K head {} was incorrectly processed during Q norm. Expected: {}, Actual: {}",
            head,
            expected_value,
            actual_value
        );
    }

    // Test K head normalization
    {
        let command_buffer = mtl_context.command_queue.command_buffer().expect("Failed to create command buffer");
        let compute_encoder = command_buffer.new_compute_command_encoder().expect("Failed to create compute encoder");

        let _ = k_kernel.encode_qk_norm(
            &compute_encoder,
            QKNormArguments {
                qkv_input_buffer: &qkv_buffer,
                scales_buffer: &k_scales_buffer,
                qkv_output_buffer: &qkv_buffer,
                batch_size,
                num_q_heads,  // Now correctly passes actual head count
                num_kv_heads, // Now correctly passes actual head count
                head_dim,
                epsilon,
                scale_offset: 0.0,
                target: QKNormTarget::KeyHeads, // Only process K heads
            },
        );

        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    // Verify K heads were normalized
    let final_data =
        unsafe { std::slice::from_raw_parts(qkv_buffer.contents().as_ptr() as *const f32, qkv_data.len()) };

    // Check that K heads were processed
    for head in 0..num_kv_heads as usize {
        let head_start = k_start + head * head_dim as usize;
        let original_value = 2.0 + (head as f32) * 0.1;
        let processed_value = final_data[head_start];

        // The processed value should be different (RMS normalized)
        assert!(
            (processed_value - original_value).abs() > 0.01,
            "K head {} was not processed. Original: {}, Processed: {}",
            head,
            original_value,
            processed_value
        );
    }

    println!("✅ QK norm buffer addressing is working correctly!");
    println!("Q heads: correctly processed only Q section");
    println!("K heads: correctly processed only K section");
}
