//! Demonstration of Metal shader logging with raw command buffers
//!
//! This example shows that Metal shader logging DOES work when using
//! raw Metal command buffers created with MTLCommandBufferDescriptor.
//!
//! Run with:
//! ```
//! UZU_METAL_LOGGING=debug cargo run --example metal_logging_demo
//! ```
//!
//! NOTE: The main inference pipeline uses MPSCommandBuffer which doesn't
//! support log state attachment. This example uses raw Metal command buffers
//! to demonstrate the logging infrastructure works correctly.

use metal::{Device, MTLResourceOptions, MTLSize};
use uzu::backends::metal::metal_extensions::{
    CommandQueueLoggingExt, initialize_metal_logging,
};

const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_logging>

using namespace metal;

kernel void test_logging(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    // Log from first thread only
    if (tid == 0) {
        os_log_default.log("Starting test_logging kernel: size=%u", size);
    }

    float value = input[tid];

    // Log suspicious values
    if (value < 0.0f) {
        os_log_default.log("WARN: tid=%u has negative value=%f", tid, value);
    }

    // Simple computation
    output[tid] = value * 2.0f;

    // Log from last thread
    if (tid == size - 1) {
        os_log_default.log("Finished test_logging kernel: processed %u elements", size);
    }
}
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Metal Shader Logging Demo ===\n");

    // Create Metal device
    let device = Device::system_default().ok_or("No Metal device found")?;
    println!("Using device: {}", device.name());

    // Initialize Metal logging
    if let Some(level) = initialize_metal_logging(&device) {
        println!("Metal logging initialized at level: {}\n", level);
    } else {
        println!("Metal logging NOT enabled. Set UZU_METAL_LOGGING=debug\n");
    }

    // Create command queue
    let command_queue = device.new_command_queue();

    // Compile shader
    println!("Compiling shader...");
    let library = device
        .new_library_with_source(SHADER_SOURCE, &metal::CompileOptions::new())
        .map_err(|e| format!("Failed to compile shader: {}", e))?;

    let function = library.get_function("test_logging", None)?;

    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Failed to create pipeline: {}", e))?;

    println!("Shader compiled successfully\n");

    // Prepare test data
    const SIZE: usize = 10;
    let input_data: Vec<f32> =
        vec![1.0, 2.0, -3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let input_buffer = device.new_buffer_with_data(
        input_data.as_ptr() as *const _,
        (SIZE * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let output_buffer = device.new_buffer(
        (SIZE * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let size_buffer = device.new_buffer_with_data(
        &SIZE as *const _ as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    println!("Test data prepared:");
    println!("  Input:  {:?}", input_data);
    println!("  Size:   {}\n", SIZE);

    // Create command buffer WITH LOGGING
    println!("Creating command buffer with logging...");
    let command_buffer = command_queue.new_command_buffer_with_logging();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input_buffer), 0);
    encoder.set_buffer(1, Some(&output_buffer), 0);
    encoder.set_buffer(2, Some(&size_buffer), 0);

    let grid_size = MTLSize::new(SIZE as u64, 1, 1);
    let threadgroup_size = MTLSize::new(
        pipeline.max_total_threads_per_threadgroup().min(SIZE as u64),
        1,
        1,
    );

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();

    println!("Dispatching kernel...\n");
    println!("--- Metal Shader Logs (if UZU_METAL_LOGGING=debug) ---");

    command_buffer.commit();
    command_buffer.wait_until_completed();

    println!("--- End of Metal Shader Logs ---\n");

    // Read results
    let output_ptr = output_buffer.contents() as *const f32;
    let output_data: Vec<f32> =
        unsafe { std::slice::from_raw_parts(output_ptr, SIZE).to_vec() };

    println!("Results:");
    println!("  Output: {:?}\n", output_data);

    // Verify results
    let expected: Vec<f32> = input_data.iter().map(|x| x * 2.0).collect();
    let all_correct = output_data
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    if all_correct {
        println!("✓ Computation results are correct!");
    } else {
        println!("✗ Computation results don't match!");
        println!("  Expected: {:?}", expected);
    }

    println!("\n=== Demo Complete ===");
    println!("\nNOTE: If you don't see shader logs above:");
    println!("  1. Make sure UZU_METAL_LOGGING=debug is set");
    println!("  2. Check that you're on macOS 15.0+ with Metal 3.2 support");
    println!("  3. Logs appear between the '---' markers above");

    Ok(())
}
