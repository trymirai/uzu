//! Shared test utilities for MoE tests
//!
//! Contains util helper functions, CPU ground truth implementations,
//! buffer allocation helpers, and test fixtures used across all MoE test files.

#![cfg(metal_backend)]

use std::rc::Rc;

use bytemuck;
use half::bf16;
use metal::{MTLBuffer, MTLDeviceExt, MTLResourceOptions};
use objc2::{rc::Retained, runtime::ProtocolObject};
use uzu::backends::{
    common::{Backend, Context},
    metal::Metal,
};

/// Create Metal context for testing
pub fn create_ctx() -> Rc<<Metal as Backend>::Context> {
    <Metal as Backend>::Context::new().expect("Failed to create <Metal as Backend>::Context")
}

/// Helper to allocate buffer with data
pub fn alloc_buffer_with_data<T: bytemuck::NoUninit>(
    ctx: &<Metal as Backend>::Context,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    if data.is_empty() {
        // Metal doesn't allow creating 0-byte buffers, create a minimal buffer instead
        ctx.create_buffer(1).expect("Failed to create empty buffer")
    } else {
        ctx.device
            .new_buffer_with_data(bytemuck::cast_slice(data), MTLResourceOptions::STORAGE_MODE_SHARED)
            .expect("Failed to create buffer")
    }
}

/// Helper to allocate empty buffer
pub fn alloc_buffer<T>(
    ctx: &<Metal as Backend>::Context,
    count: usize,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    ctx.create_buffer(count * size_of::<T>()).expect("Failed to create buffer")
}

/// Compare two bf16 slices with tolerance
///
/// # Arguments
/// * `a` - First slice
/// * `b` - Second slice
/// * `tolerance` - Maximum absolute difference allowed
/// * `name` - Name for error messages
///
/// # Panics
/// Panics if any element differs by more than tolerance
pub fn assert_bf16_close(
    a: &[bf16],
    b: &[bf16],
    tolerance: f32,
    name: &str,
) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", name);

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;

    for (i, (&a_val, &b_val)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (f32::from(a_val) - f32::from(b_val)).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    assert!(
        max_diff <= tolerance,
        "{}: max difference {:.6} at index {} exceeds tolerance {:.6} (a={:.6}, b={:.6})",
        name,
        max_diff,
        max_idx,
        tolerance,
        f32::from(a[max_idx]),
        f32::from(b[max_idx])
    );
}
