//! Shared test utilities for MoE tests
//!
//! Contains util helper functions, CPU ground truth implementations,
//! buffer allocation helpers, and test fixtures used across all MoE test files.

#![cfg(metal_backend)]

use std::rc::Rc;

use bytemuck;
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
