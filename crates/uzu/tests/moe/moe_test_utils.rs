//! Shared test utilities for MoE tests
//!
//! Contains common helper functions, CPU ground truth implementations,
//! buffer allocation helpers, and test fixtures used across all MoE test files.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use std::rc::Rc;

use bytemuck;
use half::bf16;
use metal::{MTLDeviceExt, MTLResourceOptions};
use uzu::backends::{
    common::Context,
    metal::{MTLBuffer, MTLContext, ProtocolObject, Retained},
};

/// Create Metal context for testing
pub fn create_ctx() -> Rc<MTLContext> {
    MTLContext::new().expect("Failed to create MTLContext")
}

/// Helper to allocate buffer with data
pub fn alloc_buffer_with_data<T: bytemuck::NoUninit>(
    ctx: &MTLContext,
    data: &[T],
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    if data.is_empty() {
        // Metal doesn't allow creating 0-byte buffers, create a minimal buffer instead
        ctx.create_buffer(1).expect("Failed to create empty buffer")
    } else {
        ctx.device
            .new_buffer_with_data(
                bytemuck::cast_slice(data),
                MTLResourceOptions::STORAGE_MODE_SHARED,
            )
            .expect("Failed to create buffer")
    }
}

/// Helper to allocate empty buffer
pub fn alloc_buffer<T>(
    ctx: &MTLContext,
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

/// CPU reference for tile counts: count number of BM-sized tiles per expert
///
/// # Arguments
/// * `offsets` - Expert segment offsets [E+1]
/// * `bm` - Tile size (must match kernel BM constant)
///
/// # Returns
/// Tile counts per expert [E]
pub fn cpu_tile_counts(
    offsets: &[u32],
    bm: usize,
) -> Vec<u32> {
    let e = offsets.len() - 1;
    let mut tile_counts = vec![0u32; e];
    for expert in 0..e {
        let seg_len = offsets[expert + 1] - offsets[expert];
        tile_counts[expert] = if seg_len == 0 {
            0
        } else {
            ((seg_len as usize + bm - 1) / bm) as u32
        };
    }
    tile_counts
}

/// CPU reference for tile scan: exclusive prefix sum of tile counts
///
/// # Arguments
/// * `tile_counts` - Tile counts per expert [E]
///
/// # Returns
/// * Tile offsets [E+1] (exclusive prefix sum)
/// * Total number of tiles
pub fn cpu_tile_scan(tile_counts: &[u32]) -> (Vec<u32>, u32) {
    let e = tile_counts.len();
    let mut tile_offsets = Vec::with_capacity(e + 1);
    tile_offsets.push(0);
    for i in 0..e {
        tile_offsets.push(tile_offsets[i] + tile_counts[i]);
    }
    let total_tiles = *tile_offsets.last().unwrap();
    (tile_offsets, total_tiles)
}
