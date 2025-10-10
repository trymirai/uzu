//! Shared test utilities for MoE tests
//!
//! Contains common helper functions, CPU ground truth implementations,
//! buffer allocation helpers, and test fixtures used across all MoE test files.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::{Buffer as MTLBuffer, Device, MTLResourceOptions};
use uzu::backends::metal::MTLContext;

/// Create Metal context for testing
pub fn create_ctx() -> MTLContext {
    let device = Device::system_default().expect("No Metal device");
    let queue = device.new_command_queue();
    MTLContext::new(device, queue).expect("Failed to create MTLContext")
}

/// CPU ground truth: compute bucket counts from topk_ids
pub fn cpu_bucket_counts(
    topk_ids: &[i32],
    t: usize,
    k: usize,
    e: usize,
) -> Vec<u32> {
    let mut counts = vec![0u32; e];
    for ti in 0..t {
        for kk in 0..k {
            let id = topk_ids[ti * k + kk];
            if id >= 0 {
                let ue = id as usize;
                if ue < e {
                    counts[ue] += 1;
                }
            }
        }
    }
    counts
}

/// CPU ground truth: compute exclusive prefix scan (offsets) from counts
pub fn cpu_offsets_from_counts(counts: &[u32]) -> (Vec<u32>, u32) {
    let mut offsets = vec![0u32; counts.len() + 1];
    let mut sum = 0u32;
    for (i, &c) in counts.iter().enumerate() {
        offsets[i] = sum;
        sum += c;
    }
    offsets[counts.len()] = sum;
    (offsets, sum)
}

/// Helper to allocate buffer with data
pub fn alloc_buffer_with_data<T>(
    ctx: &MTLContext,
    data: &[T],
) -> MTLBuffer {
    ctx.device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * std::mem::size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Helper to allocate empty buffer
pub fn alloc_buffer<T>(
    ctx: &MTLContext,
    count: usize,
) -> MTLBuffer {
    ctx.device.new_buffer(
        (count * std::mem::size_of::<T>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// MoE test buffers fixture
pub struct MoeBuffers {
    pub x: MTLBuffer,              // [T, d_model]
    pub logits: MTLBuffer,         // [T, E]
    pub topk_ids: MTLBuffer,       // [T, K]
    pub topk_probs: MTLBuffer,     // [T, K]
    pub bucket_counts: MTLBuffer,  // [E]
    pub expert_offsets: MTLBuffer, // [E+1]
    pub bucketed_ids: MTLBuffer,   // [sum_k]
    pub x_perm: MTLBuffer,         // [sum_k, d_model]
    pub y_partial: MTLBuffer,      // [sum_k, d_model]
    pub y_out: MTLBuffer,          // [T, d_model]
    pub scales: MTLBuffer,         // [sum_k]
    pub tile_counts: MTLBuffer,    // [E]
    pub tile_offsets: MTLBuffer,   // [E+1]
    pub tile_map: MTLBuffer,       // [max_tiles * 3]
    pub total_tiles: MTLBuffer,    // [1]
    pub sumk: MTLBuffer,           // [1]
}

impl MoeBuffers {
    /// Create all buffers needed for MoE pipeline
    pub fn new(
        ctx: &MTLContext,
        t: usize,
        e: usize,
        k: usize,
        d_model: usize,
    ) -> Self {
        let sum_k = t * k;
        let max_tiles = sum_k * 4; // Conservative estimate

        Self {
            x: alloc_buffer::<bf16>(ctx, t * d_model),
            logits: alloc_buffer::<bf16>(ctx, t * e),
            topk_ids: alloc_buffer::<i32>(ctx, t * k),
            topk_probs: alloc_buffer::<f32>(ctx, t * k),
            bucket_counts: alloc_buffer::<u32>(ctx, e),
            expert_offsets: alloc_buffer::<u32>(ctx, e + 1),
            bucketed_ids: alloc_buffer::<i32>(ctx, sum_k),
            x_perm: alloc_buffer::<bf16>(ctx, sum_k * d_model),
            y_partial: alloc_buffer::<bf16>(ctx, sum_k * d_model),
            y_out: alloc_buffer::<bf16>(ctx, t * d_model),
            scales: alloc_buffer::<f32>(ctx, sum_k),
            tile_counts: alloc_buffer::<u32>(ctx, e),
            tile_offsets: alloc_buffer::<u32>(ctx, e + 1),
            tile_map: alloc_buffer::<u32>(ctx, max_tiles * 3),
            total_tiles: alloc_buffer::<u32>(ctx, 1),
            sumk: alloc_buffer::<u32>(ctx, 1),
        }
    }
}

/// CPU reference implementation for matmul: Y = X @ W^T + bias
///
/// Used for testing expert computations, router, etc.
///
/// # Arguments
/// * `x` - Input matrix [M, K] in row-major order
/// * `weight` - Weight matrix [N, K] in row-major order (will be transposed)
/// * `bias` - Optional bias vector [N]
/// * `m` - Number of rows in X
/// * `n` - Number of rows in W (output dimension)
/// * `k` - Number of columns in both X and W
///
/// # Returns
/// Output matrix [M, N] in row-major order
pub fn cpu_matmul_ref(
    x: &[bf16],
    weight: &[bf16],
    bias: Option<&[bf16]>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<bf16> {
    let mut output = vec![bf16::from_f32(0.0); m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc = if let Some(b) = bias {
                f32::from(b[col])
            } else {
                0.0
            };

            for i in 0..k {
                let x_val = f32::from(x[row * k + i]);
                let w_val = f32::from(weight[col * k + i]);
                acc += x_val * w_val;
            }

            output[row * n + col] = bf16::from_f32(acc);
        }
    }

    output
}

/// CPU reference for gather operation: x_perm[i] = x[bucketed_ids[i]]
///
/// # Arguments
/// * `x` - Input tensor [T, d_model]
/// * `bucketed_ids` - Index mapping [sum_k], values in range [0, T)
/// * `t` - Number of tokens
/// * `d_model` - Model dimension
/// * `sum_k` - Number of output rows
///
/// # Returns
/// Gathered tensor [sum_k, d_model]
pub fn cpu_gather(
    x: &[bf16],
    bucketed_ids: &[i32],
    t: usize,
    d_model: usize,
    sum_k: usize,
) -> Vec<bf16> {
    let mut x_perm = vec![bf16::from_f32(0.0); sum_k * d_model];
    for row in 0..sum_k {
        let token_id = bucketed_ids[row];
        if token_id >= 0 && (token_id as usize) < t {
            let src_offset = (token_id as usize) * d_model;
            let dst_offset = row * d_model;
            x_perm[dst_offset..dst_offset + d_model]
                .copy_from_slice(&x[src_offset..src_offset + d_model]);
        }
    }
    x_perm
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

/// CPU reference for finalize: scale and scatter expert outputs back to token positions
///
/// # Arguments
/// * `x_exp` - Expert outputs [sum_k, d_model]
/// * `bucketed_ids` - Token IDs [sum_k]
/// * `scales` - Scaling factors [sum_k]
/// * `t` - Number of tokens
/// * `d_model` - Model dimension
/// * `sum_k` - Total expert assignments
///
/// # Returns
/// Final output [T, d_model]
pub fn cpu_finalize(
    x_exp: &[bf16],
    bucketed_ids: &[i32],
    scales: &[f32],
    t: usize,
    d_model: usize,
    sum_k: usize,
) -> Vec<bf16> {
    let mut output = vec![bf16::from_f32(0.0); t * d_model];

    for row in 0..sum_k {
        let token_id = bucketed_ids[row];
        if token_id >= 0 && (token_id as usize) < t {
            let scale = scales[row];
            let src_offset = row * d_model;
            let dst_offset = (token_id as usize) * d_model;

            for d in 0..d_model {
                let scaled = f32::from(x_exp[src_offset + d]) * scale;
                let current = f32::from(output[dst_offset + d]);
                output[dst_offset + d] = bf16::from_f32(current + scaled);
            }
        }
    }

    output
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul_ref() {
        // Simple 2x3 @ 2x3^T test
        let x = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ];
        let w = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(0.0),
        ];

        let result = cpu_matmul_ref(&x, &w, None, 2, 2, 3);

        // Expected: [1*1 + 2*0 + 3*0, 1*0 + 2*1 + 3*0] = [1, 2]
        //           [4*1 + 5*0 + 6*0, 4*0 + 5*1 + 6*0] = [4, 5]
        assert_eq!(result.len(), 4);
        assert!((f32::from(result[0]) - 1.0).abs() < 1e-3);
        assert!((f32::from(result[1]) - 2.0).abs() < 1e-3);
        assert!((f32::from(result[2]) - 4.0).abs() < 1e-3);
        assert!((f32::from(result[3]) - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_cpu_tile_counts() {
        let offsets = vec![0, 10, 25, 40];
        let tile_counts = cpu_tile_counts(&offsets, 16);

        // seg_lens: [10, 15, 15]
        // tiles (BM=16): [1, 1, 1]
        assert_eq!(tile_counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_cpu_tile_scan() {
        let tile_counts = vec![2, 0, 3, 1];
        let (tile_offsets, total) = cpu_tile_scan(&tile_counts);

        assert_eq!(tile_offsets, vec![0, 2, 2, 5, 6]);
        assert_eq!(total, 6);
    }
}
