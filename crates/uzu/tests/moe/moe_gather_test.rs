#![cfg(any(target_os = "macos", target_os = "ios"))]

use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

use half::bf16;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::backends::metal::kernel::{
    KernelDataType,
    moe::{MoeGatherArguments, MoeGatherKernels},
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, assert_bf16_close, create_ctx};

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
            x_perm[dst_offset..dst_offset + d_model].copy_from_slice(&x[src_offset..src_offset + d_model]);
        }
    }
    x_perm
}

#[test]
fn test_gather_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(2027);

    // Test multiple shapes: (T, sum_k, d_model)
    let shapes = vec![
        (1, 2, 64),     // Single token, K=2
        (4, 8, 128),    // Small batch
        (16, 32, 256),  // Medium
        (64, 128, 512), // Large
    ];

    for (t, sum_k, d_model) in shapes {
        eprintln!("[GatherTest] T={}, sum_k={}, d_model={}", t, sum_k, d_model);

        // Random input
        let x: Vec<bf16> = (0..t * d_model).map(|_| bf16::from_f32(rng.random_range(-2.0..2.0))).collect();

        // Random bucketed_ids with valid token indices
        let bucketed_ids: Vec<i32> = (0..sum_k).map(|_| rng.random_range(0..t as i32)).collect();

        let x_cpu = cpu_gather(&x, &bucketed_ids, t, d_model, sum_k);

        // GPU buffers
        let x_buf = alloc_buffer_with_data(&ctx, &x);
        let ids_buf = alloc_buffer_with_data(&ctx, &bucketed_ids);
        let x_perm_buf = alloc_buffer::<bf16>(&ctx, sum_k * d_model);

        // Create sumk_buffer for API (kernel reads sumk_buf[0])
        let sum_k_u32 = vec![sum_k as u32];
        let sumk_buf = alloc_buffer_with_data(&ctx, &sum_k_u32);

        // Execute gather kernel using kernel struct
        let gather = MoeGatherKernels::new(&ctx).expect("MoeGatherKernel::new");
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        gather.encode(
            &cb,
            KernelDataType::BFloat16,
            &MoeGatherArguments {
                x_buffer: &x_buf,
                bucketed_ids_buffer: &ids_buf,
                x_perm_buffer: &x_perm_buf,
                sumk_buffer: &sumk_buf,
                t: t,
                k: sum_k / t, // Decompose sum_k into k per token
                d_model,
            },
        );
        cb.commit();
        cb.wait_until_completed();

        // Compare
        let x_gpu =
            unsafe { std::slice::from_raw_parts(x_perm_buf.contents().as_ptr() as *const bf16, sum_k * d_model) };

        assert_bf16_close(x_gpu, &x_cpu, 1e-6, "gather output");

        eprintln!("[GatherTest] ✓ PASSED");
    }
}
