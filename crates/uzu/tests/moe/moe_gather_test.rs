#![cfg(any(target_os = "macos", target_os = "ios"))]

use half::bf16;
use metal::MTLResourceOptions;
use rand::{Rng, SeedableRng, rngs::StdRng};
use uzu::backends::metal::kernel::{KernelDataType, moe::{MoeGatherKernel, MoeGatherArguments}};

use super::test_utils::{create_ctx, cpu_gather};

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
        let x: Vec<bf16> = (0..t * d_model)
            .map(|_| bf16::from_f32(rng.random_range(-2.0..2.0)))
            .collect();

        // Random bucketed_ids with valid token indices
        let bucketed_ids: Vec<i32> = (0..sum_k)
            .map(|_| rng.random_range(0..t as i32))
            .collect();

        let x_cpu = cpu_gather(&x, &bucketed_ids, t, d_model, sum_k);

        // GPU buffers
        let x_buf = ctx.device.new_buffer_with_data(
            x.as_ptr() as *const _,
            (x.len() * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ids_buf = ctx.device.new_buffer_with_data(
            bucketed_ids.as_ptr() as *const _,
            (bucketed_ids.len() * std::mem::size_of::<i32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let x_perm_buf = ctx.device.new_buffer(
            (sum_k * d_model * std::mem::size_of::<bf16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Create sumk_buffer for API (kernel reads sumk_buf[0])
        let sum_k_u32 = sum_k as u32;
        let sumk_buf = ctx.device.new_buffer_with_data(
            &sum_k_u32 as *const u32 as *const _,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Execute gather kernel using kernel struct
        let gather = MoeGatherKernel::new(&ctx).expect("MoeGatherKernel::new");
        let cb = ctx.command_queue.new_command_buffer();
        gather.encode(
            &cb,
            KernelDataType::BFloat16,
            MoeGatherArguments {
                x_buffer: &x_buf,
                bucketed_ids_buffer: &ids_buf,
                x_perm_buffer: &x_perm_buf,
                sumk_buffer: &sumk_buf,
                t: t,
                k: sum_k / t,  // Decompose sum_k into k per token
                d_model,
            },
        ).expect("encode gather");
        cb.commit();
        cb.wait_until_completed();

        // Compare
        let x_gpu = unsafe {
            std::slice::from_raw_parts(
                x_perm_buf.contents() as *const bf16,
                sum_k * d_model,
            )
        };

        for i in 0..(sum_k * d_model) {
            let gpu_val = f32::from(x_gpu[i]);
            let cpu_val = f32::from(x_cpu[i]);
            assert!(
                (gpu_val - cpu_val).abs() < 1e-6,
                "Mismatch at {}: GPU={}, CPU={}",
                i,
                gpu_val,
                cpu_val
            );
        }

        eprintln!("[GatherTest] ✓ PASSED");
    }
}

#[test]
fn test_gather_edge_cases() {
    let ctx = create_ctx();

    // Test with negative/invalid IDs (should be ignored)
    let t = 8;
    let sum_k = 16;
    let d_model = 64;

    let x: Vec<bf16> = (0..t * d_model)
        .map(|i| bf16::from_f32((i % 10) as f32))
        .collect();

    // Mix of valid and invalid IDs
    let bucketed_ids: Vec<i32> = (0..sum_k)
        .map(|i| {
            match i % 4 {
                0 => -1,                 // Invalid
                1 => (t + 10) as i32,    // Out of bounds
                _ => ((i / 2) % t) as i32, // Valid
            }
        })
        .collect();

    let x_cpu = cpu_gather(&x, &bucketed_ids, t, d_model, sum_k);

    let x_buf = ctx.device.new_buffer_with_data(
        x.as_ptr() as *const _,
        (x.len() * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let ids_buf = ctx.device.new_buffer_with_data(
        bucketed_ids.as_ptr() as *const _,
        (bucketed_ids.len() * std::mem::size_of::<i32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let x_perm_buf = ctx.device.new_buffer(
        (sum_k * d_model * std::mem::size_of::<bf16>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Create sumk_buffer for API (kernel reads sumk_buf[0])
    let sum_k_u32 = sum_k as u32;
    let sumk_buf = ctx.device.new_buffer_with_data(
        &sum_k_u32 as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Execute gather kernel using kernel struct
    let gather = MoeGatherKernel::new(&ctx).expect("MoeGatherKernel::new");
    let cb = ctx.command_queue.new_command_buffer();
    gather.encode(
        &cb,
        KernelDataType::BFloat16,
        MoeGatherArguments {
            x_buffer: &x_buf,
            bucketed_ids_buffer: &ids_buf,
            x_perm_buffer: &x_perm_buf,
            sumk_buffer: &sumk_buf,
            t: t,
            k: sum_k / t,  // Decompose sum_k into k per token
            d_model,
        },
    ).expect("encode gather");
    cb.commit();
    cb.wait_until_completed();

    let x_gpu = unsafe {
        std::slice::from_raw_parts(
            x_perm_buf.contents() as *const bf16,
            sum_k * d_model,
        )
    };

    for i in 0..(sum_k * d_model) {
        let gpu_val = f32::from(x_gpu[i]);
        let cpu_val = f32::from(x_cpu[i]);
        assert!(
            (gpu_val - cpu_val).abs() < 1e-6,
            "Mismatch at {}: GPU={}, CPU={}",
            i,
            gpu_val,
            cpu_val
        );
    }

    eprintln!("[GatherTest] ✓ Edge cases PASSED");
}
