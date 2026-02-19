#![cfg(any(target_os = "macos", target_os = "ios"))]

use metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::backends::{
    common::kernel::moe::{MoeTileCountsArguments, MoeTileMapKernels, MoeTileScanArguments},
    metal::Metal,
};

use super::test_utils::{alloc_buffer, alloc_buffer_with_data, cpu_tile_counts, cpu_tile_scan, create_ctx};

#[test]
fn test_tile_counts_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(2028);

    // Test multiple expert counts
    let expert_counts = vec![4, 8, 16, 64];

    for e in expert_counts {
        eprintln!("[TileCountsTest] E={}", e);

        // Generate random offsets (simulating expert segment sizes)
        let mut offsets = vec![0u32];
        for _ in 0..e {
            let seg_len = rng.random_range(0..200);
            offsets.push(offsets.last().unwrap() + seg_len);
        }

        let tile_counts_cpu = cpu_tile_counts(&offsets, 16);

        // GPU buffers
        let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
        let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);

        // Execute kernel using kernel struct
        let tile_kernel = MoeTileMapKernels::<Metal>::new(&ctx).expect("MoeTileMapKernel::new");
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        tile_kernel.encode_counts(
            &cb,
            &MoeTileCountsArguments {
                offsets_buffer: &offsets_buf,
                tile_counts_buffer: &tile_counts_buf,
                e,
            },
        );
        cb.commit();
        cb.wait_until_completed();

        // Compare
        let tile_counts_gpu =
            unsafe { std::slice::from_raw_parts(tile_counts_buf.contents().as_ptr() as *const u32, e) };

        assert_eq!(tile_counts_gpu, &tile_counts_cpu[..], "Tile counts mismatch for E={}", e);

        eprintln!("[TileCountsTest] ✓ PASSED");
    }
}

#[test]
fn test_tile_scan_correctness() {
    let ctx = create_ctx();
    let mut rng = StdRng::seed_from_u64(2029);

    // Test multiple expert counts
    let expert_counts = vec![4, 8, 16, 64];

    for e in expert_counts {
        eprintln!("[TileScanTest] E={}", e);

        // Generate random tile counts
        let tile_counts: Vec<u32> = (0..e).map(|_| rng.random_range(0..20)).collect();

        let (tile_offsets_cpu, total_tiles_cpu) = cpu_tile_scan(&tile_counts);

        // GPU buffers
        let tile_counts_buf = alloc_buffer_with_data(&ctx, &tile_counts);
        let tile_offsets_buf = alloc_buffer::<u32>(&ctx, e + 1);
        let total_tiles_buf = alloc_buffer::<u32>(&ctx, 1);

        // Execute kernel using kernel struct
        let tile_kernel = MoeTileMapKernels::<Metal>::new(&ctx).expect("MoeTileMapKernel::new");
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        tile_kernel.encode_scan(
            &cb,
            &MoeTileScanArguments {
                tile_counts_buffer: &tile_counts_buf,
                tile_offsets_buffer: &tile_offsets_buf,
                total_tiles_buffer: &total_tiles_buf,
                e,
            },
        );
        cb.commit();
        cb.wait_until_completed();

        // Compare
        let tile_offsets_gpu =
            unsafe { std::slice::from_raw_parts(tile_offsets_buf.contents().as_ptr() as *const u32, e + 1) };
        let total_tiles_gpu = unsafe { *(total_tiles_buf.contents().as_ptr() as *const u32) };

        assert_eq!(tile_offsets_gpu, &tile_offsets_cpu[..], "Tile offsets mismatch for E={}", e);
        assert_eq!(total_tiles_gpu, total_tiles_cpu, "Total tiles mismatch for E={}", e);

        eprintln!("[TileScanTest] ✓ PASSED");
    }
}

#[test]
fn test_tile_edge_cases() {
    let ctx = create_ctx();

    // Test with all empty experts (seg_len=0)
    {
        let e = 8;
        let offsets = vec![0u32; e + 1]; // All zeros

        let tile_counts_cpu = cpu_tile_counts(&offsets, 16);

        let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
        let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);

        let tile_kernel = MoeTileMapKernels::<Metal>::new(&ctx).expect("MoeTileMapKernel::new");
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        tile_kernel.encode_counts(
            &cb,
            &MoeTileCountsArguments {
                offsets_buffer: &offsets_buf,
                tile_counts_buffer: &tile_counts_buf,
                e,
            },
        );
        cb.commit();
        cb.wait_until_completed();

        let tile_counts_gpu =
            unsafe { std::slice::from_raw_parts(tile_counts_buf.contents().as_ptr() as *const u32, e) };

        assert_eq!(tile_counts_gpu, &tile_counts_cpu[..]);
        assert!(tile_counts_gpu.iter().all(|&c| c == 0));
        eprintln!("[TileEdgeCases] ✓ Empty experts PASSED");
    }

    // Test with single tile per expert (seg_len < BM=16)
    {
        let e = 4;
        let offsets = vec![0, 10, 15, 30, 50]; // Small segments

        let tile_counts_cpu = cpu_tile_counts(&offsets, 16);

        let offsets_buf = alloc_buffer_with_data(&ctx, &offsets);
        let tile_counts_buf = alloc_buffer::<u32>(&ctx, e);

        let tile_kernel = MoeTileMapKernels::<Metal>::new(&ctx).expect("MoeTileMapKernel::new");
        let cb = ctx.command_queue.command_buffer().expect("Failed to create command buffer");
        tile_kernel.encode_counts(
            &cb,
            &MoeTileCountsArguments {
                offsets_buffer: &offsets_buf,
                tile_counts_buffer: &tile_counts_buf,
                e,
            },
        );
        cb.commit();
        cb.wait_until_completed();

        let tile_counts_gpu =
            unsafe { std::slice::from_raw_parts(tile_counts_buf.contents().as_ptr() as *const u32, e) };

        assert_eq!(tile_counts_gpu, &tile_counts_cpu[..]);
        // With BM=16: seg_lens=[10, 5, 15, 20] -> tile_counts=[1, 1, 1, 2]
        assert_eq!(tile_counts_gpu, &[1, 1, 1, 2]);
        eprintln!("[TileEdgeCases] ✓ Small segments PASSED");
    }
}
