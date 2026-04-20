use rand::{RngExt, SeedableRng, rngs::StdRng};
use backend_uzu::backends::common::{
    Encoder,
    kernel::moe::{MoeTileCountsArguments, MoeTileMapKernels, MoeTileScanArguments},
};

use crate::encodable_block::mlp::moe::tests::{
    common::helpers::{
        alloc_allocation, alloc_allocation_with_data, allocation_prefix_to_vec, allocation_to_vec, create_context,
    },
    cpu_tile_counts, cpu_tile_scan,
};

#[test]
fn test_tile_counts_correctness() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
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
            let offsets_buf = alloc_allocation_with_data::<B, u32>(&ctx, &offsets);
            let mut tile_counts_buf = alloc_allocation::<B, u32>(&ctx, e);

            // Execute kernel using kernel struct
            let tile_kernel = MoeTileMapKernels::<B>::new(&ctx).expect("MoeTileMapKernel::new");
            let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
            tile_kernel.encode_counts(
                &mut encoder,
                MoeTileCountsArguments {
                    offsets: &offsets_buf,
                    tile_counts: &mut tile_counts_buf,
                    e,
                },
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();

            let tile_counts_gpu = allocation_prefix_to_vec::<B, u32>(&tile_counts_buf, e);

            assert_eq!(tile_counts_gpu, tile_counts_cpu, "Tile counts mismatch for E={}", e);

            eprintln!("[TileCountsTest] ✓ PASSED");
        }
    });
}

#[test]
fn test_tile_scan_correctness() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();
        let mut rng = StdRng::seed_from_u64(2029);

        // Test multiple expert counts
        let expert_counts = vec![4, 8, 16, 64];

        for e in expert_counts {
            eprintln!("[TileScanTest] E={}", e);

            // Generate random tile counts
            let tile_counts: Vec<u32> = (0..e).map(|_| rng.random_range(0..20)).collect();

            let (tile_offsets_cpu, total_tiles_cpu) = cpu_tile_scan(&tile_counts);

            // GPU buffers
            let tile_counts_buf = alloc_allocation_with_data::<B, u32>(&ctx, &tile_counts);
            let mut tile_offsets_buf = alloc_allocation::<B, u32>(&ctx, e + 1);
            let mut total_tiles_buf = alloc_allocation::<B, u32>(&ctx, 1);

            // Execute kernel using kernel struct
            let tile_kernel = MoeTileMapKernels::<B>::new(&ctx).expect("MoeTileMapKernel::new");
            let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
            tile_kernel.encode_scan(
                &mut encoder,
                MoeTileScanArguments {
                    tile_counts: &tile_counts_buf,
                    tile_offsets: &mut tile_offsets_buf,
                    total_tiles: &mut total_tiles_buf,
                    e,
                },
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();

            let tile_offsets_gpu = allocation_prefix_to_vec::<B, u32>(&tile_offsets_buf, e + 1);
            let total_tiles_gpu = allocation_to_vec::<B, u32>(&total_tiles_buf)[0];

            assert_eq!(tile_offsets_gpu, tile_offsets_cpu, "Tile offsets mismatch for E={}", e);
            assert_eq!(total_tiles_gpu, total_tiles_cpu, "Total tiles mismatch for E={}", e);

            eprintln!("[TileScanTest] ✓ PASSED");
        }
    });
}

#[test]
fn test_tile_edge_cases() {
    for_each_non_cpu_backend!(|B| {
        let ctx = create_context::<B>();

        // Test with all empty experts (seg_len=0)
        {
            let e = 8;
            let offsets = vec![0u32; e + 1]; // All zeros

            let tile_counts_cpu = cpu_tile_counts(&offsets, 16);

            let offsets_buf = alloc_allocation_with_data::<B, u32>(&ctx, &offsets);
            let mut tile_counts_buf = alloc_allocation::<B, u32>(&ctx, e);

            let tile_kernel = MoeTileMapKernels::<B>::new(&ctx).expect("MoeTileMapKernel::new");
            let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
            tile_kernel.encode_counts(
                &mut encoder,
                MoeTileCountsArguments {
                    offsets: &offsets_buf,
                    tile_counts: &mut tile_counts_buf,
                    e,
                },
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();

            let tile_counts_gpu = allocation_prefix_to_vec::<B, u32>(&tile_counts_buf, e);

            assert_eq!(tile_counts_gpu, tile_counts_cpu);
            assert!(tile_counts_gpu.iter().all(|&c| c == 0));
            eprintln!("[TileEdgeCases] ✓ Empty experts PASSED");
        }

        // Test with single tile per expert (seg_len < BM=16)
        {
            let e = 4;
            let offsets = vec![0, 10, 15, 30, 50]; // Small segments

            let tile_counts_cpu = cpu_tile_counts(&offsets, 16);

            let offsets_buf = alloc_allocation_with_data::<B, u32>(&ctx, &offsets);
            let mut tile_counts_buf = alloc_allocation::<B, u32>(&ctx, e);

            let tile_kernel = MoeTileMapKernels::<B>::new(&ctx).expect("MoeTileMapKernels::new");
            let mut encoder = Encoder::new(ctx.as_ref()).expect("Failed to create encoder");
            tile_kernel.encode_counts(
                &mut encoder,
                MoeTileCountsArguments {
                    offsets: &offsets_buf,
                    tile_counts: &mut tile_counts_buf,
                    e,
                },
            );
            encoder.end_encoding().submit().wait_until_completed().unwrap();

            let tile_counts_gpu = allocation_prefix_to_vec::<B, u32>(&tile_counts_buf, e);

            assert_eq!(tile_counts_gpu, tile_counts_cpu);
            // With BM=16: seg_lens=[10, 5, 15, 20] -> tile_counts=[1, 1, 1, 2]
            assert_eq!(tile_counts_gpu, &[1, 1, 1, 2]);
            eprintln!("[TileEdgeCases] ✓ Small segments PASSED");
        }
    });
}
