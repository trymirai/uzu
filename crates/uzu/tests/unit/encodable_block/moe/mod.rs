#[macro_use]
#[path = "../../../common/mod.rs"]
mod common;

mod moe_block_e2e_test;
mod moe_experts_test;

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
