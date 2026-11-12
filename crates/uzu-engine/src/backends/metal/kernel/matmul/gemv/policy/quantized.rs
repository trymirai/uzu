use super::{DEFAULT_RESULTS_PER_SIMDGROUP, GemvTile};

/// Lane-sliced tiles split each quantization group across reduction lanes.
/// A lane stages 8 bytes: 16 W4 values or 8 W8 values.
const fn lane_tile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
    bits: u32,
    group: u32,
) -> GemvTile {
    let pack_factor = 32 / bits;
    GemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, group / (2 * pack_factor))
}

const fn lane_default(
    bits: u32,
    group: u32,
) -> GemvTile {
    lane_tile(8, DEFAULT_RESULTS_PER_SIMDGROUP, bits, group)
}

pub fn gathered_tile(
    bits: u32,
    group: u32,
    m: u32,
) -> Option<GemvTile> {
    if !matches!((bits, group), (4 | 8, 32 | 64) | (2, 128)) || !(1..=8).contains(&m) {
        return None;
    }
    Some(lane_default(bits, group))
}

pub fn select(
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
    bf16_io: bool,
) -> Option<GemvTile> {
    // The GEMV pipelines exist for BF16 I/O only. Rows 5..=8 stay on GEMV for tiny N only.
    if !bf16_io || (m > 4 && n > 64) {
        return None;
    }
    gathered_tile(bits, group, m)
}

pub fn interleaved_rows_tile(
    group: u32,
    k: u32,
) -> Option<GemvTile> {
    const VALUES_PER_LANE: u32 = 16;
    let strip_tile = GemvTile::quantized(8, 32, 1, 4, group / VALUES_PER_LANE);
    let tile = GemvTile {
        k_split: 8,
        ..strip_tile
    };
    let k_per_step = tile.k_split * tile.reduction_lanes * VALUES_PER_LANE;
    (matches!(group, 32 | 64) && k.is_multiple_of(k_per_step)).then_some(tile)
}
