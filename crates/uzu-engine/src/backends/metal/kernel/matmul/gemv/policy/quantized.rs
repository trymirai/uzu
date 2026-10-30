use super::{DEFAULT_RESULTS_PER_SIMDGROUP, GemvTile};

/// Lane-sliced tiles split each quantization group across reduction lanes.
/// A lane stages 8 bytes: 16 W4 values or 8 W8 values.
const fn lane_tile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
    bits: u32,
    group: u32,
) -> GemvTile {
    let pack_factor = match bits {
        4 => 8,
        8 => 4,
        _ => 1,
    };
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
    n: u32,
) -> Option<GemvTile> {
    if !matches!(bits, 4 | 8) || !matches!(group, 32 | 64) || !(1..=8).contains(&m) {
        return None;
    }
    let tile = lane_default(bits, group);
    (n >= tile.rows_per_lane()).then_some(tile)
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
    gathered_tile(bits, group, m, n)
}
