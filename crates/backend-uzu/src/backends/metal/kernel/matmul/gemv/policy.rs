use crate::backends::metal::device_tier::DeviceTier;

// Full-precision GEMV accumulates four K values per SIMD lane, so one full
// vectorized K block is 4 * 32 lanes.
pub(crate) const FP_K_BLOCK: u32 = 128;
pub(crate) const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
pub(crate) const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

/// Tile specialization selected for one GEMV dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemvTile {
    /// SIMD groups launched in one threadgroup.
    pub num_simdgroups: u32,
    /// Number of split-K slices reduced by one threadgroup.
    pub k_split: u32,
    /// Output rows computed by each SIMD group.
    pub results_per_simdgroup: u32,
}

const SMALL_G13_HUGE_N: u32 = 32768;
const SMALL_G13_WIDE_ROW_N: u32 = 6144;
const DEEP_K: u32 = 8192;
const FP_LARGE_SPLIT_K_MIN_DEPTH: u32 = 4 * FP_K_BLOCK;
const FP_K_DEPTH_N_MAX: u32 = 4095;
const FP_K_DEPTH_DEEP_MIN: u32 = 3072;
const FP_K_DEPTH_VERY_DEEP_RATIO: u32 = 16;

const fn tile(
    num_simdgroups: u32,
    k_split: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    GemvTile {
        num_simdgroups,
        k_split,
        results_per_simdgroup,
    }
}

const fn qtile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    // Q4 table sweeps only covered KS1; quant split-K is a separate future grid.
    tile(num_simdgroups, 1, results_per_simdgroup)
}

pub(crate) const DEFAULT_TILE: GemvTile = qtile(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP);
// Qxy = qtile(num_simdgroups=x, results_per_simdgroup=y), with KS1.
const Q21: GemvTile = qtile(2, 1);
const Q22: GemvTile = qtile(2, 2);
const Q24: GemvTile = qtile(2, 4);
const Q42: GemvTile = qtile(4, 2);
const Q44: GemvTile = qtile(4, 4);
const Q48: GemvTile = qtile(4, 8);
const Q82: GemvTile = qtile(8, 2);
const QUANT_N_BUCKET_MAXES: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKET_MAXES: [u32; 3] = [512, 2048, 8192];
const QUANT_RHT_TUNED_N_MIN_EXCLUSIVE: u32 = 2048;
const QUANT_RHT_TUNED_N_MAX: u32 = 4096;
const QUANT_RHT_TUNED_K_MIN: u32 = 2048;

fn table_bucket_index(
    value: u32,
    bucket_maxes: &[u32],
) -> usize {
    // Convert a dimension into a table row/column. For maxes [512, 2048],
    // indices are 0 for <=512, 1 for <=2048, and 2 for >2048.
    bucket_maxes.iter().position(|&max| value <= max).unwrap_or(bucket_maxes.len())
}

fn cap_k_split_to_complete_fp_k_blocks(
    k: u32,
    preferred: u32,
) -> u32 {
    // K_SPLIT variants are powers of two. Do not split beyond the number of
    // complete vectorized K blocks each slice can own.
    let complete_blocks = k / FP_K_BLOCK;
    if complete_blocks == 0 {
        return 1;
    }
    preferred.min((1 << complete_blocks.ilog2()).min(DEFAULT_NUM_SIMDGROUPS))
}

fn preferred_fp_k_split(
    m: u32,
    n: u32,
    k: u32,
) -> u32 {
    if m <= 2 {
        return 8;
    }
    if m <= 4 {
        return if n <= 16384 {
            8
        } else {
            1
        };
    }
    if n <= 512 {
        return 8;
    }
    if n <= 1024 {
        return if n != 0 && k / n >= FP_K_DEPTH_VERY_DEEP_RATIO {
            8
        } else {
            4
        };
    }
    if n <= FP_K_DEPTH_N_MAX {
        return if n != 0 && k / n >= FP_K_DEPTH_VERY_DEEP_RATIO {
            8
        } else if k >= FP_K_DEPTH_DEEP_MIN {
            4
        } else {
            2
        };
    }
    1
}

/// Selects the full-precision GEMV tile. `m` is the input-vector count,
/// `n` is the output row count, and `k` is the reduction depth.
pub(crate) fn fp_tile(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
    tier: DeviceTier,
) -> GemvTile {
    // FP sweeps covered SG2/SG4/SG8; SG changes did not produce portable
    // confirmed wins, so shipped FP policy keeps SG8 and tunes KS/R only.
    let should_disable_k_split = !input_aligned
        || (m == 1 && tier == DeviceTier::Large && k < FP_LARGE_SPLIT_K_MIN_DEPTH)
        || (m == 1 && tier == DeviceTier::SmallG13 && n >= SMALL_G13_HUGE_N);

    let k_split = if should_disable_k_split {
        1
    } else {
        cap_k_split_to_complete_fp_k_blocks(k, preferred_fp_k_split(m, n, k))
    };

    // R1 won most single-row FP sweeps; Large devices only switch back to R4
    // for deep-K rows, while SmallG13 wide rows keep R4.
    let results_per_simdgroup = if tier == DeviceTier::SmallG13 && m == 1 && n >= SMALL_G13_WIDE_ROW_N {
        DEFAULT_RESULTS_PER_SIMDGROUP
    } else if m == 1 && (k <= DEEP_K || tier != DeviceTier::Large) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    };

    tile(DEFAULT_NUM_SIMDGROUPS, k_split, results_per_simdgroup)
}

/// Selects the quantized GEMV tile. `m` is the input-vector count, `n` is the
/// output row count, `k` is the reduction depth, and `bits` is the quant width.
pub(crate) fn quant_tile(
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    has_rht: bool,
    tier: DeviceTier,
) -> GemvTile {
    // These tables are fitted for batch-1 Q4 only; Q8/future widths keep the
    // deterministic default until they have their own cold sweep.
    if m != 1 || bits != 4 {
        return DEFAULT_TILE;
    }
    if has_rht {
        // This special case mirrors quant bucket edges: n in (2048, 4096]
        // and k at or above the 2048 boundary.
        return if tier == DeviceTier::Large
            && n > QUANT_RHT_TUNED_N_MIN_EXCLUSIVE
            && n <= QUANT_RHT_TUNED_N_MAX
            && k >= QUANT_RHT_TUNED_K_MIN
        {
            qtile(4, 8)
        } else {
            DEFAULT_TILE
        };
    }

    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    // Q4 BF16 decode choices from June 2026 gemv_fine_tune sweeps; omitted
    // cells keep SG8_KS1_R4. Other quant widths keep DEFAULT_TILE until swept.
    let selected = match (tier, k_bucket, n_bucket) {
        (DeviceTier::Large, 0, 1) => Q42,
        (DeviceTier::Large, 1, 0) => Q21,
        (DeviceTier::Large, 1, 1..=3) => Q22,
        (DeviceTier::Large, 1, 4) => Q21,
        (DeviceTier::Large, 1, 5) => Q24,
        (DeviceTier::Large, 2, 1) => Q42,
        (DeviceTier::Large, 3, 1) => Q22,

        (DeviceTier::Small, 0, 1) => Q44,
        (DeviceTier::Small, 1, 0) => Q42,
        (DeviceTier::Small, 1, 1) => Q22,
        (DeviceTier::Small, 1, 2) => Q42,
        (DeviceTier::Small, 1, 4) => Q22,
        (DeviceTier::Small, 1, 5) => Q42,
        (DeviceTier::Small, 2, 1) => Q42,
        (DeviceTier::Small, 3, 1) => Q22,

        (DeviceTier::SmallG14, 0, 1) => Q44,
        (DeviceTier::SmallG14, 1, _) | (DeviceTier::SmallG14, 2, 1) => Q82,

        (DeviceTier::SmallG13, 0, 1) => Q48,
        (DeviceTier::SmallG13, 1, 0..=3) => Q82,

        _ => DEFAULT_TILE,
    };
    if n < selected.results_per_simdgroup {
        // Defensive: future R8 cells could otherwise overrun tiny N.
        DEFAULT_TILE
    } else {
        selected
    }
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/kernel/matmul/gemv/policy_test.rs"]
mod tests;
