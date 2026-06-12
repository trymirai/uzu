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
const FP_LARGE_MIN_SPLIT_K: u32 = 4 * FP_K_BLOCK;
const FP_K_DEPTH_N_MAX: u32 = 4095;
const FP_K_DEPTH_DEEP_MIN: u32 = 3072;
const FP_K_DEPTH_VERY_DEEP_RATIO: u32 = 16;
const FP_M_BUCKET_MAXES: [u32; 2] = [2, 4];
const FP_N_BUCKET_MAXES: [u32; 4] = [512, 1024, FP_K_DEPTH_N_MAX, 16384];
const FP_M_BUCKET_COUNT: usize = FP_M_BUCKET_MAXES.len() + 1;
const FP_N_BUCKET_COUNT: usize = FP_N_BUCKET_MAXES.len() + 1;
const FP_K_DEPTH_BUCKET_COUNT: usize = 3;

#[rustfmt::skip]
const FP_PREFERRED_K_SPLIT: [[[u32; FP_K_DEPTH_BUCKET_COUNT]; FP_N_BUCKET_COUNT]; FP_M_BUCKET_COUNT] = [
    //                     regular deep very-deep
    /* m<=2, n<=512   */ [[ 8,     8,   8 ], // decode and small async batches keep high split-K.
    /*       n<=1024  */  [ 8,     8,   8 ],
    /*       n<=4095  */  [ 8,     8,   8 ],
    /*       n<=16384 */  [ 8,     8,   8 ],
    /*       n>16384  */  [ 8,     8,   8 ]],

    /* m<=4, n<=512   */ [[ 8,     8,   8 ],
    /*       n<=1024  */  [ 8,     8,   8 ],
    /*       n<=4095  */  [ 8,     8,   8 ],
    /*       n<=16384 */  [ 8,     8,   8 ],
    /*       n>16384  */  [ 1,     1,   1 ]],

    /* m>4,  n<=512   */ [[ 8,     8,   8 ],
    /*       n<=1024  */  [ 4,     4,   8 ],
    /*       n<=4095  */  [ 2,     4,   8 ],
    /*       n<=16384 */  [ 1,     1,   1 ],
    /*       n>16384  */  [ 1,     1,   1 ]],
];

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
const D: GemvTile = DEFAULT_TILE;
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
const QUANT_N_BUCKET_COUNT: usize = QUANT_N_BUCKET_MAXES.len() + 1;
const QUANT_K_BUCKET_COUNT: usize = QUANT_K_BUCKET_MAXES.len() + 1;
const QUANT_RHT_TUNED_N_MIN_EXCLUSIVE: u32 = 2048;
const QUANT_RHT_TUNED_N_MAX: u32 = 4096;
const QUANT_RHT_TUNED_K_MIN: u32 = 2048;

// Q4 BF16 decode tables from June 2026 gemv_fine_tune sweeps; default cells
// keep SG8_KS1_R4. Other quant widths keep DEFAULT_TILE until swept.
#[rustfmt::skip]
const QUANT_TILE_LARGE: [[GemvTile; QUANT_N_BUCKET_COUNT]; QUANT_K_BUCKET_COUNT] = [
    //              n<=512  <=2048 <=4096 <=8192 <=16384 <=32768 >32768
    /* k<=512  */ [ D,      Q42,   D,     D,     D,      D,      D      ],
    /* k<=2048 */ [ Q21,    Q22,   Q22,   Q22,   Q21,    Q24,    D      ],
    /* k<=8192 */ [ D,      Q42,   D,     D,     D,      D,      D      ],
    /* k>8192  */ [ D,      Q22,   D,     D,     D,      D,      D      ],
];
#[rustfmt::skip]
const QUANT_TILE_SMALL: [[GemvTile; QUANT_N_BUCKET_COUNT]; QUANT_K_BUCKET_COUNT] = [
    //              n<=512  <=2048 <=4096 <=8192 <=16384 <=32768 >32768
    /* k<=512  */ [ D,      Q44,   D,     D,     D,      D,      D      ],
    /* k<=2048 */ [ Q42,    Q22,   Q42,   D,     Q22,    Q42,    D      ],
    /* k<=8192 */ [ D,      Q42,   D,     D,     D,      D,      D      ],
    /* k>8192  */ [ D,      Q22,   D,     D,     D,      D,      D      ],
];
#[rustfmt::skip]
const QUANT_TILE_SMALL_G14: [[GemvTile; QUANT_N_BUCKET_COUNT]; QUANT_K_BUCKET_COUNT] = [
    //              n<=512  <=2048 <=4096 <=8192 <=16384 <=32768 >32768
    /* k<=512  */ [ D,      Q44,   D,     D,     D,      D,      D      ],
    /* k<=2048 */ [ Q82,    Q82,   Q82,   Q82,   Q82,    Q82,    Q82    ],
    /* k<=8192 */ [ D,      Q82,   D,     D,     D,      D,      D      ],
    /* k>8192  */ [ D,      D,     D,     D,     D,      D,      D      ],
];
#[rustfmt::skip]
const QUANT_TILE_SMALL_G13: [[GemvTile; QUANT_N_BUCKET_COUNT]; QUANT_K_BUCKET_COUNT] = [
    //              n<=512  <=2048 <=4096 <=8192 <=16384 <=32768 >32768
    /* k<=512  */ [ D,      Q48,   D,     D,     D,      D,      D      ],
    /* k<=2048 */ [ Q82,    Q82,   Q82,   Q82,   D,      D,      D      ],
    /* k<=8192 */ [ D,      D,     D,     D,     D,      D,      D      ],
    /* k>8192  */ [ D,      D,     D,     D,     D,      D,      D      ],
];

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
    let k_split = if !input_aligned {
        1
    } else if m == 1 && tier == DeviceTier::Large && k < FP_LARGE_MIN_SPLIT_K {
        1
    } else if m == 1 && tier == DeviceTier::SmallG13 && n >= SMALL_G13_HUGE_N {
        1
    } else {
        // Only m>4 narrow-N rows use the K-depth axis today.
        const K_DEPTH_REGULAR: usize = 0;
        const K_DEPTH_DEEP: usize = 1;
        const K_DEPTH_VERY_DEEP: usize = 2;
        let k_depth_bucket = if m <= 4 || n > FP_K_DEPTH_N_MAX {
            K_DEPTH_REGULAR
        } else if n != 0 && k / n >= FP_K_DEPTH_VERY_DEEP_RATIO {
            K_DEPTH_VERY_DEEP
        } else if k >= FP_K_DEPTH_DEEP_MIN {
            K_DEPTH_DEEP
        } else {
            K_DEPTH_REGULAR
        };
        let m_bucket = table_bucket_index(m, &FP_M_BUCKET_MAXES);
        let n_bucket = table_bucket_index(n, &FP_N_BUCKET_MAXES);
        let preferred = FP_PREFERRED_K_SPLIT[m_bucket][n_bucket][k_depth_bucket];
        cap_k_split_to_complete_fp_k_blocks(k, preferred)
    };

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

    let table = match tier {
        DeviceTier::Large => &QUANT_TILE_LARGE,
        DeviceTier::Small => &QUANT_TILE_SMALL,
        DeviceTier::SmallG14 => &QUANT_TILE_SMALL_G14,
        DeviceTier::SmallG13 => &QUANT_TILE_SMALL_G13,
    };
    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    let selected = table[k_bucket][n_bucket];
    if n < selected.results_per_simdgroup {
        // Defensive: future R8 cells could otherwise overrun tiny N.
        DEFAULT_TILE
    } else {
        selected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp_policy_cases() {
        #[rustfmt::skip]
        let cases = [
            (DeviceTier::Large,    1, 12288, 1536, true,  tile(8, 8, 1)),
            (DeviceTier::Large,    1, 12288,  512, true,  tile(8, 4, 1)),
            (DeviceTier::Large,    1, 12288,  128, true,  tile(8, 1, 1)),
            (DeviceTier::Large,    1, 12288, 1536, false, tile(8, 1, 1)),
            (DeviceTier::Large,    1,  1536,  256, true,  tile(8, 1, 1)),
            (DeviceTier::Small,    1,  1536,  256, true,  tile(8, 2, 1)),
            (DeviceTier::Large,    8, 12288, 1536, true,  tile(8, 1, 4)),
            (DeviceTier::Large,    8,  1536, 6144, true,  tile(8, 4, 4)),
            (DeviceTier::Large,    1,  1536, 12288, true, tile(8, 8, 4)),
            (DeviceTier::Small,    1,  1536, 12288, true, tile(8, 8, 1)),
            (DeviceTier::SmallG13, 1, 262144, 1536, true, tile(8, 1, 4)),
            (DeviceTier::SmallG13, 1,   5120, 1536, true, tile(8, 8, 1)),
        ];

        for (tier, m, n, k, aligned, expected) in cases {
            assert_eq!(fp_tile(m, n, k, aligned, tier), expected, "tier={tier:?} m={m} n={n} k={k}");
        }
    }

    #[test]
    fn quant_policy_cases() {
        #[rustfmt::skip]
        let cases = [
            (DeviceTier::Large,    1,    256, 1536, 4, false, qtile(2, 1)),
            (DeviceTier::Large,    1,   2048, 1536, 4, false, qtile(2, 2)),
            (DeviceTier::Large,    1, 262144, 1536, 4, false, DEFAULT_TILE),
            (DeviceTier::Small,    1,   1536,  256, 4, false, qtile(4, 4)),
            (DeviceTier::SmallG14, 1,   2048, 1536, 4, false, qtile(8, 2)),
            (DeviceTier::SmallG13, 1,    256, 1536, 4, false, qtile(8, 2)),
            (DeviceTier::SmallG13, 1,   1536,  256, 4, false, qtile(4, 8)),
            (DeviceTier::Large,    2,   2048, 1536, 4, false, DEFAULT_TILE),
            (DeviceTier::Large,    1,   2048, 1536, 8, false, DEFAULT_TILE),
            (DeviceTier::Large,    1,   2560, 9216, 4, true,  qtile(4, 8)),
            (DeviceTier::Small,    1,   2560, 9216, 4, true,  DEFAULT_TILE),
        ];

        for (tier, m, n, k, bits, has_rht, expected) in cases {
            assert_eq!(
                quant_tile(m, n, k, bits, has_rht, tier),
                expected,
                "tier={tier:?} m={m} n={n} k={k} bits={bits}"
            );
        }
    }
}
