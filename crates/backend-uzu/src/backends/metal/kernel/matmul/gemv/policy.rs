use crate::backends::metal::device_tier::DeviceTier;

pub(crate) const FP_BLOCK: u32 = 128;
pub(crate) const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
pub(crate) const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemvTile {
    pub num_simdgroups: u32,
    pub k_split: u32,
    pub results_per_simdgroup: u32,
}

const FP_NUM_SIMDGROUPS: u32 = 8;
const SMALL_G13_HUGE_N: u32 = 32768;
const SMALL_G13_WIDE_ROW_N: u32 = 6144;
const DEEP_K: u32 = 8192;

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

const DEFAULT_TILE: GemvTile = qtile(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP);
const D: GemvTile = DEFAULT_TILE;
const QUANT_N_BUCKETS: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKETS: [u32; 3] = [512, 2048, 8192];

// Q4 BF16 decode tables from June 2026 gemv_fine_tune sweeps; default cells
// keep SG8_KS1_R4. Other quant widths keep DEFAULT_TILE until swept.
#[rustfmt::skip]
const QUANT_TILE_LARGE: [[GemvTile; 7]; 4] = [
    // n:  <=512       <=2048      <=4096      <=8192      <=16384     <=32768     >32768
    [      D,          qtile(4, 2), D,          D,          D,          D,          D          ], // k <= 512
    [      qtile(2, 1), qtile(2, 2), qtile(2, 2), qtile(2, 2), qtile(2, 1), qtile(2, 4), D    ], // k <= 2048
    [      D,          qtile(4, 2), D,          D,          D,          D,          D          ], // k <= 8192
    [      D,          qtile(2, 2), D,          D,          D,          D,          D          ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_SMALL: [[GemvTile; 7]; 4] = [
    // n:  <=512       <=2048      <=4096      <=8192 <=16384     <=32768     >32768
    [      D,          qtile(4, 4), D,          D,     D,          D,          D       ], // k <= 512
    [      qtile(4, 2), qtile(2, 2), qtile(4, 2), D,   qtile(2, 2), qtile(4, 2), D    ], // k <= 2048
    [      D,          qtile(4, 2), D,          D,     D,          D,          D       ], // k <= 8192
    [      D,          qtile(2, 2), D,          D,     D,          D,          D       ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_SMALL_G14: [[GemvTile; 7]; 4] = [
    // n:  <=512       <=2048      <=4096      <=8192      <=16384     <=32768     >32768
    [      D,          qtile(4, 4), D,          D,          D,          D,          D          ], // k <= 512
    [      qtile(8, 2), qtile(8, 2), qtile(8, 2), qtile(8, 2), qtile(8, 2), qtile(8, 2), qtile(8, 2)], // k <= 2048
    [      D,          qtile(8, 2), D,          D,          D,          D,          D          ], // k <= 8192
    [      D,          D,          D,          D,          D,          D,          D          ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_SMALL_G13: [[GemvTile; 7]; 4] = [
    // n:  <=512       <=2048      <=4096      <=8192      <=16384 <=32768 >32768
    [      D,          qtile(4, 8), D,          D,          D,      D,      D       ], // k <= 512
    [      qtile(8, 2), qtile(8, 2), qtile(8, 2), qtile(8, 2), D,  D,      D       ], // k <= 2048
    [      D,          D,          D,          D,          D,      D,      D       ], // k <= 8192
    [      D,          D,          D,          D,          D,      D,      D       ], // k > 8192
];

fn bucket_index(
    value: u32,
    bounds: &[u32],
) -> usize {
    bounds.iter().position(|&bound| value <= bound).unwrap_or(bounds.len())
}

fn prev_power_of_two(value: u32) -> u32 {
    debug_assert!(value > 0);
    1 << (u32::BITS - 1 - value.leading_zeros())
}

fn cap_k_split_to_full_blocks(
    k: u32,
    preferred: u32,
) -> u32 {
    preferred.min(prev_power_of_two(k / FP_BLOCK).min(DEFAULT_NUM_SIMDGROUPS))
}

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
    } else if m == 1 && tier == DeviceTier::Large && k < 4 * FP_BLOCK {
        1
    } else if m == 1 && tier == DeviceTier::SmallG13 && n >= SMALL_G13_HUGE_N {
        1
    } else {
        let preferred = match m {
            0..=2 => 8,
            3..=4 if n <= 16384 => 8,
            3..=4 => 1,
            _ if n >= 4096 => 1,
            _ if k >= 16 * n || n <= 512 => 8,
            _ if n <= 1024 || k >= 3072 => 4,
            _ => 2,
        };
        cap_k_split_to_full_blocks(k, preferred)
    };

    let results_per_simdgroup = if tier == DeviceTier::SmallG13 && m == 1 && n >= SMALL_G13_WIDE_ROW_N {
        DEFAULT_RESULTS_PER_SIMDGROUP
    } else if m == 1 && (k <= DEEP_K || tier != DeviceTier::Large) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    };

    tile(FP_NUM_SIMDGROUPS, k_split, results_per_simdgroup)
}

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
        return if tier == DeviceTier::Large && n > 2048 && n <= 4096 && k >= 2048 {
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
    let tile = table[bucket_index(k, &QUANT_K_BUCKETS)][bucket_index(n, &QUANT_N_BUCKETS)];
    if n < tile.results_per_simdgroup {
        DEFAULT_TILE
    } else {
        tile
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
