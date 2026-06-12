use crate::backends::metal::device_tier::DeviceTier;

pub(crate) const FP_BLOCK: u32 = 128;
pub(crate) const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
pub(crate) const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

type Tile = (u32, u32);

const DEFAULT_TILE: Tile = (DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP);
const D: Tile = DEFAULT_TILE;
const QUANT_N_BUCKETS: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKETS: [u32; 3] = [512, 2048, 8192];

// Tables are from June 2026 gemv_fine_tune sweeps; default cells keep SG8_R4.
#[rustfmt::skip]
const QUANT_TILE_LARGE: [[Tile; 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 2), D,      D,      D,      D,      D     ], // k <= 512
    [      (2, 1), (2, 2), (2, 2), (2, 2), (2, 1), (2, 4), D     ], // k <= 2048
    [      D,      (4, 2), D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      (2, 2), D,      D,      D,      D,      D     ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_SMALL: [[Tile; 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 4), D,      D,      D,      D,      D     ], // k <= 512
    [      (4, 2), (2, 2), (4, 2), D,      (2, 2), (4, 2), D     ], // k <= 2048
    [      D,      (4, 2), D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      (2, 2), D,      D,      D,      D,      D     ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_SMALL_G14: [[Tile; 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 4), D,      D,      D,      D,      D     ], // k <= 512
    [      (8, 2), (8, 2), (8, 2), (8, 2), (8, 2), (8, 2), (8, 2)], // k <= 2048
    [      D,      (8, 2), D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      D,      D,      D,      D,      D,      D     ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_SMALL_G13: [[Tile; 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 8), D,      D,      D,      D,      D     ], // k <= 512
    [      (8, 2), (8, 2), (8, 2), (8, 2), D,      D,      D     ], // k <= 2048
    [      D,      D,      D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      D,      D,      D,      D,      D,      D     ], // k > 8192
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

pub(crate) fn fp_tile(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
    tier: DeviceTier,
) -> (u32, u32) {
    let k_split = fp_k_split(m, n, k, input_aligned, tier);
    let results_per_simdgroup = fp_results_per_simdgroup(m, n, k, tier);
    (k_split, results_per_simdgroup)
}

fn fp_k_split(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
    tier: DeviceTier,
) -> u32 {
    if !input_aligned {
        return 1;
    }
    if m == 1 && tier == DeviceTier::Large && k < 4 * FP_BLOCK {
        return 1;
    }
    if m == 1 && tier == DeviceTier::SmallG13 && n >= 32768 {
        return 1;
    }

    let preferred = match m {
        0..=2 => 8,
        3..=4 if n <= 16384 => 8,
        3..=4 => 1,
        _ if n >= 4096 => 1,
        _ if k >= 16 * n || n <= 512 => 8,
        _ if n <= 1024 || k >= 3072 => 4,
        _ => 2,
    };
    preferred.min(prev_power_of_two(k / FP_BLOCK).min(DEFAULT_NUM_SIMDGROUPS))
}

fn fp_results_per_simdgroup(
    m: u32,
    n: u32,
    k: u32,
    tier: DeviceTier,
) -> u32 {
    if tier == DeviceTier::SmallG13 && m == 1 && n >= 6144 {
        DEFAULT_RESULTS_PER_SIMDGROUP
    } else if m == 1 && (k <= 8192 || tier != DeviceTier::Large) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    }
}

pub(crate) fn quant_tile(
    m: u32,
    n: u32,
    k: u32,
    has_rht: bool,
    tier: DeviceTier,
) -> Tile {
    if m != 1 {
        return DEFAULT_TILE;
    }
    if has_rht {
        return if tier == DeviceTier::Large && n > 2048 && n <= 4096 && k >= 2048 {
            (4, 8)
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
    if n < tile.1 {
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
            (DeviceTier::Large,   1, 12288, 1536, true,  (8, 1)),
            (DeviceTier::Large,   1, 12288,  512, true,  (4, 1)),
            (DeviceTier::Large,   1, 12288,  128, true,  (1, 1)),
            (DeviceTier::Large,   1, 12288, 1536, false, (1, 1)),
            (DeviceTier::Large,   1,  1536,  256, true,  (1, 1)),
            (DeviceTier::Small,   1,  1536,  256, true,  (2, 1)),
            (DeviceTier::Large,   8, 12288, 1536, true,  (1, 4)),
            (DeviceTier::Large,   8,  1536, 6144, true,  (4, 4)),
            (DeviceTier::Large,   1,  1536, 12288, true, (8, 4)),
            (DeviceTier::Small,   1,  1536, 12288, true, (8, 1)),
            (DeviceTier::SmallG13, 1, 262144, 1536, true, (1, 4)),
            (DeviceTier::SmallG13, 1,   5120, 1536, true, (8, 1)),
        ];

        for (tier, m, n, k, aligned, expected) in cases {
            assert_eq!(fp_tile(m, n, k, aligned, tier), expected, "tier={tier:?} m={m} n={n} k={k}");
        }
    }

    #[test]
    fn quant_policy_cases() {
        #[rustfmt::skip]
        let cases = [
            (DeviceTier::Large,    1,    256, 1536, false, (2, 1)),
            (DeviceTier::Large,    1,   2048, 1536, false, (2, 2)),
            (DeviceTier::Large,    1, 262144, 1536, false, DEFAULT_TILE),
            (DeviceTier::Small,    1,   1536,  256, false, (4, 4)),
            (DeviceTier::SmallG14, 1,   2048, 1536, false, (8, 2)),
            (DeviceTier::SmallG13, 1,    256, 1536, false, (8, 2)),
            (DeviceTier::SmallG13, 1,   1536,  256, false, (4, 8)),
            (DeviceTier::Large,    2,   2048, 1536, false, DEFAULT_TILE),
            (DeviceTier::Large,    1,   2560, 9216, true,  (4, 8)),
            (DeviceTier::Small,    1,   2560, 9216, true,  DEFAULT_TILE),
        ];

        for (tier, m, n, k, has_rht, expected) in cases {
            assert_eq!(quant_tile(m, n, k, has_rht, tier), expected, "tier={tier:?} m={m} n={n} k={k}");
        }
    }
}
