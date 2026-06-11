//! Pure GEMV dispatch policy: maps (m, n, k, mode, GpuDeviceTier) to a tile
//! (num_simdgroups, results_per_simdgroup) and a k-split. No Metal/PSO types.
//!
//! Tables are produced by
//! performance-benchmarks/benchmarks/scripts/fit_quant_table.py from
//! gemv_fine_tune sweep summaries; retune = sweep -> fit -> paste -> verify.
//! Buckets adopt a non-default tile only when every swept case in the bucket
//! beats the SG8_R4 default within the tier (do-no-harm).

use crate::backends::metal::context::GpuDeviceTier;

pub(crate) const FP_BLOCK: u32 = 128;
pub(crate) const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
pub(crate) const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

/// Output sizes at or above this prefer unsplit wide-row tiles on G13: the
/// split-k reduction is pure overhead once n alone saturates the small GPU.
/// Boundary sits above the up projections (n=12288/24576, sub-2%) and at/below
/// lm_head (n=262144).
const G13_HUGE_N: u32 = 32768;

/// Output sizes at or above this prefer wide row tiles (R4) on G13 while
/// keeping the k-split. Boundary sits below qkv n=6144 and above n=4096.
const G13_WIDE_ROW_N: u32 = 6144;

const QUANT_TILE_DEFAULT: (u32, u32) = (DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP);
const QUANT_N_BUCKETS: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKETS: [u32; 3] = [512, 2048, 8192];
const D: (u32, u32) = QUANT_TILE_DEFAULT;

/// Quant tile `(num_simdgroups, results_per_simdgroup)` by k-bucket (rows) x
/// n-bucket (columns). Wide / Apple9+ compacts prefer small SG2/SG4 tiles;
/// Apple8 (M2) prefers SG8 with halved R; G13 (M1) adopts only >2% wins.
#[rustfmt::skip]
const QUANT_TILE_WIDE: [[(u32, u32); 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 2), D,      D,      D,      D,      D     ], // k <= 512
    [      (2, 1), (2, 2), (2, 2), (2, 2), (2, 1), (2, 4), D     ], // k <= 2048
    [      D,      (4, 2), D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      (2, 2), D,      D,      D,      D,      D     ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_COMPACT: [[(u32, u32); 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 4), D,      D,      D,      D,      D     ], // k <= 512
    [      (4, 2), (2, 2), (4, 2), D,      (2, 2), (4, 2), D     ], // k <= 2048
    [      D,      (4, 2), D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      (2, 2), D,      D,      D,      D,      D     ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_COMPACT_G14: [[(u32, u32); 7]; 4] = [
    // n:  <=512   <=2048  <=4096  <=8192  <=16384 <=32768 >32768
    [      D,      (4, 4), D,      D,      D,      D,      D     ], // k <= 512
    [      (8, 2), (8, 2), (8, 2), (8, 2), (8, 2), (8, 2), (8, 2)], // k <= 2048
    [      D,      (8, 2), D,      D,      D,      D,      D     ], // k <= 8192
    [      D,      D,      D,      D,      D,      D,      D     ], // k > 8192
];
#[rustfmt::skip]
const QUANT_TILE_COMPACT_G13: [[(u32, u32); 7]; 4] = [
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

/// Full-precision k-split. Batch-1 favors maximal k-splitting (DRAM-resident
/// weights need many threadgroups in flight); larger batches re-read weight
/// rows per batch row and shift toward wider row tiles / no split.
pub(crate) fn fp_k_split(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
    device_tier: GpuDeviceTier,
) -> u32 {
    if !input_aligned {
        return 1;
    }
    // Tiny-k splits are pure reduction overhead on Max/Ultra-class GPUs.
    if m == 1 && device_tier == GpuDeviceTier::Wide && k < 4 * FP_BLOCK {
        return 1;
    }
    if device_tier == GpuDeviceTier::CompactG13 && m == 1 && n >= G13_HUGE_N {
        return 1;
    }
    let preferred = match m {
        0..=2 => 8,
        3..=4 => {
            if n <= 16384 {
                8
            } else {
                1
            }
        },
        // Beyond the validated batch range, keep the legacy selector.
        _ => {
            if n >= 4096 {
                1
            } else if k >= 16 * n || n <= 512 {
                8
            } else if n <= 1024 || k >= 3072 {
                4
            } else {
                2
            }
        },
    };
    // Every k-slice must cover at least one full block.
    let max_split = prev_power_of_two(k / FP_BLOCK).min(DEFAULT_NUM_SIMDGROUPS);
    preferred.min(max_split)
}

/// Full-precision results-per-simdgroup. Batch-1 wants single-row tiles for
/// threadgroup-count-driven latency hiding; only Wide GPUs flip to wider tiles
/// on deep k, and G13 prefers wide rows from mid-size n upward.
pub(crate) fn fp_results_per_simdgroup(
    m: u32,
    n: u32,
    k: u32,
    device_tier: GpuDeviceTier,
) -> u32 {
    if device_tier == GpuDeviceTier::CompactG13 && m == 1 && n >= G13_WIDE_ROW_N {
        return DEFAULT_RESULTS_PER_SIMDGROUP;
    }
    if m == 1 && (k <= 8192 || device_tier != GpuDeviceTier::Wide) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    }
}

/// Quant tile `(num_simdgroups, results_per_simdgroup)`. Only batch-1 decode is
/// tuned; larger batches keep the default. RHT outputs rotate one 32-row block
/// per threadgroup, so only 32-row tiles (SG8xR4, SG4xR8) are valid there.
pub(crate) fn quant_tile(
    m: u32,
    n: u32,
    k: u32,
    has_rht: bool,
    device_tier: GpuDeviceTier,
) -> (u32, u32) {
    if m != 1 {
        return QUANT_TILE_DEFAULT;
    }
    if has_rht {
        if device_tier == GpuDeviceTier::Wide && n > 2048 && n <= 4096 && k >= 2048 {
            return (4, 8);
        }
        return QUANT_TILE_DEFAULT;
    }
    let table = match device_tier {
        GpuDeviceTier::Wide => &QUANT_TILE_WIDE,
        GpuDeviceTier::Compact => &QUANT_TILE_COMPACT,
        GpuDeviceTier::CompactG14 => &QUANT_TILE_COMPACT_G14,
        GpuDeviceTier::CompactG13 => &QUANT_TILE_COMPACT_G13,
    };
    let (num_simdgroups, results_per_simdgroup) =
        table[bucket_index(k, &QUANT_K_BUCKETS)][bucket_index(n, &QUANT_N_BUCKETS)];
    if n < results_per_simdgroup {
        return QUANT_TILE_DEFAULT;
    }
    (num_simdgroups, results_per_simdgroup)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GEMV mode as it reaches the policy: alignment is already resolved.
    #[derive(Clone, Copy, Debug)]
    enum Mode {
        FpAligned,
        FpUnaligned,
        Quant,
        QuantRht,
    }

    /// Resolve the three tunable dispatch fields the way
    /// `GemvSpecialization::select` composes them, so one helper backs both the
    /// table test and the snapshot.
    fn resolve(
        tier: GpuDeviceTier,
        m: u32,
        n: u32,
        k: u32,
        mode: Mode,
    ) -> (u32, u32, u32) {
        let (is_quant, has_rht) = match mode {
            Mode::FpAligned | Mode::FpUnaligned => (false, false),
            Mode::Quant => (true, false),
            Mode::QuantRht => (true, true),
        };
        let input_aligned = match mode {
            Mode::FpAligned => k.is_multiple_of(FP_BLOCK),
            Mode::FpUnaligned => false,
            Mode::Quant | Mode::QuantRht => k.is_multiple_of(512),
        };
        let k_split = if is_quant {
            1
        } else {
            fp_k_split(m, n, k, input_aligned, tier)
        };
        let (sg, r) = if is_quant {
            quant_tile(m, n, k, has_rht, tier)
        } else {
            (DEFAULT_NUM_SIMDGROUPS, fp_results_per_simdgroup(m, n, k, tier))
        };
        // Returned as (sg, k_split, r) to read like the table rows below.
        (sg, k_split, r)
    }

    /// Intent table: (tier, m, n, k, mode, (sg, k_split, r)). Documents the
    /// notable policy decisions; `policy_snapshot` covers full breadth.
    #[rustfmt::skip]
    const CASES: &[(GpuDeviceTier, u32, u32, u32, Mode, (u32, u32, u32))] = &[
        // fp k-split: batch-1 maximizes split, capped by k-blocks.
        (GpuDeviceTier::Wide,       1, 12288, 1536, Mode::FpAligned,   (8, 8, 1)),
        (GpuDeviceTier::Wide,       1, 12288,  512, Mode::FpAligned,   (8, 4, 1)),
        (GpuDeviceTier::Wide,       1, 12288,  128, Mode::FpAligned,   (8, 1, 1)),
        (GpuDeviceTier::Compact,    1, 12288,  256, Mode::FpAligned,   (8, 2, 1)),
        // fp: unaligned k never splits.
        (GpuDeviceTier::Wide,       1, 12288, 1536, Mode::FpUnaligned, (8, 1, 1)),
        // fp tiny-k: Wide skips the split, Compact keeps it.
        (GpuDeviceTier::Wide,       1,  1536,  256, Mode::FpAligned,   (8, 1, 1)),
        (GpuDeviceTier::Compact,    1,  1536,  256, Mode::FpAligned,   (8, 2, 1)),
        // fp batch>4: legacy selector.
        (GpuDeviceTier::Wide,       8, 12288, 1536, Mode::FpAligned,   (8, 1, 4)),
        (GpuDeviceTier::Wide,       8,  1536, 6144, Mode::FpAligned,   (8, 4, 4)),
        // fp results-per-simdgroup: Wide widens on deep k, Compact never does.
        (GpuDeviceTier::Wide,       1,  1536, 12288, Mode::FpAligned,  (8, 8, 4)),
        (GpuDeviceTier::Wide,       1,  1536,  8192, Mode::FpAligned,  (8, 8, 1)),
        (GpuDeviceTier::Compact,    1,  1536, 12288, Mode::FpAligned,  (8, 8, 1)),
        // G13: huge-n unsplit wide rows; wide rows kept above n=6144.
        (GpuDeviceTier::CompactG13, 1, 262144, 1536, Mode::FpAligned,  (8, 1, 4)),
        (GpuDeviceTier::CompactG13, 1,  24576, 1536, Mode::FpAligned,  (8, 8, 4)),
        (GpuDeviceTier::CompactG13, 1,   5120, 1536, Mode::FpAligned,  (8, 8, 1)),
        (GpuDeviceTier::Compact,    1, 262144, 1536, Mode::FpAligned,  (8, 8, 1)),
        // quant tile table per tier.
        (GpuDeviceTier::Wide,       1,    256, 1536, Mode::Quant,      (2, 1, 1)),
        (GpuDeviceTier::Wide,       1,   2048, 1536, Mode::Quant,      (2, 1, 2)),
        (GpuDeviceTier::Wide,       1, 262144, 1536, Mode::Quant,      (8, 1, 4)),
        (GpuDeviceTier::Compact,    1,   1536,  256, Mode::Quant,      (4, 1, 4)),
        (GpuDeviceTier::CompactG14, 1,   2048, 1536, Mode::Quant,      (8, 1, 2)),
        (GpuDeviceTier::CompactG13, 1,    256, 1536, Mode::Quant,      (8, 1, 2)),
        (GpuDeviceTier::CompactG13, 1,   1536,  256, Mode::Quant,      (4, 1, 8)),
        // quant batch>1: default.
        (GpuDeviceTier::Wide,       2,   2048, 1536, Mode::Quant,      (8, 1, 4)),
        // quant+RHT: only 32-row tiles; Wide mid-n/deep-k wins SG4xR8.
        (GpuDeviceTier::Wide,       1,   2560, 9216, Mode::QuantRht,   (4, 1, 8)),
        (GpuDeviceTier::Wide,       1,   2048, 2048, Mode::QuantRht,   (8, 1, 4)),
        (GpuDeviceTier::Compact,    1,   2560, 9216, Mode::QuantRht,   (8, 1, 4)),
        (GpuDeviceTier::CompactG13, 1,   3072, 2048, Mode::QuantRht,   (8, 1, 4)),
    ];

    #[test]
    fn policy_table() {
        for &(tier, m, n, k, mode, expected) in CASES {
            assert_eq!(resolve(tier, m, n, k, mode), expected, "tier={tier:?} m={m} n={n} k={k} mode={mode:?}");
        }
    }

    /// FNV-1a digest of the (k_split, r, sg) triple across a dense grid. This
    /// pins the selector policy byte-for-byte: any policy change flips the
    /// digest and must be updated deliberately by running this test, reading
    /// the new value from the failure, and pasting it below.
    #[test]
    fn policy_snapshot() {
        const EXPECTED: u64 = 0x6d04b1d4a2039896;
        const TIERS: [GpuDeviceTier; 4] =
            [GpuDeviceTier::Wide, GpuDeviceTier::Compact, GpuDeviceTier::CompactG14, GpuDeviceTier::CompactG13];
        const MS: [u32; 4] = [1, 2, 4, 8];
        const KS: [u32; 14] = [128, 256, 512, 992, 1024, 1536, 2048, 2560, 4096, 6144, 8192, 9216, 12288, 262144];
        const NS: [u32; 13] = [256, 512, 1536, 2048, 2560, 3072, 4096, 8192, 8224, 12288, 16384, 32768, 262144];
        const MODES: [Mode; 4] = [Mode::FpAligned, Mode::FpUnaligned, Mode::Quant, Mode::QuantRht];
        let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
        for tier in TIERS {
            for m in MS {
                for k in KS {
                    for n in NS {
                        for mode in MODES {
                            let (sg, ks, r) = resolve(tier, m, n, k, mode);
                            for byte in [ks, r, sg].iter().flat_map(|v| v.to_le_bytes()) {
                                hash ^= byte as u64;
                                hash = hash.wrapping_mul(0x100000001b3);
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(hash, EXPECTED, "policy digest changed: 0x{hash:016x}");
    }
}
