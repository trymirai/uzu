//! Tuned GEMM tile and split-K policy.
//!
//! These functions are the retunable part of GEMM dispatch: they map shape
//! features to choices. Eligibility, fallbacks, and legality stay in
//! `selection.rs`.

use metal::MTLGPUFamily;

use crate::backends::common::gpu_types::gemm::GemmTiling;

pub(super) const MXU_DEFAULT_TILE: GemmTiling = GemmTiling::Tile64x64x256_Simdgroups2x2;

const MXU_SKINNY_M_MAX: u32 = 16;
const SKINNY_SQUARE_K_MAX: u32 = 2560;
const READOUT_N_TO_K_RATIO: u32 = 32;
const WIDE_N_DEEP_K_MIN: u32 = 4096;
const WIDE_N_DEEP_K_RATIO: u32 = 4;
const WIDE_N_MODERATE_K: u32 = 2560;
const WIDE_N_MODERATE_K_RATIO: u32 = 6;

const MXU_M_BUCKET_MAXES: [u32; 4] = [16, 63, 255, 511];
const MXU_N_BUCKET_MAXES: [u32; 2] = [63, 127];

const SIMDGROUP_QUANT_SMALL_M_MAX: u32 = 32;
const SIMDGROUP_QUANT_NARROW_BLOCK_M: u32 = 8;
const SIMDGROUP_QUANT_LARGE_M_MIN: u32 = 64;
const SIMDGROUP_QUANT_WIDE_N_MIN: u32 = 6144;

/// Split-K buys threadgroups, and how many threadgroups are worth having is a
/// property of the machine rather than of a fixed tile count. This is the
/// production INT4 retune's `SPLIT_K_TARGET_SIMDGROUPS_PER_CORE`, reused
/// verbatim: the trellis arm feeds the same integer schedule on the same MXU,
/// so the supply it wants is the same supply.
const TRELLIS_TARGET_SIMDGROUPS_PER_CORE: u32 = 206;

/// How far a trellis split may go, as `(max M, ceiling)` rungs falling through
/// to "do not split", measured on a 40-core M5 Max.
///
/// A cap on the split-K partial planes as a share of the tape was tried here
/// and REMOVED: three probe cells that force the uncapped split measured 0.996,
/// 0.940 and 1.010 against it, so the planes cost nothing the ceiling does not
/// already bound. Unlike a dequantizing GEMM, splitting a tape multiplies
/// DECODE work as well as partial traffic, which is what these rungs bound.
const TRELLIS_SPLIT_K_CEILINGS: [(u32, u32); 2] = [(32, 8), (64, 4)];

const SPLIT_K_TARGET_TILES_FP: u32 = 512;
const SPLIT_K_TARGET_TILES_A8: u32 = 256;
const SPLIT_K_TARGET_TILES_A8_TILE16X32: u32 = 4 * SPLIT_K_TARGET_TILES_A8;
const SPLIT_K_TARGET_TILES_A8_TILE32_W4: u32 = 512;
const SPLIT_K_TARGET_TILES_A8_TILE32_W8: u32 = 1024;

fn bucket(
    value: u32,
    bucket_maxes: &[u32],
) -> usize {
    bucket_maxes.partition_point(|&max| value > max)
}

pub(super) fn mxu_mn_tile(
    is_a_int8: bool,
    m: u32,
    n: u32,
) -> GemmTiling {
    match (is_a_int8, bucket(m, &MXU_M_BUCKET_MAXES), bucket(n, &MXU_N_BUCKET_MAXES)) {
        (_, _, 0) => GemmTiling::Tile64x32x256_Simdgroups4x1,
        (false, 0..=1, _) => GemmTiling::Tile32x64x256_Simdgroups2x2,
        (false, 3..=4, 2) => GemmTiling::Tile128x128x256_Simdgroups4x4,
        (true, 0, _) => GemmTiling::Tile16x32x256_Simdgroups1x1,
        (true, 1, _) => GemmTiling::Tile32x64x256_Simdgroups2x2,
        (true, 4, _) => GemmTiling::Tile128x128x256_Simdgroups4x4,
        _ => MXU_DEFAULT_TILE,
    }
}

pub(super) fn mxu_fp_tile(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if m >= 64 || n < 64 {
        return mxu_mn_tile(false, m, n);
    }
    if n == k {
        return if m < MXU_SKINNY_M_MAX && k <= SKINNY_SQUARE_K_MAX {
            GemmTiling::Tile16x32x256_Simdgroups1x1
        } else {
            GemmTiling::Tile32x64x256_Simdgroups2x2
        };
    }
    if m >= MXU_SKINNY_M_MAX {
        return mxu_mn_tile(false, m, n);
    }
    if k > n {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    if n > READOUT_N_TO_K_RATIO.saturating_mul(k) {
        return GemmTiling::Tile16x32x256_Simdgroups1x1;
    }
    if (k >= WIDE_N_DEEP_K_MIN && n >= WIDE_N_DEEP_K_RATIO.saturating_mul(k))
        || (k == WIDE_N_MODERATE_K && n >= WIDE_N_MODERATE_K_RATIO.saturating_mul(k))
    {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    GemmTiling::Tile32x64x256_Simdgroups2x2
}

pub(super) fn simdgroup_fp_tile(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if 2_u32.saturating_mul(m.max(n)) > k {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else {
        GemmTiling::Tile64x32x32_Simdgroups2x2
    }
}

/// A partial trailing M block costs a second pass over the weights. Older GPUs are bound by that;
/// Apple9 and newer would rather keep the narrow tile's parallelism.
fn prefers_wide_partial_m_tile(apple_gpu_family: MTLGPUFamily) -> bool {
    apple_gpu_family <= MTLGPUFamily::Apple8
}

pub(super) fn simdgroup_quant_tile(
    m: u32,
    n: u32,
    group_size: u32,
    apple_gpu_family: MTLGPUFamily,
) -> GemmTiling {
    if group_size < 32 {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else if m < SIMDGROUP_QUANT_SMALL_M_MAX {
        if !prefers_wide_partial_m_tile(apple_gpu_family)
            || m <= SIMDGROUP_QUANT_NARROW_BLOCK_M
            || m.is_multiple_of(SIMDGROUP_QUANT_NARROW_BLOCK_M)
        {
            GemmTiling::Tile8x32x32_Simdgroups1x1
        } else {
            GemmTiling::Tile32x32x32_Simdgroups2x2
        }
    } else if m >= SIMDGROUP_QUANT_LARGE_M_MIN && n >= SIMDGROUP_QUANT_WIDE_N_MIN && n.is_multiple_of(64) {
        GemmTiling::Tile64x64x32_Simdgroups2x2
    } else {
        GemmTiling::Tile32x32x32_Simdgroups2x2
    }
}

pub(super) fn split_k_target_tiles(
    is_a_int8: bool,
    tiling: GemmTiling,
    b_bits: Option<u32>,
) -> u32 {
    match (is_a_int8, tiling, b_bits) {
        (true, GemmTiling::Tile32x64x256_Simdgroups2x2, Some(4)) => SPLIT_K_TARGET_TILES_A8_TILE32_W4,
        (true, GemmTiling::Tile32x64x256_Simdgroups2x2, _) => SPLIT_K_TARGET_TILES_A8_TILE32_W8,
        (true, GemmTiling::Tile16x32x256_Simdgroups1x1, _) => SPLIT_K_TARGET_TILES_A8_TILE16X32,
        (true, _, _) => SPLIT_K_TARGET_TILES_A8,
        (false, _, _) => SPLIT_K_TARGET_TILES_FP,
    }
}

/// The smallest split that lifts a trellis dispatch to
/// [`TRELLIS_TARGET_SIMDGROUPS_PER_CORE`] simdgroups per GPU core, bounded by
/// the M ceiling.
///
/// Why the trellis arm needs its own rule at all: `split_k_target_tiles` is a
/// fixed tile count, so a grid at least half as wide as the target answers "do
/// not split" however little of the machine it fills. `Tile16x32x256` over
/// N = 16480 is 515 threadgroups of one simdgroup -- 12.9 simdgroups per core
/// on a 40-core part -- and the a8 tile target still says 1. The A/B that
/// retired the tile target for this prologue is the round 7 trellis split-K
/// bench (`scratchpad/round7/draft.md`); the plumbing that measured both rules
/// in one process has been removed and is recoverable from git history.
pub(super) fn trellis_split_k(
    m: u32,
    base_tiles: u32,
    tiling: GemmTiling,
    gpu_core_count: u32,
) -> u32 {
    let simdgroups_per_tile = tiling.simdgroups_m().saturating_mul(tiling.simdgroups_n()).max(1);
    let unsplit_simdgroups = base_tiles.saturating_mul(simdgroups_per_tile).max(1);
    let target = TRELLIS_TARGET_SIMDGROUPS_PER_CORE.saturating_mul(gpu_core_count.max(1));
    let ceiling = TRELLIS_SPLIT_K_CEILINGS.iter().find(|(max_m, _)| m <= *max_m).map_or(1, |&(_, c)| c);
    target.div_ceil(unsplit_simdgroups).clamp(1, ceiling.max(1))
}
