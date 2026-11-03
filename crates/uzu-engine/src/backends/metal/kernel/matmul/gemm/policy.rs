//! Tuned GEMM tile and split-K policy.
//!
//! These functions are the retunable part of GEMM dispatch: they map shape
//! features to choices. Eligibility, fallbacks, and legality stay in
//! `selection.rs`.

use crate::backends::metal::gpu_types::gemm::GemmTiling;

pub(super) const SIMDGROUP_FP_TILE: GemmTiling = GemmTiling::Tile64x32x32_Simdgroups2x2;
const MXU_DEFAULT_TILE: GemmTiling = GemmTiling::Tile64x64x256_Simdgroups2x2;

const MXU_SKINNY_M_MAX: u32 = 16;
const SKINNY_SQUARE_K_MAX: u32 = 2560;
const READOUT_N_TO_K_RATIO: u32 = 32;
const WIDE_N_DEEP_K_MIN: u32 = 4096;
const WIDE_N_DEEP_K_RATIO: u32 = 4;
const WIDE_N_MODERATE_K: u32 = 2560;
const WIDE_N_MODERATE_K_RATIO: u32 = 6;

const MXU_M_BUCKET_MAXES: [u32; 4] = [16, 63, 255, 511];
const MXU_N_BUCKET_MAXES: [u32; 2] = [63, 127];

// Two 8-row blocks: past this the 32-row tile has fewer, fuller blocks and wins on every chip.
const SIMDGROUP_QUANT_NARROW_TILE_M_MAX: u32 = 16;
const SIMDGROUP_QUANT_WIDE_BLOCK_M: u32 = 64;

const SPLIT_K_TARGET_TILES_FP: u32 = 512;
pub(super) const SPLIT_K_TARGET_TILES_A8: u32 = 256;
pub(super) const SPLIT_K_TARGET_TILES_A8_TILE16X32: u32 = 4 * SPLIT_K_TARGET_TILES_A8;
const SPLIT_K_TARGET_TILES_A8_TILE16X32_K_SPLIT: u32 = 2 * SPLIT_K_TARGET_TILES_A8;
pub(super) const SPLIT_K_TARGET_TILES_A8_TILE32_W4: u32 = 512;
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

pub(super) fn mxu_int8_gemm_tile(
    m: u32,
    n: u32,
) -> GemmTiling {
    match mxu_mn_tile(true, m, n) {
        GemmTiling::Tile128x128x256_Simdgroups4x4 => MXU_DEFAULT_TILE,
        GemmTiling::Tile16x32x256_Simdgroups1x1 => GemmTiling::Tile16x32x256_Simdgroups1x1x4,
        tiling => tiling,
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

pub(super) fn simdgroup_quant_tile(m: u32) -> GemmTiling {
    if m <= SIMDGROUP_QUANT_NARROW_TILE_M_MAX {
        GemmTiling::Tile8x32x32_Simdgroups1x1
    } else if (1..=SIMDGROUP_QUANT_WIDE_BLOCK_M / 2).contains(&(m % SIMDGROUP_QUANT_WIDE_BLOCK_M)) {
        // The last wide block would be at most half full: the narrow tile pads fewer rows.
        GemmTiling::Tile32x32x32_Simdgroups2x2
    } else {
        GemmTiling::Tile64x64x32_Simdgroups2x2
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
        (true, GemmTiling::Tile16x32x256_Simdgroups1x1x4, Some(4)) => SPLIT_K_TARGET_TILES_A8,
        (true, GemmTiling::Tile16x32x256_Simdgroups1x1x4, _) => SPLIT_K_TARGET_TILES_A8_TILE16X32_K_SPLIT,
        (true, _, _) => SPLIT_K_TARGET_TILES_A8,
        (false, _, _) => SPLIT_K_TARGET_TILES_FP,
    }
}
