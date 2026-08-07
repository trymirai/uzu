//! Tuned GEMM tile and split-K policy.
//!
//! These functions are the retunable part of GEMM dispatch: they map shape
//! features to choices. Eligibility, fallbacks, and legality stay in
//! `selection.rs`.

use super::GemmPlan;
use crate::backends::common::{
    gpu_types::gemm::{GemmBPrologueKind, GemmTiling},
    kernel::matmul::MatmulShape,
};

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
const SIMDGROUP_QUANT_LARGE_M_MIN: u32 = 64;
const SIMDGROUP_QUANT_WIDE_N_MIN: u32 = 6144;

const SPLIT_K_TARGET_TILES_FP: u32 = 512;
const SPLIT_K_TARGET_TILES_A8: u32 = 256;
const SPLIT_K_TARGET_TILES_A8_TILE32_W4: u32 = 512;
const SPLIT_K_TARGET_TILES_A8_TILE32_W8: u32 = 1024;

const STAGE_WEIGHT_SCALE_MIN_GROUPS: u32 = 6;

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
        (true, 0, _) => GemmTiling::Tile16x128x256_Simdgroups1x4,
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
    if 2 * m.max(n) > k {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else {
        GemmTiling::Tile64x32x32_Simdgroups2x2
    }
}

pub(super) fn simdgroup_quant_tile(
    m: u32,
    n: u32,
    group_size: u32,
) -> GemmTiling {
    if group_size < 32 {
        GemmTiling::Tile64x64x16_Simdgroups2x2
    } else if m < SIMDGROUP_QUANT_SMALL_M_MAX {
        GemmTiling::Tile8x32x32_Simdgroups1x1
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
        (true, _, _) => SPLIT_K_TARGET_TILES_A8,
        (false, _, _) => SPLIT_K_TARGET_TILES_FP,
    }
}

impl GemmPlan {
    pub(super) fn should_stage_weight_scales(
        self,
        shape: MatmulShape,
    ) -> bool {
        if shape.b_bits == Some(4) && shape.b_group_size == Some(32) && self.tiling.block_m() <= 32 {
            return false;
        }
        if self.tiling == GemmTiling::Tile32x64x256_Simdgroups2x2 {
            return shape
                .b_group_size
                .is_none_or(|group_size| shape.k / self.split_k / group_size >= STAGE_WEIGHT_SCALE_MIN_GROUPS);
        }
        true
    }

    pub(super) fn should_hoist_operand_addressing(
        self,
        shape: MatmulShape,
    ) -> bool {
        if matches!(shape.b_prologue, GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant) {
            return true;
        }
        self.tiling != GemmTiling::Tile128x128x256_Simdgroups4x4
    }
}
