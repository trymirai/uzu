use crate::backends::metal::device_profile::{DeviceProfile, DeviceSize, GpuFamily};

mod quantized;

pub(super) use quantized::{gathered_tile, select as quantized_tile};

// Full-precision GEMV accumulates four K values per SIMD lane, so one full
// vectorized K block is 4 * 32 lanes.
pub(super) const FP_K_BLOCK: u32 = 128;
pub(super) const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
pub(super) const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemvTile {
    /// SIMD groups launched in one threadgroup.
    pub(super) num_simdgroups: u32,
    /// Number of split-K slices reduced by one threadgroup.
    pub(super) k_split: u32,
    /// Output rows computed by each SIMD group.
    pub(super) results_per_simdgroup: u32,
    /// Input rows processed by one threadgroup.
    pub(super) input_row_tile: u32,
    /// SIMD lanes cooperating over K for one output-row block.
    pub(super) reduction_lanes: u32,
    /// SIMD lanes cooperating on one quantization group.
    pub(super) group_lanes: u32,
}

const SMALL_G13_HUGE_N: u32 = 32768;
const SMALL_G13_BROAD_ROW_N: u32 = 6144;
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
        input_row_tile: 1,
        reduction_lanes: 32,
        group_lanes: 1,
    }
}

pub(super) const DEFAULT_TILE: GemvTile = tile(DEFAULT_NUM_SIMDGROUPS, 1, DEFAULT_RESULTS_PER_SIMDGROUP);

impl GemvTile {
    pub const fn quantized(
        num_simdgroups: u32,
        results_per_simdgroup: u32,
        input_row_tile: u32,
        reduction_lanes: u32,
        group_lanes: u32,
    ) -> Self {
        Self {
            num_simdgroups,
            k_split: 1,
            results_per_simdgroup,
            input_row_tile,
            reduction_lanes,
            group_lanes,
        }
    }

    pub const fn quantized_output_tile(
        num_simdgroups: u32,
        output_row_tile: u32,
        input_row_tile: u32,
        reduction_lanes: u32,
        group_lanes: u32,
    ) -> Self {
        assert!(output_row_tile.is_multiple_of(num_simdgroups));
        Self::quantized(num_simdgroups, output_row_tile / num_simdgroups, input_row_tile, reduction_lanes, group_lanes)
    }

    pub(super) const fn output_row_tile(self) -> u32 {
        (self.num_simdgroups / self.k_split) * self.results_per_simdgroup
    }

    pub(super) const fn rows_per_lane(self) -> u32 {
        self.results_per_simdgroup / (32 / self.reduction_lanes)
    }
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
pub(super) fn fp_tile(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
    profile: DeviceProfile,
) -> GemvTile {
    let size = profile.size();
    let gpu_family = profile.gpu_family();
    let is_small_legacy = size == DeviceSize::Small && gpu_family == GpuFamily::Legacy;
    // SG8 is the portable full-precision geometry.
    let should_disable_k_split = !input_aligned
        || (m == 1 && size == DeviceSize::Large && k < FP_LARGE_SPLIT_K_MIN_DEPTH)
        || (m == 1 && is_small_legacy && n >= SMALL_G13_HUGE_N);

    let k_split = if should_disable_k_split {
        1
    } else {
        cap_k_split_to_complete_fp_k_blocks(k, preferred_fp_k_split(m, n, k))
    };

    let results_per_simdgroup = if is_small_legacy && m == 1 && n >= SMALL_G13_BROAD_ROW_N {
        DEFAULT_RESULTS_PER_SIMDGROUP
    } else if m == 1 && (k <= DEEP_K || size != DeviceSize::Large) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    };

    tile(DEFAULT_NUM_SIMDGROUPS, k_split, results_per_simdgroup)
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/kernel/matmul/gemv/policy_test.rs"]
mod tests;
