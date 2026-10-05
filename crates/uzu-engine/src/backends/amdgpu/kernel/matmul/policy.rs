//! GEMV tile policy for AMD GPUs. The tile geometry is the Metal backend's (the kernel's VARIANTS and
//! CONSTRAINTs are shared); the selection is a first, untuned cut: Apple-family branches use their
//! "small GPU, Apple9" defaults, and there is no batch limit — without matrix cores every M goes
//! through GEMV (one input row per threadgroup in full precision).

// Full-precision GEMV accumulates four K values per SIMD lane, so one full vectorized K block is
// 4 * 32 lanes.
pub(super) const FP_K_BLOCK: u32 = 128;
const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
const DEFAULT_NUM_SIMDGROUPS: u32 = 8;
const DEEP_K: u32 = 8192;
const FP_K_DEPTH_N_MAX: u32 = 4095;
const FP_K_DEPTH_DEEP_MIN: u32 = 3072;
const FP_K_DEPTH_VERY_DEEP_RATIO: u32 = 16;

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

impl GemvTile {
    pub(super) const fn output_row_tile(self) -> u32 {
        (self.num_simdgroups / self.k_split) * self.results_per_simdgroup
    }

    pub(super) const fn rows_per_lane(self) -> u32 {
        self.results_per_simdgroup / (32 / self.reduction_lanes)
    }
}

const fn fp_tile_with(
    k_split: u32,
    results_per_simdgroup: u32,
) -> GemvTile {
    GemvTile {
        num_simdgroups: DEFAULT_NUM_SIMDGROUPS,
        k_split,
        results_per_simdgroup,
        input_row_tile: 1,
        reduction_lanes: 32,
        group_lanes: 1,
    }
}

fn cap_k_split_to_complete_fp_k_blocks(
    k: u32,
    preferred: u32,
) -> u32 {
    // K_SPLIT variants are powers of two; do not split beyond the complete K blocks a slice can own.
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

/// Full-precision tile: `m` input vectors, `n` output rows, reduction depth `k`.
pub(super) fn fp_tile(
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
) -> GemvTile {
    let k_split = if input_aligned {
        cap_k_split_to_complete_fp_k_blocks(k, preferred_fp_k_split(m, n, k))
    } else {
        1
    };
    // one output row per SIMD group also covers n < 4, which Metal sends to GEMM
    let results_per_simdgroup = if n < DEFAULT_RESULTS_PER_SIMDGROUP || (m == 1 && k <= DEEP_K) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    };
    fp_tile_with(k_split, results_per_simdgroup)
}

/// `UZU_AMDGPU_GEMV_ROW_TILE` caps the input rows per quantized tile (1 = single-row path), for
/// tuning; default 8. With every function inlined (no scratch) the multi-row tiles beat the single-row
/// GEMV 2.1-2.6x on the 890M (gate_up, W4 ZP G32: 6.5 vs 13.9 ms at M = 16, 3.2 vs 8.2 at M = 8); with
/// the default inlining they were 3x slower.
fn max_input_row_tile() -> u32 {
    static VALUE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("UZU_AMDGPU_GEMV_ROW_TILE").ok().and_then(|value| value.parse().ok()).unwrap_or(8).clamp(1, 8)
    })
}

/// Multi-row quantized tile: one threadgroup reads a weight tile once for up to 8 input rows (the
/// Apple M1/M2 qmv routes for M = 2..8, used here for every M > 1; tree verification and drafts run
/// M = 16). `gemv.metal` compiles these only for W4 zero-point / W8 symmetric, group 32 or 64,
/// bf16 activations and output, K aligned to the quantized block, no gather.
pub(super) fn multi_row_quantized_tile(
    m: u32,
    bits: u32,
    group: u32,
    zero_point: bool,
    symmetric: bool,
    bf16_io: bool,
    input_aligned: bool,
    gathered: bool,
) -> Option<GemvTile> {
    let format = (bits == 4 && zero_point) || (bits == 8 && symmetric);
    let max_rows = max_input_row_tile();
    if m < 2 || max_rows < 2 || !format || !matches!(group, 32 | 64) || !bf16_io || !input_aligned || gathered {
        return None;
    }
    let m = m.min(max_rows);
    let tile = |input_rows: u32, output_rows: u32| GemvTile {
        num_simdgroups: 2,
        k_split: 1,
        results_per_simdgroup: output_rows / 2,
        input_row_tile: input_rows,
        reduction_lanes: 8,
        group_lanes: 1,
    };
    Some(match m {
        2..=7 => tile(m, 16),
        _ if bits == 4 && group == 32 => tile(8, 8),
        _ => tile(7, 16),
    })
}

/// Lane-sliced quantized tile: each quantization group is split across reduction lanes; a lane
/// stages 8 bytes (16 W4 or 8 W8 values). One input row per threadgroup, so any batch size works
/// and gathered rows need no shared input tile.
pub(super) fn quantized_tile(
    bits: u32,
    group: u32,
    n: u32,
    bf16_io: bool,
) -> Option<GemvTile> {
    if !matches!(bits, 4 | 8) || !matches!(group, 16 | 32 | 64 | 128) {
        return None;
    }
    let pack_factor = if bits == 4 {
        8
    } else {
        4
    };
    // gemv.metal compiles the other single-row geometries only for W4 with bf16 activations and output
    let (num_simdgroups, results_per_simdgroup) = if bits == 4 && bf16_io {
        single_row_geometry()
    } else {
        (8, DEFAULT_RESULTS_PER_SIMDGROUP)
    };
    let tile = GemvTile {
        num_simdgroups,
        k_split: 1,
        results_per_simdgroup,
        input_row_tile: 1,
        reduction_lanes: 32,
        group_lanes: group / (2 * pack_factor),
    };
    (n >= tile.rows_per_lane()).then_some(tile)
}

/// Single-row quantized tile geometry: 8 SIMD groups x 4 output rows. Its 32-row tile carries the output
/// random Hadamard transform in the epilogue (`output_transform_for_tile`), which saves a separate
/// ActivationTransform dispatch per matmul (~128 per 9B decode token). On the 890M (W4 ZP G32, 9B shapes,
/// M = 1, coarse-grained buffers) 8x4 and 8x2 read weights equally fast (gate_up 0.72 ms, ~81 GB/s); before
/// the inlining fix 8x4 spilled (120 VGPRs, 140 B of scratch per lane) and lost to 8x2 (41 against 51 GB/s).
/// `UZU_AMDGPU_GEMV_GEOMETRY=<simdgroups>x<results>` overrides it for tuning.
fn single_row_geometry() -> (u32, u32) {
    static VALUE: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("UZU_AMDGPU_GEMV_GEOMETRY")
            .ok()
            .and_then(|value| {
                let (simdgroups, results) = value.split_once('x')?;
                Some((simdgroups.trim().parse().ok()?, results.trim().parse().ok()?))
            })
            .unwrap_or((8, 4))
    })
}
