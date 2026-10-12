// Vulkan counterpart of Metal gemv/policy, with the same tile selection on Asahi.
const LARGE_MIN_GPU_CORES: u32 = 30;

// Full-precision GEMV accumulates four K values per SIMD lane, so one full
// vectorized K block is 4 * 32 lanes.
const FP_K_BLOCK: u32 = 128;
const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VkGemvTile {
    pub input_aligned: bool,
    pub full_tile: bool,
    pub metal_layout: bool,
    /// SIMD groups launched in one threadgroup.
    pub num_simdgroups: u32,
    /// Number of split-K slices reduced by one threadgroup.
    pub k_split: u32,
    /// Output rows computed by each SIMD group.
    pub results_per_simdgroup: u32,
    /// Input rows processed by one threadgroup.
    pub input_row_tile: u32,
    /// SIMD lanes cooperating over K for one output-row block.
    pub reduction_lanes: u32,
    /// SIMD lanes cooperating on one quantization group.
    pub group_lanes: u32,
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
) -> VkGemvTile {
    VkGemvTile {
        input_aligned: false,
        full_tile: false,
        metal_layout: true,
        num_simdgroups,
        k_split,
        results_per_simdgroup,
        input_row_tile: 1,
        reduction_lanes: 32,
        group_lanes: 1,
    }
}

impl VkGemvTile {
    const fn quantized(
        num_simdgroups: u32,
        results_per_simdgroup: u32,
        input_row_tile: u32,
        reduction_lanes: u32,
        group_lanes: u32,
    ) -> Self {
        Self {
            input_aligned: false,
            full_tile: false,
            metal_layout: true,
            num_simdgroups,
            k_split: 1,
            results_per_simdgroup,
            input_row_tile,
            reduction_lanes,
            group_lanes,
        }
    }

    pub const fn output_row_tile(self) -> u32 {
        (self.num_simdgroups / self.k_split) * self.results_per_simdgroup
    }

    const fn rows_per_lane(self) -> u32 {
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
fn fp_tile(
    gpu_core_count: u32,
    apple_gpu_family: u32,
    m: u32,
    n: u32,
    k: u32,
    input_aligned: bool,
) -> VkGemvTile {
    let is_large_gpu = gpu_core_count >= LARGE_MIN_GPU_CORES;
    let is_small_legacy = !is_large_gpu && apple_gpu_family < 8;
    // SG8 is the portable full-precision geometry.
    let should_disable_k_split = !input_aligned
        || (m == 1 && is_large_gpu && k < FP_LARGE_SPLIT_K_MIN_DEPTH)
        || (m == 1 && is_small_legacy && n >= SMALL_G13_HUGE_N);

    let k_split = if should_disable_k_split {
        1
    } else {
        cap_k_split_to_complete_fp_k_blocks(k, preferred_fp_k_split(m, n, k))
    };

    let results_per_simdgroup = if is_small_legacy && m == 1 && n >= SMALL_G13_BROAD_ROW_N {
        DEFAULT_RESULTS_PER_SIMDGROUP
    } else if m == 1 && (k <= DEEP_K || !is_large_gpu) {
        1
    } else {
        DEFAULT_RESULTS_PER_SIMDGROUP
    };

    tile(DEFAULT_NUM_SIMDGROUPS, k_split, results_per_simdgroup)
}

const QUANT_N_BUCKET_MAXES: [u32; 6] = [512, 2048, 4096, 8192, 16384, 32768];
const QUANT_K_BUCKET_MAXES: [u32; 3] = [512, 2048, 8192];

/// Lane-sliced tiles split each quantization group across reduction lanes.
/// A lane stages 8 bytes: 16 W4 values or 8 W8 values.
const fn lane_tile(
    num_simdgroups: u32,
    results_per_simdgroup: u32,
    bits: u32,
    group: u32,
) -> VkGemvTile {
    let pack_factor = match bits {
        4 => 8,
        8 => 4,
        _ => 1,
    };
    VkGemvTile::quantized(num_simdgroups, results_per_simdgroup, 1, 32, group / (2 * pack_factor))
}

const fn lane_default(
    bits: u32,
    group: u32,
) -> VkGemvTile {
    lane_tile(8, DEFAULT_RESULTS_PER_SIMDGROUP, bits, group)
}

fn gathered_tile(
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
) -> Option<VkGemvTile> {
    if !matches!(bits, 4 | 8) || !matches!(group, 16 | 32 | 64 | 128) || !(1..=8).contains(&m) {
        return None;
    }
    let tile = lane_default(bits, group);
    (n >= tile.rows_per_lane()).then_some(tile)
}

fn table_bucket_index(
    value: u32,
    bucket_maxes: &[u32],
) -> usize {
    bucket_maxes.partition_point(|&max| value > max)
}

fn lane_policy(
    gpu_core_count: u32,
    apple_gpu_family: u32,
    m: u32,
    n: u32,
    k: u32,
    bits: u32,
    group: u32,
) -> VkGemvTile {
    let is_large_gpu = gpu_core_count >= LARGE_MIN_GPU_CORES;
    if m != 1 || bits != 4 {
        return lane_default(bits, group);
    }

    let k_bucket = table_bucket_index(k, &QUANT_K_BUCKET_MAXES);
    let n_bucket = table_bucket_index(n, &QUANT_N_BUCKET_MAXES);
    let (num_simdgroups, results_per_simdgroup) = match (is_large_gpu, apple_gpu_family, k_bucket, n_bucket) {
        (true, _, 0, 1) => (4, 2),
        (true, _, 1, 0) => (2, 1),
        (true, _, 1, 1..=3) => (2, 2),
        (true, _, 1, 4) => (2, 1),
        (true, _, 1, 5) => (2, 4),
        (true, _, 2, 1) => (4, 2),
        (true, _, 3, 1) => (2, 2),
        (false, 9, 0, 1) => (4, 4),
        (false, 9, 1, 0) => (4, 2),
        (false, 9, 1, 1) => (2, 2),
        (false, 9, 1, 2) => (4, 2),
        (false, 9, 1, 4) => (2, 2),
        (false, 9, 1, 5) => (4, 2),
        (false, 9, 2, 1) => (4, 2),
        (false, 9, 3, 1) => (2, 2),
        (false, 8, 0, 1) => (4, 4),
        (false, 8, 1, _) | (false, 8, 2, 1) => (8, 2),
        (false, family, 0, 1) if family < 8 => (4, 8),
        (false, family, 1, 0..=3) if family < 8 => (8, 2),
        _ => (8, 4),
    };
    let selected = lane_tile(num_simdgroups, results_per_simdgroup, bits, group);
    if n < selected.results_per_simdgroup {
        lane_default(bits, group)
    } else {
        selected
    }
}

fn select(
    gpu_core_count: u32,
    apple_gpu_family: u32,
    bits: u32,
    group: u32,
    m: u32,
    n: u32,
    k: u32,
    bf16_io: bool,
) -> Option<VkGemvTile> {
    if !matches!(bits, 4 | 8) || !matches!(group, 16 | 32 | 64 | 128) || !(1..=8).contains(&m) {
        return None;
    }

    if !bf16_io {
        let tile = lane_default(bits, group);
        return (m <= 4 && n >= tile.rows_per_lane()).then_some(tile);
    }

    let tile = match m {
        1..=4 => lane_policy(gpu_core_count, apple_gpu_family, m, n, k, bits, group),
        5..=8 if n <= 64 => lane_default(bits, group),
        _ => return None,
    };
    // Input rows are tiled independently; partial tiles clamp loads and stores.
    if n < tile.rows_per_lane() {
        return None;
    }
    Some(tile)
}

impl VkGemvTile {
    pub fn dense(
        topology: Option<(u32, u32)>,
        m: u32,
        n: u32,
        k: u32,
        max_groups: u32,
    ) -> Self {
        let aligned = k.is_multiple_of(FP_K_BLOCK);
        let mut selected = match topology {
            Some((cores, family)) => fp_tile(cores, family, m, n, k, aligned),
            None => tile(DEFAULT_NUM_SIMDGROUPS, 1, DEFAULT_RESULTS_PER_SIMDGROUP),
        };
        // Vulkan limits the X dispatch dimension; reduce split K before widening the same Metal tile.
        while n.div_ceil(selected.output_row_tile()) > max_groups && selected.k_split > 1 {
            selected.k_split /= 2;
        }
        if n.div_ceil(selected.output_row_tile()) > max_groups {
            selected.results_per_simdgroup = DEFAULT_RESULTS_PER_SIMDGROUP;
        }
        selected.input_aligned = aligned;
        selected.full_tile = n.is_multiple_of(selected.output_row_tile());
        selected
    }

    pub fn quantized_shape(
        topology: Option<(u32, u32)>,
        bits: u32,
        group: u32,
        m: u32,
        n: u32,
        k: u32,
        bf16_io: bool,
        gathered: bool,
    ) -> Self {
        let selected = if bits == 4 && !k.is_multiple_of(2) {
            None
        } else if gathered {
            gathered_tile(bits, group, m, n)
        } else if let Some((cores, family)) = topology {
            select(cores, family, bits, group, m, n, k, bf16_io)
        } else if matches!(group, 16 | 32 | 64 | 128) {
            Some(lane_default(bits, group))
        } else {
            None
        };
        let mut selected = selected.unwrap_or_else(|| {
            let mut value = Self::quantized(8, 1, 1, 32, 1);
            value.metal_layout = false;
            value
        });
        selected.input_aligned = selected.metal_layout && k.is_multiple_of(group);
        selected.full_tile = selected.metal_layout
            && n.is_multiple_of(selected.output_row_tile())
            && m.is_multiple_of(selected.input_row_tile);
        selected
    }
}
