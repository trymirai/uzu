use super::gpu_types::gemm_b_layout::GemmBLayout;

// Full-precision GEMV accumulates four K values per SIMD lane, so one full
// vectorized K block is 4 * 32 lanes.
const FP_K_BLOCK: u32 = 128;
const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
const DEFAULT_NUM_SIMDGROUPS: u32 = 8;
/// The quantized kernel's GEMV_MAX_INPUT_ROWS.
const MAX_INPUT_ROW_TILE: u32 = 16;
pub const NVIDIA_VENDOR_ID: u32 = 0x10de;
/// Output rows up to which an NVIDIA single-row U4 group-64 GEMV takes 8-row workgroups.
const NVIDIA_NARROW_N: u32 = 8192;

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

impl VkGemvTile {
    const fn new(
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

/// Lane-sliced tiles split each quantization group across reduction lanes.
/// A lane stages 8 bytes: 16 W4 values or 8 W8 values.
const fn lane_default(
    bits: u32,
    group: u32,
) -> VkGemvTile {
    let pack_factor = match bits {
        4 => 8,
        8 => 4,
        _ => 1,
    };
    VkGemvTile::new(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP, 1, 32, group / (2 * pack_factor))
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

impl VkGemvTile {
    /// The reader tile for shapes outside the lane layout: 8 outputs of one row per workgroup of 8 simdgroups.
    pub fn reader() -> Self {
        let mut tile = Self::new(8, 1, 1, 32, 1);
        tile.metal_layout = false;
        tile
    }

    pub fn dense(
        n: u32,
        k: u32,
    ) -> Self {
        let mut selected = Self::new(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP, 1, 32, 1);
        selected.input_aligned = k.is_multiple_of(FP_K_BLOCK);
        selected.full_tile = n.is_multiple_of(selected.output_row_tile());
        selected
    }

    pub fn quantized_shape(
        vendor_id: u32,
        bits: u32,
        group: u32,
        m: u32,
        n: u32,
        k: u32,
        bf16_io: bool,
        gathered: bool,
        b_layout: GemmBLayout,
    ) -> Self {
        let selected = if bits == 4 && !k.is_multiple_of(2) {
            None
        } else if gathered {
            gathered_tile(bits, group, m, n)
        } else if matches!(group, 16 | 32 | 64 | 128) {
            // A BF16 tile takes every input row at once, so each weight load serves all of them; fewer output rows per
            // lane bound the accumulators to 16 per lane.
            let mut tile = lane_default(bits, group);
            if bf16_io && (2..=MAX_INPUT_ROW_TILE).contains(&m) {
                tile.input_row_tile = m;
                tile.results_per_simdgroup = (16 / m.next_power_of_two()).clamp(1, DEFAULT_RESULTS_PER_SIMDGROUP);
            } else if vendor_id == NVIDIA_VENDOR_ID
                && bf16_io
                && m == 1
                && bits == 4
                && group == 64
                && (n <= NVIDIA_NARROW_N || b_layout == GemmBLayout::Tiled)
            {
                // The smaller tile also limits strided accesses when GEMM has packed B into 32-row tiles.
                tile = Self::new(4, 2, 1, 32, 2);
            }
            Some(tile)
        } else {
            None
        };
        let mut selected = selected.unwrap_or_else(Self::reader);
        selected.input_aligned = selected.metal_layout && k.is_multiple_of(group);
        selected.full_tile = selected.metal_layout
            && n.is_multiple_of(selected.output_row_tile())
            && m.is_multiple_of(selected.input_row_tile);
        selected
    }
}
