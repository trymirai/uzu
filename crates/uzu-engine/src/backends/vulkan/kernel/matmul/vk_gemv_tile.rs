use crate::backends::vulkan::gpu_types::{gemm::GEMV_INPUT_ROW_TILE, gemm_b_layout::GemmBLayout};

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

    /// The tiles that take this quantized shape, the default first and the reader last: the general tile, lane-sliced
    /// over the group or every input row at once for 2 to GEMV_INPUT_ROW_TILE rows of BF16 or FP16, and the narrow tile
    /// of 4 simdgroups by 2 rows where its two group lanes hold whole runs of 16 U4 or 8 U8 values. The narrow tile
    /// leads for one BF16 or FP16 row from B packed in 32-row tiles, where fewer rows a workgroup stride less.
    pub fn quantized_tiles(
        bits: u32,
        group: u32,
        m: u32,
        n: u32,
        k: u32,
        bf16_io: bool,
        gathered: bool,
        b_layout: GemmBLayout,
    ) -> impl Iterator<Item = Self> + Clone {
        let general = if bits == 4 && !k.is_multiple_of(2) {
            None
        } else if gathered {
            gathered_tile(bits, group, m, n)
        } else if matches!(group, 16 | 32 | 64 | 128) {
            // A BF16 tile takes every input row at once, so each weight load serves all of them; fewer output rows per
            // lane bound the accumulators to 16 per lane.
            let mut tile = lane_default(bits, group);
            if bf16_io && (2..=GEMV_INPUT_ROW_TILE).contains(&m) {
                tile.input_row_tile = m;
                tile.results_per_simdgroup = (16 / m.next_power_of_two()).clamp(1, DEFAULT_RESULTS_PER_SIMDGROUP);
            }
            Some(tile)
        } else {
            None
        };
        let lane_run = if bits == 4 {
            16
        } else {
            8
        };
        let narrow =
            (general.is_some() && !gathered && (group / 2).is_multiple_of(lane_run)).then(|| Self::new(4, 2, 1, 32, 2));
        let tiles = match bf16_io && m == 1 && b_layout == GemmBLayout::Tiled {
            true => [narrow, general],
            false => [general, narrow],
        };
        tiles.into_iter().flatten().chain([Self::reader()]).map(move |tile| tile.finished(group, m, n, k))
    }

    /// Why this tile breaks Gemv's `new` preconditions for B of `bits` in groups of `group`, `quantized` or not, and
    /// `gathered` rows; it validates every field before the derived output tile divides by them.
    pub fn contract(
        self,
        quantized: bool,
        bits: u32,
        group: u32,
        gathered: bool,
    ) -> Result<(), &'static str> {
        let require = |met: bool, reason: &'static str| met.then_some(()).ok_or(reason);
        let (simdgroups, split, lanes) = (self.num_simdgroups, self.k_split, self.reduction_lanes);
        require(
            [2, 4, 8].contains(&simdgroups)
                && [1, 2, 4, 8].contains(&split)
                && split <= simdgroups
                && (1..=8).contains(&self.results_per_simdgroup)
                && (1..=GEMV_INPUT_ROW_TILE).contains(&self.input_row_tile)
                && [8, 16, 32].contains(&lanes)
                && [1, 2, 4, 8, 16].contains(&self.group_lanes),
            "Gemv tile fields outside the ranges its entry point specializes",
        )?;
        let (rows, per_simdgroup) = (self.output_row_tile(), self.output_row_tile() / simdgroups);
        if !self.metal_layout {
            return require(
                rows == 8 && simdgroups == 8 && self.input_row_tile == 1,
                "the reader is 8 rows of 8 simdgroups",
            );
        }
        require(!gathered || self.input_row_tile == 1, "gather takes 1 input row")?;
        if !quantized {
            return require(
                simdgroups == 8
                    && self.input_row_tile == 1
                    && lanes == 32
                    && self.group_lanes == 1
                    && (rows == 8 / split || rows == 32 / split)
                    && (self.input_aligned || split == 1),
                "the full-precision lane tile is 8 simdgroups of 32 lanes over 8 or 32 rows split across K",
            );
        }
        let lane_run = if bits == 4 {
            16
        } else {
            8
        };
        require(
            split == 1
                && [16, 32, 64, 128].contains(&group)
                && (self.input_row_tile != 1 || lanes == 32)
                && lanes.is_multiple_of(self.group_lanes)
                && group.is_multiple_of(self.group_lanes)
                && (group / self.group_lanes).is_multiple_of(lane_run)
                && rows.is_multiple_of(simdgroups)
                && [1, 2, 4, 8].contains(&per_simdgroup)
                && per_simdgroup.is_multiple_of(32 / lanes),
            "the quantized lane tile's lanes, groups and rows break its entry point's contract",
        )
    }

    /// The tile with the K alignment and whole-tile flags the shape allows; neither applies outside the lane layout.
    fn finished(
        mut self,
        group: u32,
        m: u32,
        n: u32,
        k: u32,
    ) -> Self {
        self.input_aligned = self.metal_layout && k.is_multiple_of(group);
        self.full_tile =
            self.metal_layout && n.is_multiple_of(self.output_row_tile()) && m.is_multiple_of(self.input_row_tile);
        self
    }
}
