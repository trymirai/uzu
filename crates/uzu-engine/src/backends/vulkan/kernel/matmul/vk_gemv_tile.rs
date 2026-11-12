use crate::backends::{
    common::kernel::matmul::MatmulShape,
    vulkan::gpu_types::{gemm::GEMV_INPUT_ROW_TILE, gemm_b_layout::GemmBLayout},
};

const FP_K_BLOCK: u32 = 128;
const DEFAULT_RESULTS_PER_SIMDGROUP: u32 = 4;
const DEFAULT_NUM_SIMDGROUPS: u32 = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VkGemvTile {
    pub input_aligned: bool,
    pub full_tile: bool,
    pub lane_layout: bool,
    pub num_simdgroups: u32,
    pub results_per_simdgroup: u32,
    pub input_row_tile: u32,
    pub group_lanes: u32,
}

const fn lane_default(
    bits: u32,
    group: u32,
) -> VkGemvTile {
    let pack_factor = match bits {
        4 => 8,
        8 => 4,
        _ => 1,
    };
    VkGemvTile::new(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP, group / (2 * pack_factor))
}

impl VkGemvTile {
    const fn new(
        num_simdgroups: u32,
        results_per_simdgroup: u32,
        group_lanes: u32,
    ) -> Self {
        Self {
            input_aligned: false,
            full_tile: false,
            lane_layout: true,
            num_simdgroups,
            results_per_simdgroup,
            input_row_tile: 1,
            group_lanes,
        }
    }

    pub const fn output_row_tile(self) -> u32 {
        self.num_simdgroups * self.results_per_simdgroup
    }

    pub fn reader() -> Self {
        let mut tile = Self::new(8, 1, 1);
        tile.lane_layout = false;
        tile
    }

    pub fn dense(
        n: u32,
        k: u32,
    ) -> Self {
        let mut selected = Self::new(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP, 1);
        selected.input_aligned = k.is_multiple_of(FP_K_BLOCK);
        selected.full_tile = n.is_multiple_of(selected.output_row_tile());
        selected
    }

    pub fn quantized(
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        bf16_io: bool,
    ) -> Self {
        let MatmulShape {
            m,
            n,
            k,
            gathered,
            ..
        } = *shape;
        let bits = shape.b_bits.expect("quantized B has bits");
        let group = shape.b_group_size.expect("quantized B has a group size");
        let general = if bits == 4 && !k.is_multiple_of(2) {
            None
        } else if gathered {
            let tile = lane_default(bits, group);
            let valid = matches!(bits, 4 | 8) && matches!(group, 16 | 32 | 64 | 128) && (1..=8).contains(&m);
            (valid && n >= tile.results_per_simdgroup).then_some(tile)
        } else if matches!(group, 16 | 32 | 64 | 128) {
            let mut tile = lane_default(bits, group);
            if bf16_io && (2..=GEMV_INPUT_ROW_TILE).contains(&m) {
                tile.input_row_tile = m;
                tile.results_per_simdgroup = (16 / m.next_power_of_two()).min(4);
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
            bf16_io && m == 1 && b_layout == GemmBLayout::Tiled && !gathered && (group / 2).is_multiple_of(lane_run);
        let mut tile = match general {
            Some(_) if narrow => Self::new(4, 2, 2),
            Some(general) => general,
            None => Self::reader(),
        };
        tile.input_aligned = tile.lane_layout && k.is_multiple_of(group);
        tile.full_tile =
            tile.lane_layout && n.is_multiple_of(tile.output_row_tile()) && m.is_multiple_of(tile.input_row_tile);
        tile
    }
}
