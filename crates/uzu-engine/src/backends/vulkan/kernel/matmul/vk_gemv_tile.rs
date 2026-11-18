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
        let lane_run = if bits == 4 {
            16
        } else {
            8
        };
        let lanes = matches!(group, 16 | 32 | 64 | 128)
            && (bits != 4 || k.is_multiple_of(2))
            && (!gathered || (m <= 8 && n >= DEFAULT_RESULTS_PER_SIMDGROUP));
        let mut tile = if !lanes {
            Self {
                lane_layout: false,
                ..Self::new(8, 1, 1)
            }
        } else if bf16_io
            && m == 1
            && b_layout == GemmBLayout::Tiled
            && !gathered
            && (group / 2).is_multiple_of(lane_run)
        {
            Self::new(4, 2, 2)
        } else {
            let mut tile = Self::new(DEFAULT_NUM_SIMDGROUPS, DEFAULT_RESULTS_PER_SIMDGROUP, group / lane_run);
            if bf16_io && !gathered && (2..=GEMV_INPUT_ROW_TILE).contains(&m) {
                tile.input_row_tile = m;
                tile.results_per_simdgroup = (16 / m.next_power_of_two()).min(4);
            }
            tile
        };
        tile.input_aligned = tile.lane_layout && k.is_multiple_of(group);
        tile.full_tile =
            tile.lane_layout && n.is_multiple_of(tile.output_row_tile()) && m.is_multiple_of(tile.input_row_tile);
        tile
    }
}
