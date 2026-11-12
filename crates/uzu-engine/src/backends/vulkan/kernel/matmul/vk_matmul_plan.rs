use crate::{
    backends::{
        common::{
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::{
                ActivationQuantization,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{ActivationFormat, Int8CodeLayout, MatmulShape, QuantParamsLayout},
            },
        },
        vulkan::{
            VkDeviceCapabilities, VkPhysicalDevice,
            gpu_types::{gemm::INT8_FOLD_ENTRIES, gemm_b_layout::GemmBLayout, gemm_tiling::GemmTiling},
            kernel::matmul::VkGemvTile,
        },
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VkMatmulPlan {
    Gemv(VkGemvTile),
    GemvInt8 {
        rows: u32,
        input_rows: u32,
    },
    GemmInt8 {
        element_fold: bool,
        row_block: u32,
        simdgroups: u32,
        stage_depth: u32,
        prefetch: u32,
        splits: u32,
        k_slices: u32,
    },
    GemmTensor(GemmTiling),
    Gemm(GemmTiling),
}

impl VkMatmulPlan {
    pub fn baseline(
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: &VkPhysicalDevice,
    ) -> Self {
        let MatmulShape {
            m,
            n,
            k,
            ..
        } = *shape;
        let [weights, input, output] = data_types;
        let capabilities = device.capabilities;
        let units = device.compute_units;
        let max_shared = device.properties.limits.max_compute_shared_memory_size;
        if shape.a == ActivationFormat::Bf16
            && m <= 8
            && capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
            && Self::a8_quantization(shape, data_types, capabilities).is_some()
        {
            return Self::gemv_int8(m, shape.gathered, device.subgroup_properties.size);
        }
        if shape.a == ActivationFormat::Bf16 {
            let gemm_rows = Self::gemm_rows(shape);
            return if shape.gathered || m <= 8 || !gemm_rows {
                Self::Gemv(match (shape.is_quant(), gemm_rows) {
                    (false, true) => VkGemvTile::dense(n, k),
                    (false, false) => VkGemvTile::reader(),
                    (true, _) => VkGemvTile::quantized(shape, b_layout, input == DataType::BF16 && output == input),
                })
            } else if capabilities.float_tensor_cores(input) && input == weights {
                let tiling = [tensor_tiling(m), tensor_tiling(m.min(64))]
                    .into_iter()
                    .find(|tiling| 16 * tiling.tensor_staging() <= max_shared)
                    .expect("Vulkan 1.4 guarantees the shared memory of the 64-row tensor tiling");
                Self::GemmTensor(tiling)
            } else {
                Self::Gemm(scalar_tiling(m, n, k, shape.b_group_size))
            };
        }
        if m <= 8 || !capabilities.prefers_int8_gemm() {
            return Self::gemv_int8(m, shape.gathered, device.subgroup_properties.size);
        }
        let element_fold = capabilities.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD);
        let core = |row_block: u32, stage_depth: u32, prefetch: u32, splits: u32, k_slices: u32| Self::GemmInt8 {
            element_fold,
            row_block,
            simdgroups: if row_block == 64 || (row_block == 32 && element_fold) {
                8
            } else {
                4
            },
            stage_depth,
            prefetch,
            splits,
            k_slices,
        };
        let row_block = match m {
            0..=16 => 16,
            17..=32 => 32,
            _ if core(64, 64, 1, 1, 1).int8_shared_bytes(shape) <= max_shared => 64,
            _ => 32,
        };
        let tiles = n.div_ceil(64) * m.div_ceil(row_block);
        let deep_residency = if row_block == 64 {
            2
        } else {
            3
        };
        let deep = element_fold
            && (row_block != 64 || shape.b_bits == Some(4))
            && core(row_block, 128, 1, 1, 1).int8_shared_bytes(shape) <= max_shared
            && units.is_some_and(|units| tiles <= deep_residency * units);
        let stage_depth = if deep {
            128
        } else {
            64
        };
        let residency = if deep {
            deep_residency
        } else {
            5
        };
        let ring = deep && shape.b_bits == Some(4) && row_block <= 32;
        if ring && units.is_some_and(|units| tiles <= units) {
            let sliced = core(16, 64, 4, 1, 4);
            if row_block == 16 && k.is_multiple_of(512) && sliced.int8_shared_bytes(shape) <= max_shared {
                return sliced;
            }
            if k.is_multiple_of(256) {
                return core(row_block, stage_depth, 4, 2, 1);
            }
        }
        let prefetch = if ring && row_block == 16 && shape.b_group_size != Some(32) {
            4
        } else {
            1
        };
        core(row_block, stage_depth, prefetch, units.map_or(1, |units| int8_splits(tiles, k, residency, units)), 1)
    }

    pub fn check(
        &self,
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: &VkPhysicalDevice,
    ) -> Result<(), &'static str> {
        let [weights, input, _] = data_types;
        let MatmulShape {
            m,
            n,
            k,
            ..
        } = *shape;
        let limits = &device.properties.limits;
        let require = |met: bool, reason: &'static str| met.then_some(()).ok_or(reason);
        require(m > 0 && n > 0 && k > 0, "matmul dimensions must be positive")?;
        require(
            !shape.is_quant() || shape.b_group_size.is_some_and(|group| group > 0),
            "quantized B needs a positive group size",
        )?;
        require(shape.b_bits.is_none_or(|bits| matches!(bits, 4 | 8)), "quantized B has 4- or 8-bit codes")?;
        let gemm = matches!(self, Self::GemmTensor(_) | Self::Gemm(_));
        let a8 = matches!(self, Self::GemvInt8 { .. } | Self::GemmInt8 { .. });
        require(
            !a8 || Self::a8_shape(shape),
            "A8 takes U4, or U8 symmetric, groups of 32 to 128 over whole 128-value K blocks",
        )?;
        require(
            b_layout == GemmBLayout::RowMajor || (!gemm && shape.is_quant() && n % 32 == 0 && k % 128 == 0),
            "tiled B is whole 32 x 128 tiles of codes, which only Gemv and the A8 kernels read",
        )?;
        let shared = match *self {
            Self::GemmInt8 {
                ..
            } => {
                require(!shape.gathered, "INT8 GEMM stages contiguous B rows, without gather")?;
                self.int8_shared_bytes(shape)
            },
            Self::Gemm(tiling) => {
                let bytes = |left: bool, size: u32| tiling.scalar_staging(left, size) * size;
                bytes(true, input.size_in_bytes() as u32) + bytes(false, weights.size_in_bytes() as u32)
            },
            _ => 0,
        };
        require(shared <= limits.max_compute_shared_memory_size, "the staging exceeds the device's shared memory")?;
        let transform = (shape.d_transform.contains(GemmDTransform::RHT) && !self.fuses_output_rht()).then_some([
            n.div_ceil(128),
            m,
            1,
        ]);
        let counts = limits.max_compute_work_group_count;
        require(
            [Some(self.groups(m, n)), transform]
                .into_iter()
                .flatten()
                .all(|groups| groups.iter().zip(counts).all(|(&count, limit)| count <= limit)),
            "a dispatch's workgroups exceed maxComputeWorkGroupCount",
        )
    }

    fn groups(
        &self,
        m: u32,
        n: u32,
    ) -> [u32; 3] {
        match *self {
            Self::Gemv(tile) => [n.div_ceil(tile.output_row_tile()), m.div_ceil(tile.input_row_tile), 1],
            Self::GemvInt8 {
                rows,
                input_rows,
                ..
            } => [m.div_ceil(input_rows), n.div_ceil(8 * rows), 1],
            Self::GemmInt8 {
                row_block,
                splits,
                k_slices,
                ..
            } => [n.div_ceil(int8_block_n(k_slices)), m.div_ceil(row_block), splits],
            Self::GemmTensor(tiling) | Self::Gemm(tiling) => {
                [n.div_ceil(tiling.block_n()), m.div_ceil(tiling.block_m()), 1]
            },
        }
    }

    pub fn fuses_output_rht(&self) -> bool {
        match *self {
            Self::GemmInt8 {
                ..
            } => true,
            Self::GemvInt8 {
                rows,
                input_rows,
                ..
            } => input_rows == 1 && rows >= 4,
            _ => false,
        }
    }

    pub fn int8_shared_words(&self) -> u32 {
        match *self {
            Self::GemmInt8 {
                element_fold,
                row_block,
                stage_depth,
                k_slices,
                ..
            } => match element_fold {
                true => k_slices * 2 * (row_block + int8_block_n(k_slices)) * (stage_depth / 16 + 1),
                false => 2 * (row_block + 64) * 5 + 4 * 64 * (row_block / 16),
            },
            _ => 0,
        }
    }

    fn int8_shared_bytes(
        &self,
        shape: &MatmulShape,
    ) -> u32 {
        let parts = match *self {
            Self::GemmInt8 {
                element_fold: false,
                ..
            } => 0,
            Self::GemmInt8 {
                k_slices,
                ..
            } if k_slices > 1 => k_slices * (1 + u32::from(shape.b_group_size == Some(32))),
            Self::GemmInt8 {
                stage_depth: 128,
                ..
            } if shape.b_group_size == Some(32) => 4,
            _ => 2,
        };
        16 * self.int8_shared_words() + 4 * 2 * INT8_FOLD_ENTRIES * 2 * parts
    }

    pub fn a8_quantization(
        shape: &MatmulShape,
        [_, input, output]: [DataType; 3],
        capabilities: VkDeviceCapabilities,
    ) -> Option<ActivationQuantization> {
        let bits = shape.b_bits?;
        let group_size = shape.b_group_size?;
        let scale_bias = shape.b_prologue == GemmBPrologueKind::ScaleBiasDequant;
        let code_layout = Int8CodeLayout::for_right_bits(bits)?;
        let quantization =
            ActivationQuantization::new(ACTIVATION_SCALE_GROUP_SIZE, group_size, scale_bias, code_layout)?;
        (input == DataType::BF16
            && output == input
            && Self::a8_shape(shape)
            && match shape.params_layout {
                Some(QuantParamsLayout::GroupOutput) => true,
                Some(QuantParamsLayout::OutputGroup) => {
                    capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
                },
                None => false,
            }
            && shape.b_transpose
            && shape.b_leading_dimension.is_none())
        .then_some(quantization)
    }

    pub fn a8_rows(
        m: u32,
        b_layout: GemmBLayout,
        capabilities: VkDeviceCapabilities,
    ) -> bool {
        match b_layout {
            GemmBLayout::Tiled => m >= 2,
            GemmBLayout::RowMajor => {
                m >= 1
                    && (capabilities.contains(VkDeviceCapabilities::UNSIGNED_DOT_PRODUCT)
                        || (m >= 2 && (m <= 8 || capabilities.prefers_int8_gemm())))
            },
        }
    }

    pub fn tiles_weights(
        shape: &MatmulShape,
        data_types: [DataType; 3],
        capabilities: VkDeviceCapabilities,
    ) -> bool {
        capabilities.prefers_int8_gemm()
            && shape.n.is_multiple_of(32)
            && shape.k.is_multiple_of(128)
            && Self::a8_quantization(shape, data_types, capabilities).is_some()
    }

    fn gemv_int8(
        m: u32,
        gathered: bool,
        subgroup_size: u32,
    ) -> Self {
        Self::GemvInt8 {
            rows: match m {
                _ if subgroup_size > 64 => 8,
                0..=1 => 8,
                2 => 4,
                3..=4 => 2,
                _ => 1,
            },
            input_rows: if gathered {
                1
            } else {
                m.min(8)
            },
        }
    }

    fn a8_shape(shape: &MatmulShape) -> bool {
        let scheme = match shape.b_bits {
            Some(4) => shape.b_prologue != GemmBPrologueKind::FullPrecision,
            Some(8) => shape.b_prologue == GemmBPrologueKind::ScaleSymmetricDequant,
            _ => false,
        };
        scheme
            && shape.b_group_size.is_some_and(|group| matches!(group, 32 | 64 | 128))
            && shape.k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
    }

    fn gemm_rows(shape: &MatmulShape) -> bool {
        let MatmulShape {
            n,
            k,
            ..
        } = *shape;
        match shape.b_group_size {
            Some(group) => (shape.b_bits != Some(4) || k.is_multiple_of(2)) && group.is_multiple_of(16),
            None if shape.b_transpose => shape.b_leading_dimension.is_none_or(|leading| leading == k),
            None => k == 1 && shape.b_leading_dimension.unwrap_or(n) == 1,
        }
    }
}

fn int8_block_n(k_slices: u32) -> u32 {
    if k_slices > 1 {
        32
    } else {
        64
    }
}

fn tensor_tiling(m: u32) -> GemmTiling {
    match m {
        0..=16 => GemmTiling::Tile16x64x64_Simdgroups1x4x2,
        17..=32 => GemmTiling::Tile32x64x64_Simdgroups1x4x2,
        33..=64 => GemmTiling::Tile64x128x32_Simdgroups2x4x1,
        _ => GemmTiling::Tile128x128x32_Simdgroups2x4x1,
    }
}

fn scalar_tiling(
    m: u32,
    n: u32,
    k: u32,
    group: Option<u32>,
) -> GemmTiling {
    const SIMDGROUP_QUANT_SMALL_M_MAX: u32 = 32;
    const SIMDGROUP_QUANT_NARROW_BLOCK_M: u32 = 8;
    const SIMDGROUP_QUANT_LARGE_M_MIN: u32 = 64;
    const SIMDGROUP_QUANT_WIDE_N_MIN: u32 = 6144;
    let tiling = match group {
        None if 2 * m.max(n) > k => GemmTiling::Tile64x64x16_Simdgroups2x2,
        None => GemmTiling::Tile64x32x32_Simdgroups2x2,
        Some(group) if group < 32 => GemmTiling::Tile64x64x16_Simdgroups2x2,
        Some(_) if m < SIMDGROUP_QUANT_SMALL_M_MAX => {
            if m <= SIMDGROUP_QUANT_NARROW_BLOCK_M || m.is_multiple_of(SIMDGROUP_QUANT_NARROW_BLOCK_M) {
                GemmTiling::Tile8x32x32_Simdgroups1x1
            } else {
                GemmTiling::Tile32x32x32_Simdgroups2x2
            }
        },
        Some(_) if m >= SIMDGROUP_QUANT_LARGE_M_MIN && n >= SIMDGROUP_QUANT_WIDE_N_MIN && n.is_multiple_of(64) => {
            GemmTiling::Tile64x64x32_Simdgroups2x2
        },
        Some(_) => GemmTiling::Tile32x32x32_Simdgroups2x2,
    };
    match group {
        Some(group) if !group.is_multiple_of(tiling.block_k()) => GemmTiling::Tile64x64x16_Simdgroups2x2,
        _ => tiling,
    }
}

fn int8_splits(
    tiles: u32,
    k: u32,
    residency: u32,
    units: u32,
) -> u32 {
    const COLUMNS: u32 = 1536;
    const SHORT_COLUMNS: u32 = 512;
    let fits = |splits: u32| {
        let workgroups = tiles * splits;
        let columns = if workgroups / 2 <= units {
            SHORT_COLUMNS
        } else {
            COLUMNS
        };
        workgroups <= residency * units && k.is_multiple_of(128 * splits) && k / splits >= columns
    };
    [8, 4, 2].into_iter().find(|&splits| fits(splits)).unwrap_or(1)
}
