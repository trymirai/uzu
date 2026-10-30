use super::VkGemvTile;
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
            VkDeviceCapabilities as Capability, VkPhysicalDevice,
            gpu_types::{gemm::INT8_FOLD_ENTRIES, gemm_b_layout::GemmBLayout, gemm_tiling::GemmTiling},
        },
    },
    data_type::DataType,
};

/// The fold metadata arrays every Int8Gemm core declares in shared memory.
const INT8_FOLD_BYTES: u32 = 4 * 2 * INT8_FOLD_ENTRIES;
/// One dispatch family of the matmul kernels with its specialization parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VkMatmulPlan {
    /// Gemv on full-precision A, with its tile.
    Gemv(VkGemvTile),
    /// GemvInt8 on A8: `8 * rows` outputs by `input_rows` inputs a workgroup of 256 invocations.
    GemvInt8 {
        rows: u32,
        input_rows: u32,
    },
    /// An INT8 GEMM core, GemmInt8Nv under `element_fold`: `row_block` rows by 64 columns a workgroup of `simdgroups`,
    /// staging `stage_depth` K values a step, with K in `splits` partial phases then a reduction phase where it splits.
    GemmInt8 {
        element_fold: bool,
        row_block: u32,
        simdgroups: u32,
        stage_depth: u32,
        splits: u32,
    },
    /// GemmTensor's cooperative core on BF16 A and B.
    GemmTensor(GemmTiling),
    /// Gemm's scalar core.
    Gemm(GemmTiling),
}

impl VkMatmulPlan {
    /// The plan for `shape` from weights in `b_layout` on `device`, by its capabilities and limits, never its identity.
    pub fn baseline(
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: &VkPhysicalDevice,
    ) -> Self {
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let [weights, input, output] = data_types;
        let (capabilities, units) = (device.capabilities, device.compute_units);
        let max_shared = device.limits().max_compute_shared_memory_size;
        // Where only unsigned dot products are accelerated, up to 8 rows of full-precision A quantize themselves.
        if shape.a == ActivationFormat::Bf16
            && m <= 8
            && capabilities.contains(Capability::UNSIGNED_DOT_PRODUCT)
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
                // The widest tiling whose staging fits the workgroup's shared memory.
                let tiling = [tensor_tiling(m), tensor_tiling(m.min(64))]
                    .into_iter()
                    .find(|tiling| 16 * tiling.tensor_staging() <= max_shared)
                    .expect("Vulkan 1.4 guarantees the shared memory of the 64-row tensor tiling");
                Self::GemmTensor(tiling)
            } else {
                Self::Gemm(scalar_tiling(m, n, k, shape.b_group_size))
            };
        }
        // Up to 8 rows, and on devices without INT8 tensor cores or whose unsigned dot products are the fast ones,
        // GemvInt8.
        if m <= 8 || !capabilities.prefers_int8_gemm() {
            return Self::gemv_int8(m, shape.gathered, device.subgroup_properties.size);
        }
        let element_fold = capabilities.contains(Capability::INT8_ELEMENT_FOLD);
        let core = |row_block: u32, stage_depth: u32, splits: u32| Self::GemmInt8 {
            element_fold,
            row_block,
            simdgroups: if row_block == 64 || (row_block == 32 && element_fold) {
                8
            } else {
                4
            },
            stage_depth,
            splits,
        };
        let row_block = match m {
            0..=16 => 16,
            17..=32 => 32,
            _ if core(64, 64, 1).int8_shared_bytes() <= max_shared => 64,
            _ => 32,
        };
        // GemmInt8Nv's 16-row tiles stage 128 deep on grids of at most a tile a unit, three workgroups a unit.
        let tiles = u64::from(n.div_ceil(64)) * u64::from(m.div_ceil(row_block));
        let deep = element_fold
            && row_block == 16
            && shape.b_group_size.is_some_and(|group| group.is_multiple_of(64))
            && units.is_some_and(|units| tiles <= u64::from(units));
        let (stage_depth, residency) = if deep {
            (128, 3)
        } else {
            (64, 5)
        };
        core(row_block, stage_depth, units.map_or(1, |units| int8_splits(tiles, k, residency, units)))
    }

    /// Why this plan cannot run `shape` from weights in `b_layout`, checked before any pipeline is created or command
    /// recorded: the caller's A format, B layout and gather against the family, then the device's shared memory,
    /// workgroup and subgroup limits. The kernels check their own specializations when created.
    pub fn check(
        &self,
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: &VkPhysicalDevice,
    ) -> Result<(), &'static str> {
        let [weights, input, _] = data_types;
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let (capabilities, limits, subgroups) = (device.capabilities, device.limits(), &device.subgroup_properties);
        let require = |met: bool, reason: &'static str| met.then_some(()).ok_or(reason);
        require(m > 0 && n > 0 && k > 0, "matmul dimensions must be positive")?;
        require(!shape.is_quant() || shape.b_group_size > Some(0), "quantized B needs a positive group size")?;
        let gemm = matches!(self, Self::GemmTensor(_) | Self::Gemm(_));
        let a8 = matches!(self, Self::GemvInt8 { .. } | Self::GemmInt8 { .. });
        // GemvInt8 also takes full-precision A, which the encoder quantizes first.
        let full_precision = shape.a == ActivationFormat::Bf16;
        let quantizes_a = full_precision && matches!(self, Self::GemvInt8 { .. });
        require(
            a8 != full_precision || quantizes_a,
            "A8 takes GemvInt8 or GemmInt8, full precision GemvInt8 or the other families",
        )?;
        require(
            !quantizes_a || Self::a8_quantization(shape, data_types, capabilities).is_some(),
            "GemvInt8 quantizes full-precision A only within the A8 contract",
        )?;
        require(
            !a8 || Self::a8_shape(shape),
            "A8 takes U4, or U8 symmetric, groups of 32 to 128 over whole 128-value K blocks",
        )?;
        require(
            b_layout == GemmBLayout::RowMajor || (!gemm && shape.is_quant() && n % 32 == 0 && k % 128 == 0),
            "tiled B is whole 32 x 128 tiles of codes, which only Gemv and the A8 kernels read",
        )?;
        require(!gemm || (!shape.gathered && Self::gemm_rows(shape)), "Gemm stages contiguous B rows, without gather")?;
        // Gemv and GemvInt8 stage fixed arrays, and the baseline picks GemmTensor's tiling by its staging.
        let shared = match *self {
            Self::GemmInt8 {
                ..
            } => {
                require(!shape.gathered, "INT8 GEMM stages contiguous B rows, without gather")?;
                self.int8_shared_bytes()
            },
            Self::Gemm(tiling) => {
                let bytes = |left: bool, size: u32| tiling.scalar_staging(left, size) * size;
                bytes(true, input.size_in_bytes() as u32) + bytes(false, weights.size_in_bytes() as u32)
            },
            _ => 0,
        };
        require(shared <= limits.max_compute_shared_memory_size, "the staging exceeds the device's shared memory")?;
        let ([x, y], required) = self.workgroup();
        let size = limits.max_compute_work_group_size;
        require(
            x <= size[0] && y <= size[1] && x * y <= limits.max_compute_work_group_invocations,
            "the workgroup exceeds the device's dimensions or invocations",
        )?;
        require(
            required.is_none_or(|lanes| {
                subgroups.requirable(lanes)
                    && x % lanes == 0
                    && x * y <= lanes.saturating_mul(subgroups.max_compute_workgroup_subgroups)
            }),
            "the workgroup is not whole required subgroups within maxComputeWorkgroupSubgroups",
        )?;
        // The family's dispatches, then the output transform the encoder appends where the epilogue does not fuse it:
        // ActivationTransform's 128-value tiles by rows.
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

    /// The largest workgroup grid of the family's dispatches for m x n outputs, as the entry points declare it: split
    /// K's partial phase spans the splits, its reduction phase one.
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
                ..
            } => [n.div_ceil(64), m.div_ceil(row_block), splits],
            Self::GemmTensor(tiling) | Self::Gemm(tiling) => {
                [n.div_ceil(tiling.block_n()), m.div_ceil(tiling.block_m()), 1]
            },
        }
    }

    /// The workgroup's invocations along x and y and the subgroup size its pipeline requires, as the entry points
    /// declare them: `[32, simdgroups]`, or 256 for GemvInt8.
    fn workgroup(&self) -> ([u32; 2], Option<u32>) {
        match *self {
            Self::Gemv(tile) => ([32, tile.num_simdgroups], None),
            Self::GemvInt8 {
                ..
            } => ([256, 1], None),
            Self::GemmInt8 {
                simdgroups,
                ..
            } => ([32, simdgroups], Some(32)),
            Self::GemmTensor(tiling) => ([32, tiling.simdgroups()], Some(32)),
            Self::Gemm(tiling) => ([32, tiling.simdgroups()], None),
        }
    }

    /// Whether the plan's epilogue applies the output transform: every INT8 GEMM core, and GemvInt8 of one input row
    /// whose workgroups hold whole 32-row blocks.
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

    /// The 16-byte shared words of an INT8 GEMM core's staging: GemmInt8Nv folds weight groups in registers, the other
    /// cores also stage their fold scratch. Zero for the other families.
    pub fn int8_shared_words(&self) -> u32 {
        match *self {
            Self::GemmInt8 {
                element_fold,
                row_block,
                stage_depth,
                ..
            } => match element_fold {
                true => 2 * (row_block + 64) * (stage_depth / 16 + 1),
                false => 2 * (row_block + 64) * 5 + 4 * 64 * (row_block / 16),
            },
            _ => 0,
        }
    }

    /// The shared memory of an INT8 GEMM core's workgroup: its staging and the fold metadata of every core.
    fn int8_shared_bytes(&self) -> u32 {
        16 * self.int8_shared_words() + INT8_FOLD_BYTES
    }

    /// The activation quantization the A8 kernels take for `shape`'s B, whatever its A: BF16 activations and
    /// output, transposed contiguous B within [`Self::a8_shape`], and group-output metadata or output-group metadata
    /// on unsigned-dot devices.
    pub fn a8_quantization(
        shape: &MatmulShape,
        [_, input, output]: [DataType; 3],
        capabilities: Capability,
    ) -> Option<ActivationQuantization> {
        let (bits, group_size) = (shape.b_bits?, shape.b_group_size?);
        let scale_bias = shape.b_prologue == GemmBPrologueKind::ScaleBiasDequant;
        let code_layout = Int8CodeLayout::for_right_bits(bits)?;
        let quantization =
            ActivationQuantization::new(ACTIVATION_SCALE_GROUP_SIZE, group_size, scale_bias, code_layout)?;
        (input == DataType::BF16
            && output == input
            && Self::a8_shape(shape)
            && match shape.params_layout {
                Some(QuantParamsLayout::GroupOutput) => true,
                Some(QuantParamsLayout::OutputGroup) => capabilities.contains(Capability::UNSIGNED_DOT_PRODUCT),
                None => false,
            }
            && shape.b_transpose
            && shape.b_leading_dimension.is_none())
        .then_some(quantization)
    }

    /// Whether callers give m rows of A as A8 from weights in `b_layout` that take it: from one row on unsigned-dot
    /// devices, otherwise from 2 to 8 rows and, where the device prefers INT8 GEMM, beyond; tiled weights from 2 rows.
    pub fn a8_rows(
        m: u32,
        b_layout: GemmBLayout,
        capabilities: Capability,
    ) -> bool {
        match b_layout {
            GemmBLayout::Tiled => m >= 2,
            GemmBLayout::RowMajor => {
                m >= 1
                    && (capabilities.contains(Capability::UNSIGNED_DOT_PRODUCT)
                        || (m >= 2 && (m <= 8 || capabilities.prefers_int8_gemm())))
            },
        }
    }

    /// Whether loaded weights of the A8 `shape` are packed into tiles: whole 32 x 128 tiles of codes within the A8
    /// contract, on devices that prefer INT8 GEMM; unsigned-dot devices keep rows for GEMV.
    pub fn tiles_weights(
        shape: &MatmulShape,
        data_types: [DataType; 3],
        capabilities: Capability,
    ) -> bool {
        capabilities.prefers_int8_gemm()
            && shape.n.is_multiple_of(32)
            && shape.k.is_multiple_of(128)
            && Self::a8_quantization(shape, data_types, capabilities).is_some()
    }

    /// GemvInt8 for m rows: more outputs a workgroup for fewer inputs, one input row a workgroup where `gathered`.
    /// Subgroups wider than 64 lanes take 8 rows for every input count: their reductions run through shared memory,
    /// where split rows measured faster.
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
                m.clamp(1, 8)
            },
        }
    }

    /// Whether the A8 kernels take the shape: B of U4 codes under any scheme or U8 symmetric ones, in groups of 32, 64 or
    /// 128, over whole 128-value activation scale groups of K. Only GemvInt8 gathers, one input row a workgroup.
    pub fn a8_shape(shape: &MatmulShape) -> bool {
        let scheme = match shape.b_bits {
            Some(4) => shape.b_prologue != GemmBPrologueKind::FullPrecision,
            Some(8) => shape.b_prologue == GemmBPrologueKind::ScaleSymmetricDequant,
            _ => false,
        };
        scheme
            && shape.b_group_size.is_some_and(|group| matches!(group, 32 | 64 | 128))
            && shape.k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
    }

    /// Whether Gemm's loaders can stage B: contiguous full-precision rows, or quantized rows of whole 16-value runs of a
    /// group from byte-aligned U4 rows.
    pub fn gemm_rows(shape: &MatmulShape) -> bool {
        let (n, k) = (shape.n, shape.k);
        match shape.b_group_size {
            Some(group) => (shape.b_bits != Some(4) || k.is_multiple_of(2)) && group.is_multiple_of(16),
            // The row and column strides are (leading, 1) transposed and (1, leading) otherwise.
            None => match shape.b_leading_dimension.unwrap_or(if shape.b_transpose {
                k
            } else {
                n
            }) {
                leading if shape.b_transpose => leading == k,
                leading => k == 1 && leading == 1,
            },
        }
    }
}

/// The cooperative core's block for m rows: one row block up to 64 rows, so each weight is read once.
fn tensor_tiling(m: u32) -> GemmTiling {
    match m {
        0..=16 => GemmTiling::Tile16x64x64_Simdgroups1x4x2,
        17..=32 => GemmTiling::Tile32x64x64_Simdgroups1x4x2,
        33..=64 => GemmTiling::Tile64x128x32_Simdgroups2x4x1,
        _ => GemmTiling::Tile128x128x32_Simdgroups2x4x1,
    }
}

/// The scalar core's tiling, `group` the quantization group, which must hold whole K slices of the tiling.
fn scalar_tiling(
    m: u32,
    n: u32,
    k: u32,
    group: Option<u32>,
) -> GemmTiling {
    use GemmTiling::*;
    // Metal's `simdgroup_quant_tile` thresholds (metal gemm/policy.rs), with its Apple8-and-older partial-M rule.
    const SIMDGROUP_QUANT_SMALL_M_MAX: u32 = 32;
    const SIMDGROUP_QUANT_NARROW_BLOCK_M: u32 = 8;
    const SIMDGROUP_QUANT_LARGE_M_MIN: u32 = 64;
    const SIMDGROUP_QUANT_WIDE_N_MIN: u32 = 6144;
    let tiling = match group {
        None if 2_u32.saturating_mul(m.max(n)) > k => Tile64x64x16_Simdgroups2x2,
        None => Tile64x32x32_Simdgroups2x2,
        Some(group) if group < 32 => Tile64x64x16_Simdgroups2x2,
        Some(_) if m < SIMDGROUP_QUANT_SMALL_M_MAX => {
            if m <= SIMDGROUP_QUANT_NARROW_BLOCK_M || m.is_multiple_of(SIMDGROUP_QUANT_NARROW_BLOCK_M) {
                Tile8x32x32_Simdgroups1x1
            } else {
                Tile32x32x32_Simdgroups2x2
            }
        },
        Some(_) if m >= SIMDGROUP_QUANT_LARGE_M_MIN && n >= SIMDGROUP_QUANT_WIDE_N_MIN && n.is_multiple_of(64) => {
            Tile64x64x32_Simdgroups2x2
        },
        Some(_) => Tile32x32x32_Simdgroups2x2,
    };
    match group {
        Some(group) if !group.is_multiple_of(tiling.block_k()) => Tile64x64x16_Simdgroups2x2,
        _ => tiling,
    }
}

/// The baseline's K splits for grids too small to stream from every one of `units` units: the most that keep at most
/// `residency` workgroups a unit, as many as fit one, and K runs of at least `COLUMNS`, or `SHORT_COLUMNS` where half the
/// splits would leave units idle. Past that, partials and their reduction cost more than the extra streaming gains, as
/// decoding runs one product after another.
fn int8_splits(
    tiles: u64,
    k: u32,
    residency: u32,
    units: u32,
) -> u32 {
    const COLUMNS: u32 = 1536;
    const SHORT_COLUMNS: u32 = 512;
    let fits = |splits: u32| {
        let workgroups = tiles * u64::from(splits);
        let columns = if workgroups / 2 <= u64::from(units) {
            SHORT_COLUMNS
        } else {
            COLUMNS
        };
        workgroups <= u64::from(residency) * u64::from(units) && k.is_multiple_of(128 * splits) && k / splits >= columns
    };
    [8, 4, 2].into_iter().find(|&splits| fits(splits)).unwrap_or(1)
}
