use itertools::iproduct;

use super::VkGemvTile;
use crate::{
    backends::{
        common::{
            allocator::MAX_ALLOCATION_SIZE,
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::{
                ActivationQuantization,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{ActivationFormat, Int8CodeLayout, MatmulShape, QuantParamsLayout},
            },
        },
        vulkan::{
            VkDeviceCapabilities as Capability, VkDeviceFacts,
            gpu_types::{
                gemm::{GemmTiling, INT8_FOLD_ENTRIES, INT8_FOLD_PADDING},
                gemm_b_layout::GemmBLayout,
            },
        },
    },
    data_type::DataType,
};

/// The most plans calibration times for one problem.
const MAX_CANDIDATES: usize = 24;
/// The cooperative core's existing tilings, then the scalar core's.
const TILINGS: [GemmTiling; 9] = [
    GemmTiling::Tile16x64x64_Simdgroups1x4x2,
    GemmTiling::Tile32x64x64_Simdgroups1x4x2,
    GemmTiling::Tile64x128x32_Simdgroups2x4x1,
    GemmTiling::Tile128x128x32_Simdgroups2x4x1,
    GemmTiling::Tile8x32x32_Simdgroups1x1,
    GemmTiling::Tile32x32x32_Simdgroups2x2,
    GemmTiling::Tile64x32x32_Simdgroups2x2,
    GemmTiling::Tile64x64x16_Simdgroups2x2,
    GemmTiling::Tile64x64x32_Simdgroups2x2,
];
/// The INT8 GEMM cores by the device matrices each runs on: GemmInt8, GemmInt8K16 and GemmInt8Nv.
const INT8_CORES: [Capability; 3] = [
    Capability::INT8_TENSOR_CORES_K32,
    Capability::INT8_TENSOR_CORES_K16,
    Capability::INT8_TENSOR_CORES_K32.union(Capability::INT8_ELEMENT_FOLD),
];
/// The fold metadata arrays every Int8Gemm core declares in shared memory.
const INT8_FOLD_BYTES: u32 = 4 * (2 * INT8_FOLD_ENTRIES + INT8_FOLD_PADDING);
/// One dispatch family of the existing matmul kernels with its specialization parameters: the committed baseline or an
/// explicit calibration candidate, which the one encoder runs alike. A plan's cost is all of its dispatches: split K's
/// reduction phase and the output transform the encoder appends. The loaded weight layout and the activation format
/// are the problem's, never the plan's. Legality and the baseline read [`VkDeviceFacts`], never the device's identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VkMatmulPlan {
    /// Gemv on full-precision A, with its tile.
    Gemv(VkGemvTile),
    /// GemvInt8 on A8: `8 * rows` outputs by `input_rows` inputs a workgroup of 256 invocations. One input row or 8 or
    /// more rows give each output 32 / rows invocations, which split its runs for every input; fewer rows by more inputs
    /// give each 32-lane simdgroup `rows` outputs.
    GemvInt8 {
        rows: u32,
        input_rows: u32,
    },
    /// The INT8 GEMM core of `matrices`, one of INT8_CORES: `row_block` rows by 64 columns a workgroup of
    /// `simdgroups`, staging `stage_depth` K values a step, with K in `splits` partial phases then a reduction phase
    /// where it splits.
    GemmInt8 {
        matrices: Capability,
        row_block: u32,
        simdgroups: u32,
        stage_depth: u32,
        splits: u32,
    },
    /// GemmTensor's cooperative core on BF16 or FP16 A and B.
    GemmTensor(GemmTiling),
    /// Gemm's scalar core.
    Gemm(GemmTiling),
}

impl VkMatmulPlan {
    /// The default, without allocating: a generic rule of the shape, the loaded weight layout and raw device
    /// capabilities, and the context's separate eligibility for floating-point cooperative products.
    pub fn baseline(
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: VkDeviceFacts<'_>,
        float_tensor_cores_eligible: bool,
    ) -> Self {
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let [weights, input, _] = data_types;
        let (capabilities, units) = (device.capabilities, device.compute_units);
        let max_shared = device.limits.max_compute_shared_memory_size;
        // Where only unsigned dot products are accelerated, up to 8 rows of full-precision A quantize themselves.
        if shape.a == ActivationFormat::Bf16
            && m <= 8
            && capabilities.contains(Capability::UNSIGNED_DOT_PRODUCT)
            && Self::a8_quantization(shape, data_types, capabilities).is_some()
        {
            return Self::gemv_int8(m, shape.gathered, device.subgroups.size);
        }
        if shape.a == ActivationFormat::Bf16 {
            return if shape.gathered || m <= 8 || !Self::gemm_rows(shape) {
                Self::Gemv(match (shape.is_quant(), Self::gemm_rows(shape)) {
                    (false, true) => VkGemvTile::dense(n, k),
                    (false, false) => VkGemvTile::reader(),
                    (true, _) => {
                        quantized_gemv_tiles(shape, b_layout, data_types).next().expect("the reader ends every list")
                    },
                })
            } else if float_tensor_cores_eligible && capabilities.float_tensor_cores(input) && input == weights {
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
            return Self::gemv_int8(m, shape.gathered, device.subgroups.size);
        }
        // The element fold, else 16 x 16 x 32 matrices before 16 x 16 x 16 ones where the device reports both.
        let element_fold = capabilities.contains(Capability::INT8_ELEMENT_FOLD);
        let matrices = match (element_fold, capabilities.contains(Capability::INT8_TENSOR_CORES_K32)) {
            (true, _) => INT8_CORES[2],
            (false, true) => INT8_CORES[0],
            (false, false) => INT8_CORES[1],
        };
        let core = |row_block: u32, stage_depth: u32, splits: u32| Self::GemmInt8 {
            matrices,
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
        // The committed heuristic, not an occupancy model: GemmInt8Nv's 16-row tile stages 128 values deep for grids of
        // no more tiles than units, which split K anyway; it counts three such workgroups a unit against five 64 deep.
        // Other depths and splits stay legal candidates, also without a unit count.
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
    /// recorded. Family fields are validated before any arithmetic uses them; then the problem's format against the
    /// family (the A8 contract included), the shaders' layout, tile, split and capability contracts, the workgroup
    /// against the device's dimensions, invocations and required subgroups, shared memory of static plus specialized
    /// staging, split K's scratch against the pool and the largest buffer, and every dispatch's workgroup count. Buffer
    /// alignment, scales and metadata planes remain the encoder's checks.
    pub fn check(
        &self,
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: VkDeviceFacts<'_>,
    ) -> Result<(), &'static str> {
        let [weights, input, _] = data_types;
        let (m, n, k) = (shape.m, shape.n, shape.k);
        let (capabilities, limits, subgroups) = (device.capabilities, device.limits, device.subgroups);
        let group = shape.b_group_size.unwrap_or(0);
        let max_shared = limits.max_compute_shared_memory_size;
        let require = |met: bool, reason: &'static str| met.then_some(()).ok_or(reason);
        require(m > 0 && n > 0 && k > 0, "matmul dimensions must be positive")?;
        require(!shape.is_quant() || group > 0, "quantized B needs a positive group size")?;
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
        match *self {
            Self::Gemv(tile) => {
                tile.contract(shape.is_quant(), shape.b_bits.unwrap_or(0), group, shape.gathered)?;
                let k_block = if shape.is_quant() {
                    group
                } else {
                    128
                };
                require(
                    !tile.metal_layout || shape.is_quant() || Self::gemm_rows(shape),
                    "the lane layout reads contiguous full-precision B rows",
                )?;
                require(
                    !tile.metal_layout || shape.b_bits != Some(4) || k % 2 == 0,
                    "the lane layout reads whole U4 bytes",
                )?;
                require(
                    !tile.full_tile || (n % tile.output_row_tile() == 0 && m % tile.input_row_tile == 0),
                    "a full tile needs whole output and input tiles",
                )?;
                require(
                    !tile.input_aligned || (k_block > 0 && k % k_block == 0),
                    "aligned input needs whole K blocks",
                )?;
            },
            Self::GemvInt8 {
                rows,
                input_rows,
            } => {
                require(
                    [1, 2, 4, 8, 16].contains(&rows) && (1..=8).contains(&input_rows),
                    "GemvInt8 takes 1, 2, 4, 8 or 16 rows by 1 to 8 inputs",
                )?;
                require(!shape.gathered || input_rows == 1, "gathered GemvInt8 takes one input row a workgroup")?;
            },
            Self::GemmInt8 {
                matrices,
                row_block,
                simdgroups,
                stage_depth,
                splits,
            } => {
                let element_fold = matrices.contains(Capability::INT8_ELEMENT_FOLD);
                require(!shape.gathered, "INT8 GEMM stages contiguous B rows, without gather")?;
                require(
                    INT8_CORES.contains(&matrices)
                        && [16, 32, 64].contains(&row_block)
                        && [1, 2, 4, 8].contains(&splits)
                        && (simdgroups
                            == (if row_block == 64 {
                                8
                            } else {
                                4
                            })
                            || (element_fold && row_block == 32 && simdgroups == 8))
                        && (stage_depth == 64
                            || (element_fold && stage_depth == 128 && row_block == 16 && group.is_multiple_of(64))),
                    "INT8 cores take row blocks of 16, 32 or 64, their simdgroups, 64-deep staging or the element \
                     fold's 128 on 16 rows of 64-value groups, and 1, 2, 4 or 8 splits",
                )?;
                require(
                    capabilities.contains(matrices | Capability::SUBGROUP_SIZE_32),
                    "the device reports none of this core's INT8 matrices on requirable 32-lane subgroups",
                )?;
                require(k % (128 * splits) == 0, "split K takes whole 128-value blocks a split")?;
                require(self.int8_shared_bytes() <= max_shared, "the staging exceeds the device's shared memory")?;
                // The encoder counts the 32-bit partial sums in u32 and takes them from the scratch pool, whose page of
                // the next power of two bytes one buffer and one allocation must hold.
                let partials = splits.checked_mul(m).and_then(|count| count.checked_mul(n));
                let bytes = partials.map(|count| 4 * u64::from(count));
                require(
                    splits == 1
                        || bytes.is_some_and(|bytes| {
                            bytes <= MAX_ALLOCATION_SIZE as u64
                                && bytes.next_power_of_two() <= device.max_buffer_bytes()
                        }),
                    "split K's partial sums exceed 32-bit indexing, the scratch pool or the device's largest buffer",
                )?;
            },
            Self::GemmTensor(tiling) => {
                require(
                    tiling.is_tensor() && input == weights && capabilities.float_tensor_cores(input),
                    "the cooperative core takes a tensor tiling of equal BF16 or FP16 A and B on matrices the device \
                     reports",
                )?;
                require(16 * tiling.tensor_staging() <= max_shared, "the staging exceeds the device's shared memory")?;
            },
            Self::Gemm(tiling) => {
                let bytes = |left: bool, size: u32| tiling.scalar_staging(left, size) * size;
                require(!tiling.is_tensor(), "the scalar core takes a scalar tiling")?;
                require(
                    !shape.is_quant() || group.is_multiple_of(tiling.block_k()),
                    "groups hold whole K blocks of the tiling",
                )?;
                require(
                    bytes(true, input.size_in_bytes() as u32) + bytes(false, weights.size_in_bytes() as u32)
                        <= max_shared,
                    "the staging exceeds the device's shared memory",
                )?;
            },
        }
        // The family fields are in range now, so the workgroup's dimensions are small.
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
    /// K's partial phase spans the splits, its reduction phase one. Called on validated fields only.
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

    /// The baseline, then the other legal plans of the activation format's families, taken from each family in turn up
    /// to MAX_CANDIDATES; every tuple left out comes back with its reason: illegal, a bounded-search omission of row
    /// tiles past the problem's rows, or past the cap. A8 searches GemvInt8's tiles and every INT8 GEMM core the
    /// device's matrices allow, whatever the committed routing prefers; full precision searches Gemv's existing tiles,
    /// Gemm's tilings and GemvInt8 on the A it quantizes. Only this explicit search allocates.
    pub fn candidates(
        baseline: Self,
        shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        device: VkDeviceFacts<'_>,
    ) -> (Vec<Self>, Vec<(Self, &'static str)>) {
        let (m, n, k) = (shape.m, shape.n, shape.k);
        // The bounded search omits tiles of more rows than the next power of two of the problem's, not measured slower.
        let omitted = |rows: u32, least: u32| rows > m.checked_next_power_of_two().unwrap_or(u32::MAX).max(least);
        let grid = iproduct!([1, 2, 4, 8, 16], [1, 2, 4, 8]);
        let gemv_int8 = grid
            .map(|(rows, input_rows)| {
                let plan = Self::GemvInt8 {
                    rows,
                    input_rows,
                };
                (plan, omitted(input_rows, 1))
            })
            .collect::<Vec<_>>();
        let families: Vec<Vec<(Self, bool)>> = if shape.a == ActivationFormat::Bf16 {
            let gemv: Vec<VkGemvTile> = match (shape.is_quant(), Self::gemm_rows(shape)) {
                (false, true) => vec![VkGemvTile::dense(n, k), VkGemvTile::reader()],
                (false, false) => vec![VkGemvTile::reader()],
                (true, _) => quantized_gemv_tiles(shape, b_layout, data_types).collect(),
            };
            let (tensor, scalar) = TILINGS.split_at(4);
            let families = vec![
                gemv.into_iter().map(|tile| (Self::Gemv(tile), false)).collect(),
                tensor.iter().map(|&tiling| (Self::GemmTensor(tiling), false)).collect(),
                scalar.iter().map(|&tiling| (Self::Gemm(tiling), false)).collect(),
            ];
            families.into_iter().chain([gemv_int8]).collect()
        } else {
            let gemm = INT8_CORES.map(|matrices| {
                let grid = iproduct!([16, 32, 64], [4, 8], [64, 128], [1, 2, 4, 8]);
                grid.map(|(row_block, simdgroups, stage_depth, splits)| {
                    let plan = Self::GemmInt8 {
                        matrices,
                        row_block,
                        simdgroups,
                        stage_depth,
                        splits,
                    };
                    (plan, omitted(row_block, 16))
                })
                .collect::<Vec<_>>()
            });
            [gemv_int8].into_iter().chain(gemm).collect()
        };
        let mut excluded = Vec::new();
        let mut candidates = Vec::new();
        match baseline.check(shape, b_layout, data_types, device) {
            Ok(()) => candidates.push(baseline),
            Err(reason) => excluded.push((baseline, reason)),
        }
        let mut legal = Vec::new();
        for family in families {
            let mut plans = Vec::new();
            for (plan, omitted) in family.into_iter().filter(|&(plan, _)| plan != baseline) {
                match plan.check(shape, b_layout, data_types, device) {
                    Err(reason) => excluded.push((plan, reason)),
                    Ok(()) if omitted => excluded.push((plan, "bounded search: a row tile past the problem's rows")),
                    Ok(()) => plans.push(plan),
                }
            }
            legal.push(plans.into_iter());
        }
        // A plan from each family in turn, so the cap leaves every legal family represented.
        loop {
            let round = legal.iter_mut().filter_map(Iterator::next).collect::<Vec<_>>();
            if round.is_empty() {
                return (candidates, excluded);
            }
            for plan in round {
                match candidates.len() < MAX_CANDIDATES {
                    true => candidates.push(plan),
                    false => excluded.push((plan, "past the candidate cap")),
                }
            }
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
                matrices,
                row_block,
                stage_depth,
                ..
            } => match matrices.contains(Capability::INT8_ELEMENT_FOLD) {
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

    /// The activation quantization the A8 kernels take for `shape`'s B, whatever its A: BF16 or FP16 activations and
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
        (matches!(input, DataType::BF16 | DataType::F16)
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

    /// The format callers give `bf16_shape`'s A from weights in `b_layout`: A8 within its contract from one row on
    /// unsigned-dot devices, otherwise from 2 to 8 rows and, where the device prefers INT8 GEMM, beyond; tiled weights
    /// take A8 from 2 rows.
    pub fn activation_format(
        bf16_shape: &MatmulShape,
        b_layout: GemmBLayout,
        data_types: [DataType; 3],
        capabilities: Capability,
    ) -> ActivationFormat {
        let m = bf16_shape.m;
        let rows = m >= 1
            && (capabilities.contains(Capability::UNSIGNED_DOT_PRODUCT)
                || (m >= 2 && (m <= 8 || capabilities.prefers_int8_gemm())));
        let int8 = match b_layout {
            GemmBLayout::Tiled => m >= 2,
            GemmBLayout::RowMajor => rows,
        };
        match Self::a8_quantization(bf16_shape, data_types, capabilities).filter(|_| int8) {
            Some(quantization) => ActivationFormat::Int8(quantization),
            None => ActivationFormat::Bf16,
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

/// Gemv's tiles of a quantized shape from weights in `b_layout`, the default first, for BF16 or FP16 throughout or other
/// data types.
fn quantized_gemv_tiles(
    shape: &MatmulShape,
    b_layout: GemmBLayout,
    [_, input, output]: [DataType; 3],
) -> impl Iterator<Item = VkGemvTile> {
    let bf16_io = matches!(input, DataType::BF16 | DataType::F16) && output == input;
    let (bits, group) = (shape.b_bits.unwrap_or(0), shape.b_group_size.unwrap_or(0));
    VkGemvTile::quantized_tiles(bits, group, shape.m, shape.n, shape.k, bf16_io, shape.gathered, b_layout)
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
    let tiling = match group {
        None if 2_u32.saturating_mul(m.max(n)) > k => Tile64x64x16_Simdgroups2x2,
        None => Tile64x32x32_Simdgroups2x2,
        Some(group) if group < 32 => Tile64x64x16_Simdgroups2x2,
        Some(_) if m < 32 => {
            if m <= 8 || m.is_multiple_of(8) {
                Tile8x32x32_Simdgroups1x1
            } else {
                Tile32x32x32_Simdgroups2x2
            }
        },
        Some(_) if m >= 64 && n >= 6144 && n.is_multiple_of(64) => Tile64x64x32_Simdgroups2x2,
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

#[cfg(test)]
#[path = "../../../../../unit/backends/vulkan/kernel/matmul/vk_matmul_plan_test.rs"]
mod tests;
