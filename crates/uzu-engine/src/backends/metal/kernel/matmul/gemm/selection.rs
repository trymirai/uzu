use metal::MTLGPUFamily;
use thiserror::Error;

use super::kernel::GemmKernel;
use crate::{
    backends::{
        common::{
            gpu_types::{
                gemm::{GemmBPrologueKind, GemmDTransform},
                hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            },
            kernel::{
                ActivationQuantization,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{ActivationFormat, Int8CodeLayout, MatmulShape, QuantParamsLayout},
            },
        },
        metal::{
            gpu_types::gemm::{GemmBLayout, GemmTiling},
            kernel::matmul::gemm::{GemmEngine, GemmPlan, policy},
        },
    },
    data_type::DataType,
};

pub(super) const TRELLIS_K_STEP: u32 = 64;
pub const MXU_INT8_MIN_BATCH: u32 = 5;

#[derive(Clone, Copy)]
pub struct GemmProblem {
    shape: MatmulShape,
    data_types: (DataType, DataType, DataType),
    supports_mxu: bool,
    apple_gpu_family: MTLGPUFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GemmPlanError {
    #[error("MXU engine is not available for this GEMM")]
    MxuUnavailable,
    #[error("GEMM input layout is unsupported: {0}")]
    UnsupportedLayout(&'static str),
}

pub fn supports_integer_right_operand(shape: &MatmulShape) -> bool {
    matches!((shape.b_bits, shape.signed_codes), (Some(4), _) | (Some(8), true))
}

fn small_batch_format(
    shape: MatmulShape,
    layout: GemmBLayout,
) -> Option<(ActivationFormat, GemmTiling)> {
    let (activation, tiling, k_alignment) = match (shape.b_prologue, shape.b_bits, shape.b_group_size) {
        (GemmBPrologueKind::ScaleZeroPointDequant, Some(4), Some(group_size @ (32 | 64))) => (
            ActivationFormat::PackedHalf {
                group_size,
            },
            GemmTiling::Tile8x32x64_Simdgroups1x2x4,
            512,
        ),
        (GemmBPrologueKind::ScaleSymmetricDequant, Some(8), Some(64)) => {
            (ActivationFormat::Bf16, GemmTiling::Tile8x32x128_Simdgroups1x2x4, 256)
        },
        _ => return None,
    };
    (shape.b_transpose
        && !shape.gathered
        && !shape.b_is_trellis
        && layout == GemmBLayout::RowMajor
        && shape.b_leading_dimension.is_none()
        && shape.k.is_multiple_of(k_alignment)
        && (!shape.d_transform.contains(GemmDTransform::RHT) || shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)))
    .then_some((activation, tiling))
}

pub(super) fn small_batch_plan_supported(
    shape: MatmulShape,
    gemm_b_layout: GemmBLayout,
    plan: GemmPlan,
) -> bool {
    let Some((activation, tiling)) = small_batch_format(shape, gemm_b_layout) else {
        return false;
    };
    let is_w4 = tiling.is_packed_w4();
    let partition_alignment: u32 = if is_w4 || plan.split_k == 1 {
        256
    } else {
        512
    };
    let split_alignment = partition_alignment.checked_mul(plan.split_k);
    let split_is_valid = plan.split_k > 0 && split_alignment.is_some_and(|alignment| shape.k.is_multiple_of(alignment));

    shape.a == activation
        && (4..=8).contains(&shape.m)
        && plan.engine == GemmEngine::Simdgroup
        && plan.tiling == tiling
        && split_is_valid
}

impl GemmProblem {
    pub fn new(
        shape: MatmulShape,
        data_types: (DataType, DataType, DataType),
        supports_mxu: bool,
        apple_gpu_family: MTLGPUFamily,
    ) -> Self {
        Self {
            shape,
            data_types,
            supports_mxu,
            apple_gpu_family,
        }
    }

    pub fn select_plan(self) -> GemmPlan {
        if self.shape.b_is_trellis {
            return select_trellis_plan(self.shape.m, self.shape.n, self.shape.k);
        }
        let engine = if self.supports_mxu && mxu_is_eligible(self.shape) {
            GemmEngine::Mxu
        } else {
            GemmEngine::Simdgroup
        };
        self.finish_plan(engine, select_tiling(self.shape, engine, self.apple_gpu_family))
    }

    pub fn small_batch_plan(
        self,
        layout: GemmBLayout,
    ) -> Option<GemmPlan> {
        if self.data_types != (DataType::BF16, DataType::BF16, DataType::BF16)
            || (self.supports_mxu && self.shape.m < MXU_INT8_MIN_BATCH)
        {
            return None;
        }
        let (_, tiling) = small_batch_format(self.shape, layout)?;
        let plan = self.finish_plan(GemmEngine::Simdgroup, tiling);
        small_batch_plan_supported(self.shape, layout, plan).then_some(plan)
    }

    fn int8_quantization(&self) -> Option<ActivationQuantization> {
        let shape = &self.shape;
        let weight_group_size = shape.b_group_size?;
        let emit_group_sums = GemmKernel::requires_activation_group_sums(shape);
        let code_layout = shape.b_bits.and_then(Int8CodeLayout::for_right_bits)?;
        let quantization =
            ActivationQuantization::new(ACTIVATION_SCALE_GROUP_SIZE, weight_group_size, emit_group_sums, code_layout)?;
        if !self.supports_mxu
            || self.data_types.1 != DataType::BF16
            || self.data_types.2 != DataType::BF16
            || shape.params_layout != Some(QuantParamsLayout::GroupOutput)
            || !supports_integer_right_operand(shape)
            || !shape.is_quant()
            || !shape.b_transpose
            || shape.b_leading_dimension.is_some()
            || !shape.k.is_multiple_of(ACTIVATION_SCALE_GROUP_SIZE)
            || !shape.k.is_multiple_of(weight_group_size)
            || !shape.n.is_multiple_of(32)
            || shape.b_bits.and_then(|bits| GemmKernel::mxu_tiled_words_per_thread(shape.k, bits)).is_none()
        {
            return None;
        }

        Some(quantization)
    }

    pub fn activation_preparation(
        self,
        layout: GemmBLayout,
    ) -> ActivationFormat {
        if let Some(quantization) = self.int8_quantization() {
            return ActivationFormat::Int8(quantization);
        }
        if self.shape.params_layout == Some(QuantParamsLayout::GroupOutput)
            && self.data_types == (DataType::BF16, DataType::BF16, DataType::BF16)
        {
            return small_batch_format(self.shape, layout).map_or(ActivationFormat::Bf16, |(format, _)| format);
        }
        ActivationFormat::Bf16
    }

    pub fn prepared_plan(
        self,
        layout: GemmBLayout,
    ) -> Option<GemmPlan> {
        match self.shape.a {
            ActivationFormat::PackedHalf {
                ..
            } => self.small_batch_plan(layout),
            ActivationFormat::Int8(quantization) if self.int8_quantization() == Some(quantization) => {
                Some(self.select_plan())
            },
            _ => None,
        }
    }

    #[cfg(test)]
    pub(super) fn select_plan_for_engine(
        self,
        engine: GemmEngine,
    ) -> Result<GemmPlan, GemmPlanError> {
        if self.shape.b_is_trellis {
            let mut plan = select_trellis_plan(self.shape.m, self.shape.n, self.shape.k);
            plan.engine = engine;
            return Ok(plan);
        }
        self.validate_engine(engine)?;
        Ok(self.finish_plan(engine, select_tiling(self.shape, engine, self.apple_gpu_family)))
    }

    pub(super) fn validate_engine(
        &self,
        engine: GemmEngine,
    ) -> Result<(), GemmPlanError> {
        if engine == GemmEngine::Mxu && !self.supports_mxu {
            return Err(GemmPlanError::MxuUnavailable);
        }
        if self.shape.is_quant() && (!self.shape.b_transpose || self.shape.b_leading_dimension.is_some()) {
            return Err(GemmPlanError::UnsupportedLayout("quantized weights require transposed contiguous B"));
        }
        if self.shape.b_is_trellis && !self.shape.k.is_multiple_of(TRELLIS_K_STEP) {
            return Err(GemmPlanError::UnsupportedLayout("Trellis K must be divisible by 64"));
        }
        Ok(())
    }

    fn finish_plan(
        &self,
        engine: GemmEngine,
        tiling: GemmTiling,
    ) -> GemmPlan {
        GemmPlan {
            engine,
            tiling,
            split_k: self.select_split_k(engine, tiling),
        }
    }

    fn select_split_k(
        &self,
        engine: GemmEngine,
        tiling: GemmTiling,
    ) -> u32 {
        let shape = self.shape;
        let splittable = shape.is_quant() || (shape.b_transpose && shape.b_leading_dimension.is_none());
        if !splittable || !self.split_k_output_supported() {
            return 1;
        }
        let base_tiles = shape.n.div_ceil(tiling.block_n()).saturating_mul(shape.m.div_ceil(tiling.block_m()));
        if base_tiles == 0 || !((shape.m as u64) * (shape.n as u64)).is_multiple_of(4) {
            return 1;
        }
        let Some(step) = outer_block_k(shape, engine, tiling) else {
            return 1;
        };
        let align = step.max(ACTIVATION_SCALE_GROUP_SIZE) * tiling.simdgroups_k();
        let target_tiles = policy::split_k_target_tiles(shape.a_is_quantized(), tiling, shape.b_bits);
        let mut split_k = (target_tiles / base_tiles).max(1).min((shape.k / align).max(1));
        if shape.a_is_quantized() && engine == GemmEngine::Mxu && tiling.block_k() != 0 {
            split_k = split_k.min((shape.k / tiling.block_k()).max(1));
        }
        while split_k > 1 && !shape.k.is_multiple_of(split_k * align) {
            split_k -= 1;
        }
        split_k
    }

    fn split_k_output_supported(&self) -> bool {
        use crate::backends::common::gpu_types::gemm::GemmDTransform;

        let mut output_transform = self.shape.d_transform;
        if self.shape.is_quant()
            && output_transform.contains(GemmDTransform::RHT)
            && output_transform.contains(GemmDTransform::BIAS)
        {
            output_transform.remove(GemmDTransform::BIAS);
        }
        !output_transform.contains(GemmDTransform::BIAS)
            || (self.shape.n.is_multiple_of(4) && self.data_types.0 == self.data_types.2)
    }
}

pub(super) fn select_trellis_plan(
    m: u32,
    n: u32,
    k: u32,
) -> GemmPlan {
    // The M16 wide finalist favored a 32x64 tile and a 256-workgroup target.
    let (tiling, target_workgroups) =
        if m == 16 && n >= 4096 && (2048..=5120).contains(&k) && k.is_multiple_of(TRELLIS_K_STEP) {
            (GemmTiling::Tile32x64x256_Simdgroups2x2, 256)
        } else {
            match policy::mxu_mn_tile(true, m, n) {
                GemmTiling::Tile16x32x256_Simdgroups1x1 => {
                    (GemmTiling::Tile16x32x256_Simdgroups1x1, policy::SPLIT_K_TARGET_TILES_A8_TILE16X32)
                },
                GemmTiling::Tile32x64x256_Simdgroups2x2 => {
                    (GemmTiling::Tile64x64x256_Simdgroups2x2, policy::SPLIT_K_TARGET_TILES_A8_TILE32_W4)
                },
                GemmTiling::Tile128x128x256_Simdgroups4x4 => {
                    (GemmTiling::Tile128x128x256_Simdgroups4x4, policy::SPLIT_K_TARGET_TILES_A8)
                },
                _ => (GemmTiling::Tile64x64x256_Simdgroups2x2, policy::SPLIT_K_TARGET_TILES_A8),
            }
        };
    let output_workgroups = (n.div_ceil(tiling.block_n()) * m.div_ceil(tiling.block_m())).max(1);
    let k_steps = (k / TRELLIS_K_STEP).max(1);
    let mut split_k = (target_workgroups / output_workgroups).clamp(1, k_steps);
    while !k_steps.is_multiple_of(split_k) {
        split_k -= 1;
    }
    GemmPlan {
        engine: GemmEngine::Mxu,
        tiling,
        split_k,
    }
}

pub(super) fn outer_block_k(
    shape: MatmulShape,
    engine: GemmEngine,
    tiling: GemmTiling,
) -> Option<u32> {
    if engine == GemmEngine::Mxu && shape.is_quant() {
        shape.b_group_size.filter(|&group_size| group_size != 0)
    } else {
        Some(tiling.block_k()).filter(|&block_k| block_k != 0)
    }
}

fn mxu_is_eligible(shape: MatmulShape) -> bool {
    if shape.a_is_quantized() || shape.b_prologue == GemmBPrologueKind::FullPrecision {
        return true;
    }
    shape.b_transpose
        && shape.b_leading_dimension.is_none()
        && shape.k.is_multiple_of(policy::mxu_mn_tile(shape.a_is_quantized(), shape.m, shape.n).block_k())
}

fn select_tiling(
    shape: MatmulShape,
    engine: GemmEngine,
    apple_gpu_family: MTLGPUFamily,
) -> GemmTiling {
    match engine {
        GemmEngine::Simdgroup if shape.is_quant() => policy::simdgroup_quant_tile(shape.m, shape.n, apple_gpu_family),
        GemmEngine::Simdgroup => policy::simdgroup_fp_tile(shape.m, shape.n, shape.k),
        GemmEngine::Mxu if shape.a_is_quantized() => match policy::mxu_int8_gemm_tile(shape.m, shape.n) {
            tiling
                if tiling.simdgroups_k() > 1
                    && !shape.k.is_multiple_of(tiling.simdgroups_k() * ACTIVATION_SCALE_GROUP_SIZE) =>
            {
                GemmTiling::Tile16x32x256_Simdgroups1x1
            },
            tiling => tiling,
        },
        GemmEngine::Mxu if shape.is_quant() => policy::mxu_mn_tile(false, shape.m, shape.n),
        GemmEngine::Mxu if shape.b_transpose => policy::mxu_fp_tile(shape.m, shape.n, shape.k),
        GemmEngine::Mxu => policy::mxu_mn_tile(false, shape.m, shape.n),
    }
}

#[cfg(test)]
#[path = "../../../../../../unit/backends/metal/kernel/matmul/gemm/selection_test.rs"]
mod tests;
