use thiserror::Error;

use crate::{
    backends::{
        common::{
            gpu_types::{
                gemm::{GemmBPrologueKind, GemmDTransform},
                hadamard_order::HADAMARD_TRANSFORM_BLOCK_SIZE,
            },
            kernel::{
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{ActivationFormat, MatmulShape},
            },
        },
        metal::{
            gpu_types::gemm::{GemmBLayout, GemmTiling},
            kernel::matmul::{
                gemm::{GemmEngine, GemmPlan, policy},
                weight_block_k,
            },
        },
    },
    data_type::DataType,
};

pub(super) const TRELLIS_K_STEP: u32 = 64;

#[derive(Clone, Copy)]
pub struct GemmProblem {
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    supports_mxu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum GemmPlanError {
    #[error("MXU engine is not available for this GEMM")]
    MxuUnavailable,
    #[error("GEMM input layout is unsupported: {0}")]
    UnsupportedLayout(&'static str),
}

/// Activation format, tiling and K alignment of the small-M GEMM for W4 and W8 (group 32 or 64) and W2 (group 128).
pub fn small_m_weights(shape: &MatmulShape) -> Option<(ActivationFormat, GemmTiling, u32)> {
    let bits = shape.b_bits?;
    let group_size = weight_block_k(shape.b_bits, shape.b_group_size)?;
    let packed_half = ActivationFormat::PackedHalf {
        block_k: group_size,
        bits,
    };
    match (shape.b_prologue, bits, group_size) {
        (GemmBPrologueKind::ScaleBiasDequant, 2, 128) => {
            Some((packed_half, GemmTiling::Tile8x32x128_Simdgroups1x2x4, 512))
        },
        (GemmBPrologueKind::ScaleZeroPointDequant | GemmBPrologueKind::ScaleBiasDequant, 4, 32 | 64) => {
            Some((packed_half, GemmTiling::Tile8x32x64_Simdgroups1x2x4, 512))
        },
        (GemmBPrologueKind::ScaleSymmetricDequant, 8, 32 | 64) => {
            Some((ActivationFormat::Bf16, GemmTiling::Tile8x32x128_Simdgroups1x2x4, 256))
        },
        _ => None,
    }
}

pub fn small_batch_format(
    shape: MatmulShape,
    layout: GemmBLayout,
) -> Option<(ActivationFormat, GemmTiling)> {
    let (activation, row_major_tiling, k_alignment) = small_m_weights(&shape)?;
    let tiling = match layout {
        GemmBLayout::RowMajor => row_major_tiling,
        GemmBLayout::InterleavedRows if shape.b_bits == Some(4) => GemmTiling::Tile8x32x64_Simdgroups1x1x4,
        _ => return None,
    };
    (shape.b_transpose
        && !shape.gathered
        && !shape.b_is_trellis
        && shape.b_leading_dimension.is_none()
        && shape.k.is_multiple_of(k_alignment)
        && (!shape.d_transform.contains(GemmDTransform::RHT) || shape.n.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)))
    .then_some((activation, tiling))
}

impl GemmProblem {
    pub fn new(
        shape: MatmulShape,
        weights_data_type: DataType,
        output_data_type: DataType,
        supports_mxu: bool,
    ) -> Self {
        Self {
            shape,
            weights_data_type,
            output_data_type,
            supports_mxu,
        }
    }

    pub fn select_plan(self) -> GemmPlan {
        let engine = if self.supports_mxu && mxu_is_eligible(self.shape) {
            GemmEngine::Mxu
        } else {
            GemmEngine::Simdgroup
        };
        if self.shape.b_is_trellis {
            return select_trellis_plan(self.shape.m, self.shape.n, self.shape.k, engine);
        }
        self.finish_plan(engine, select_tiling(self.shape, engine))
    }

    pub fn small_batch_plan(
        self,
        layout: GemmBLayout,
    ) -> Option<GemmPlan> {
        let (activation, tiling) = small_batch_format(self.shape, layout)?;
        let plan = self.finish_plan(GemmEngine::Simdgroup, tiling);
        let partition_alignment = if plan.split_k == 1 {
            256
        } else {
            tiling.simdgroups_k() * tiling.block_k()
        };
        (self.shape.a == activation
            && (2..=8).contains(&self.shape.m)
            && (self.shape.m >= 4 || self.shape.b_bits == Some(2))
            && self.shape.k.is_multiple_of(partition_alignment * plan.split_k))
        .then_some(plan)
    }

    #[cfg(test)]
    pub(super) fn select_plan_for_engine(
        self,
        engine: GemmEngine,
    ) -> Result<GemmPlan, GemmPlanError> {
        if self.shape.b_is_trellis {
            return Ok(select_trellis_plan(self.shape.m, self.shape.n, self.shape.k, engine));
        }
        self.validate_engine(engine)?;
        Ok(self.finish_plan(engine, select_tiling(self.shape, engine)))
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
        if !splittable
            || !self.split_k_output_supported()
            || (shape.b_bits == Some(2) && tiling == GemmTiling::Tile8x32x128_Simdgroups1x2x4)
            || tiling == GemmTiling::Tile8x32x64_Simdgroups1x1x4
        {
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
        let mut output_transform = self.shape.d_transform;
        if self.shape.is_quant()
            && output_transform.contains(GemmDTransform::RHT)
            && output_transform.contains(GemmDTransform::BIAS)
        {
            output_transform.remove(GemmDTransform::BIAS);
        }
        !output_transform.contains(GemmDTransform::BIAS)
            || (self.shape.n.is_multiple_of(4) && self.weights_data_type == self.output_data_type)
    }
}

fn select_trellis_plan(
    m: u32,
    n: u32,
    k: u32,
    engine: GemmEngine,
) -> GemmPlan {
    let (tiling, target_workgroups) = match engine {
        GemmEngine::Simdgroup if m <= 16 => {
            (GemmTiling::Tile16x32x32_Simdgroups2x1, policy::SPLIT_K_TARGET_TILES_TRELLIS_TILE16)
        },
        GemmEngine::Simdgroup => (policy::simdgroup_quant_tile(m), policy::SPLIT_K_TARGET_TILES_TRELLIS_SIMDGROUP),
        // The M16 wide finalist favored a 32x64 tile and a 256-workgroup target.
        GemmEngine::Mxu if m == 16 && n >= 4096 && (2048..=5120).contains(&k) && k.is_multiple_of(TRELLIS_K_STEP) => {
            (GemmTiling::Tile32x64x256_Simdgroups2x2, 256)
        },
        GemmEngine::Mxu => match policy::mxu_mn_tile(true, m, n) {
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
        },
    };
    let output_workgroups = (n.div_ceil(tiling.block_n()) * m.div_ceil(tiling.block_m())).max(1);
    let k_steps = (k / TRELLIS_K_STEP).max(1);
    let mut split_k = (target_workgroups / output_workgroups).clamp(1, k_steps);
    // The split-K reduce reads four partial sums at a time.
    if !n.is_multiple_of(4) {
        split_k = 1;
    }
    while !k_steps.is_multiple_of(split_k) {
        split_k -= 1;
    }
    GemmPlan {
        engine,
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
        weight_block_k(shape.b_bits, shape.b_group_size).filter(|&block_k| block_k != 0)
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
) -> GemmTiling {
    match engine {
        GemmEngine::Simdgroup if shape.is_quant() => policy::simdgroup_quant_tile(shape.m),
        GemmEngine::Simdgroup => GemmTiling::Tile64x32x32_Simdgroups2x2,
        GemmEngine::Mxu if shape.a_is_quantized() => match policy::mxu_int8_gemm_tile(shape.m, shape.n, shape.b_bits) {
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
