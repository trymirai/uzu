use thiserror::Error;

use super::{GemmEngine, GemmPlan};
use crate::{
    backends::common::{
        gpu_types::gemm::{GemmBPrologueKind, GemmTiling},
        kernel::{activation_transform::ACTIVATION_SCALE_GROUP_SIZE, matmul::MatmulShape},
    },
    data_type::DataType,
};

#[derive(Clone, Copy)]
pub struct GemmProblem {
    shape: MatmulShape,
    weights_data_type: DataType,
    input_data_type: DataType,
    output_data_type: DataType,
    supports_mxu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub(super) enum GemmPlanError {
    #[error("MXU engine is not available for this GEMM")]
    MxuUnavailable,
    #[error("tiling {tiling} does not match engine {engine:?}")]
    EngineTilingMismatch {
        engine: GemmEngine,
        tiling: GemmTiling,
    },
    #[error("quantized GEMM requires transposed contiguous B")]
    UnsupportedQuantLayout,
    #[error("tiling {tiling} is incompatible with quantization group size {group_size}")]
    QuantGroupTooSmall {
        tiling: GemmTiling,
        group_size: u32,
    },
    #[error("split_k={split_k} is not legal for this GEMM problem")]
    InvalidSplitK {
        split_k: u32,
    },
}

impl GemmProblem {
    pub fn new(
        shape: MatmulShape,
        weights_data_type: DataType,
        input_data_type: DataType,
        output_data_type: DataType,
        supports_mxu: bool,
    ) -> Self {
        Self {
            shape,
            weights_data_type,
            input_data_type,
            output_data_type,
            supports_mxu,
        }
    }

    fn select_plan(self) -> GemmPlan {
        select_plan(self.shape, self.weights_data_type, self.output_data_type, self.supports_mxu)
    }

    pub fn plan_for_dispatch(
        self,
        gemv_available: bool,
    ) -> Option<GemmPlan> {
        let plan = self.select_plan();
        (!gemv_available || prefer_gemm_over_gemv(self, plan)).then_some(plan)
    }

    #[cfg(test)]
    pub(super) fn select_plan_for_engine(
        self,
        engine: GemmEngine,
    ) -> Result<GemmPlan, GemmPlanError> {
        select_plan_for_engine(self.shape, self.weights_data_type, self.output_data_type, self.supports_mxu, engine)
    }

    pub(super) fn validate(
        self,
        plan: GemmPlan,
    ) -> Result<(), GemmPlanError> {
        validate_plan(self.shape, self.weights_data_type, self.output_data_type, self.supports_mxu, plan)
    }
}

impl GemmPlan {
    pub(super) fn should_stage_weight_scales(
        self,
        shape: MatmulShape,
    ) -> bool {
        should_stage_weight_scales(shape, self)
    }

    pub(super) fn should_hoist_operand_addressing(
        self,
        shape: MatmulShape,
    ) -> bool {
        should_hoist_operand_addressing(shape, self)
    }

    pub(super) fn split_k_step(
        self,
        shape: MatmulShape,
    ) -> Option<u32> {
        split_k_step(shape, self)
    }
}

fn select_plan(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    supports_mxu: bool,
) -> GemmPlan {
    if supports_mxu {
        if !shape.a_full_precision {
            return finish_plan(
                shape,
                weights_data_type,
                output_data_type,
                GemmEngine::Mxu,
                select_mxu_quant_tiling(shape),
            );
        }
        if let Some(tiling) = select_mxu_tiling(shape) {
            return finish_plan(shape, weights_data_type, output_data_type, GemmEngine::Mxu, tiling);
        }
    }

    let tiling = select_simdgroup_tiling(shape);
    finish_plan(shape, weights_data_type, output_data_type, GemmEngine::Simdgroup, tiling)
}

#[cfg(test)]
fn select_plan_for_engine(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    supports_mxu: bool,
    engine: GemmEngine,
) -> Result<GemmPlan, GemmPlanError> {
    let tiling = match engine {
        GemmEngine::Mxu => {
            if !supports_mxu {
                return Err(GemmPlanError::MxuUnavailable);
            }
            if shape.is_quant() {
                if !shape.b_transpose || shape.b_leading_dimension.is_some() {
                    return Err(GemmPlanError::UnsupportedQuantLayout);
                }
                select_mxu_quant_tiling(shape)
            } else if shape.b_transpose {
                dense_mxu_tiling(shape.m, shape.n, shape.k)
            } else {
                mxu_tiling_by_mn(shape.m, shape.n)
            }
        },
        GemmEngine::Simdgroup => select_simdgroup_tiling(shape),
    };
    let plan = finish_plan(shape, weights_data_type, output_data_type, engine, tiling);
    validate_plan(shape, weights_data_type, output_data_type, supports_mxu, plan)?;
    Ok(plan)
}

fn validate_plan(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    supports_mxu: bool,
    plan: GemmPlan,
) -> Result<(), GemmPlanError> {
    if matches!(plan.engine, GemmEngine::Mxu) != plan.tiling.is_mxu_variant() {
        return Err(GemmPlanError::EngineTilingMismatch {
            engine: plan.engine,
            tiling: plan.tiling,
        });
    }
    if matches!(plan.engine, GemmEngine::Mxu) && !supports_mxu {
        return Err(GemmPlanError::MxuUnavailable);
    }
    if shape.is_quant() {
        if !shape.b_transpose || shape.b_leading_dimension.is_some() {
            return Err(GemmPlanError::UnsupportedQuantLayout);
        }
        let group_size = shape.b_group_size.unwrap_or(0);
        let fits = if matches!(plan.engine, GemmEngine::Mxu) {
            plan.tiling.fits_quant_group_size(group_size)
        } else {
            plan.tiling.simdgroup_block_k() <= group_size
        };
        if !fits {
            return Err(GemmPlanError::QuantGroupTooSmall {
                tiling: plan.tiling,
                group_size,
            });
        }
    }
    if !split_k_is_legal(shape, weights_data_type, output_data_type, plan) {
        return Err(GemmPlanError::InvalidSplitK {
            split_k: plan.split_k,
        });
    }
    Ok(())
}

fn prefer_gemm_over_gemv(
    problem: GemmProblem,
    plan: GemmPlan,
) -> bool {
    let shape = problem.shape;
    if shape.gathered || plan.engine != GemmEngine::Mxu {
        return false;
    }
    match (shape.m, shape.n == shape.k, (problem.weights_data_type, problem.input_data_type, problem.output_data_type))
    {
        (4, true, (DataType::F32, DataType::F32, DataType::F32))
        | (5, _, (DataType::BF16, DataType::BF16, DataType::BF16)) => return false,
        _ => {},
    }
    match shape.m {
        0..=3 => return false,
        4 => {
            let small_enough_for_mxu = shape.n <= 6144 && shape.k <= 9728;
            let k_dominates = shape.k > 3_u32.saturating_mul(shape.n);
            if !(small_enough_for_mxu || k_dominates) {
                return false;
            }
        },
        _ => {},
    }
    matches!(plan.tiling, GemmTiling::Tile16x32x256_Simdgroups1x1 | GemmTiling::Tile16x128x256_Simdgroups1x4)
}

fn should_stage_weight_scales(
    shape: MatmulShape,
    plan: GemmPlan,
) -> bool {
    const MIN_GROUPS: u32 = 6;
    let dispatch_k = shape.k / plan.split_k;
    if shape.b_bits == Some(4) && shape.b_group_size == Some(32) && plan.tiling.block_m() <= 32 {
        return false;
    }
    plan.tiling != GemmTiling::Tile32x64x256_Simdgroups2x2
        || shape.b_group_size.is_none_or(|group_size| dispatch_k / group_size >= MIN_GROUPS)
}

fn should_hoist_operand_addressing(
    shape: MatmulShape,
    plan: GemmPlan,
) -> bool {
    let needs_correction =
        matches!(shape.b_prologue, GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant);
    needs_correction || plan.tiling != GemmTiling::Tile128x128x256_Simdgroups4x4
}

fn split_k_step(
    shape: MatmulShape,
    plan: GemmPlan,
) -> Option<u32> {
    let step = if plan.engine == GemmEngine::Mxu && shape.is_quant() {
        shape.b_group_size.unwrap_or(0)
    } else {
        plan.tiling.block_k()
    };
    (step != 0).then_some(step)
}

fn finish_plan(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    engine: GemmEngine,
    tiling: GemmTiling,
) -> GemmPlan {
    let split_k = select_split_k(shape, weights_data_type, output_data_type, engine, tiling);
    GemmPlan {
        engine,
        tiling,
        split_k,
    }
}

fn select_mxu_tiling(shape: MatmulShape) -> Option<GemmTiling> {
    if !shape.a_full_precision {
        return None;
    }
    match shape.b_prologue {
        GemmBPrologueKind::FullPrecision => Some(if shape.b_transpose {
            dense_mxu_tiling(shape.m, shape.n, shape.k)
        } else {
            mxu_tiling_by_mn(shape.m, shape.n)
        }),
        _ => {
            if !shape.b_transpose || shape.b_leading_dimension.is_some() {
                return None;
            }
            let tiling = select_mxu_quant_tiling(shape);
            shape.k.is_multiple_of(tiling.block_k()).then_some(tiling)
        },
    }
}

fn select_mxu_quant_tiling(shape: MatmulShape) -> GemmTiling {
    let tiling = if shape.a_full_precision {
        mxu_tiling_by_mn(shape.m, shape.n)
    } else {
        a8_mxu_tiling(shape.m, shape.n)
    };
    if tiling.fits_quant_group_size(shape.b_group_size.unwrap_or(0)) {
        tiling
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

fn select_simdgroup_tiling(shape: MatmulShape) -> GemmTiling {
    if !shape.is_quant() {
        if 2 * shape.m.max(shape.n) > shape.k {
            GemmTiling::Tile64x64x16_Simdgroups2x2
        } else {
            GemmTiling::Tile64x32x32_Simdgroups2x2
        }
    } else {
        let group_size = shape.b_group_size.unwrap_or(0);
        if group_size < 32 {
            GemmTiling::Tile64x64x16_Simdgroups2x2
        } else if shape.m < 32 {
            GemmTiling::Tile8x32x32_Simdgroups1x1
        } else if shape.m >= 64 && shape.n <= 2048 {
            GemmTiling::Tile32x32x32_Simdgroups2x2
        } else if shape.m >= 64 && shape.n >= 6144 && shape.n.is_multiple_of(64) {
            GemmTiling::Tile64x64x32_Simdgroups2x2
        } else {
            GemmTiling::Tile32x32x32_Simdgroups2x2
        }
    }
}

fn dense_mxu_tiling(
    m: u32,
    n: u32,
    k: u32,
) -> GemmTiling {
    if m < 64 && n >= 64 {
        if n == k {
            return if m < 16 && k <= 2560 {
                GemmTiling::Tile16x32x256_Simdgroups1x1
            } else {
                GemmTiling::Tile32x64x256_Simdgroups2x2
            };
        }
        return if m < 16 {
            small_m_mxu_tiling(n, k)
        } else {
            mxu_tiling_by_mn(m, n)
        };
    }
    mxu_tiling_by_mn(m, n)
}

fn mxu_tiling_by_mn(
    m: u32,
    n: u32,
) -> GemmTiling {
    if m >= 256 && n >= 128 {
        GemmTiling::Tile128x128x256_Simdgroups4x4
    } else if n < 64 {
        GemmTiling::Tile64x32x256_Simdgroups4x1
    } else if m < 64 {
        GemmTiling::Tile32x64x256_Simdgroups2x2
    } else {
        GemmTiling::Tile64x64x256_Simdgroups2x2
    }
}

fn small_m_mxu_tiling(
    n: u32,
    k: u32,
) -> GemmTiling {
    if k > n {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    if n > 32_u32.saturating_mul(k) {
        return GemmTiling::Tile16x32x256_Simdgroups1x1;
    }
    if (k >= 4096 && n >= 4_u32.saturating_mul(k)) || (k == 2560 && n >= 6_u32.saturating_mul(k)) {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    GemmTiling::Tile32x64x256_Simdgroups2x2
}

fn a8_mxu_tiling(
    m: u32,
    n: u32,
) -> GemmTiling {
    if n < 64 {
        return GemmTiling::Tile64x32x256_Simdgroups4x1;
    }
    if m <= 16 {
        return GemmTiling::Tile16x128x256_Simdgroups1x4;
    }
    if m < 64 {
        return GemmTiling::Tile32x64x256_Simdgroups2x2;
    }
    if m >= 512 {
        return GemmTiling::Tile128x128x256_Simdgroups4x4;
    }
    GemmTiling::Tile64x64x256_Simdgroups2x2
}

fn select_split_k(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    engine: GemmEngine,
    tiling: GemmTiling,
) -> u32 {
    if (!shape.is_quant() && (!shape.b_transpose || shape.b_leading_dimension.is_some()))
        || !split_k_output_supported(shape, weights_data_type, output_data_type)
    {
        return 1;
    }
    let base_tiles = shape.n.div_ceil(tiling.block_n()) * shape.m.div_ceil(tiling.block_m());
    if base_tiles == 0 || !((shape.m as u64) * (shape.n as u64)).is_multiple_of(4) {
        return 1;
    }
    let a_is_int8 = !shape.a_full_precision;
    let target_tiles = match (a_is_int8, tiling, shape.b_bits) {
        (true, GemmTiling::Tile32x64x256_Simdgroups2x2, Some(4)) => 512,
        (true, GemmTiling::Tile32x64x256_Simdgroups2x2, _) => 1024,
        (true, _, _) => 256,
        (false, _, _) => 512,
    };
    let group_size = shape.b_group_size.unwrap_or(0);
    let full_precision = !shape.is_quant();
    let step = if engine == GemmEngine::Mxu && !full_precision {
        group_size
    } else {
        tiling.block_k()
    };
    if step == 0 {
        return 1;
    }
    let mut split_k = (target_tiles / base_tiles).max(1);
    let mut align = if engine == GemmEngine::Mxu || full_precision {
        step
    } else {
        step.max(group_size)
    };
    if shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant && shape.b_bits == Some(4) {
        align = align.max(2 * group_size);
    }
    align = align.max(ACTIVATION_SCALE_GROUP_SIZE).max(group_size);
    split_k = split_k.min((shape.k / align).max(1));
    if a_is_int8 && engine == GemmEngine::Mxu && tiling.block_k() != 0 {
        split_k = split_k.min((shape.k / tiling.block_k()).max(1));
    }
    while split_k > 1 && !shape.k.is_multiple_of(split_k * align) {
        split_k -= 1;
    }
    split_k
}

fn split_k_output_supported(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
) -> bool {
    use crate::backends::common::gpu_types::gemm::GemmDTransform;

    let mut output_transform = shape.d_transform;
    if shape.is_quant()
        && output_transform.contains(GemmDTransform::RHT)
        && output_transform.contains(GemmDTransform::BIAS)
    {
        output_transform.remove(GemmDTransform::BIAS);
    }
    !output_transform.contains(GemmDTransform::BIAS)
        || (shape.n.is_multiple_of(4) && weights_data_type == output_data_type)
}

fn split_k_is_legal(
    shape: MatmulShape,
    weights_data_type: DataType,
    output_data_type: DataType,
    plan: GemmPlan,
) -> bool {
    if plan.split_k == 0 {
        return false;
    }
    if plan.split_k == 1 {
        return true;
    }
    if (!shape.is_quant() && (!shape.b_transpose || shape.b_leading_dimension.is_some()))
        || !split_k_output_supported(shape, weights_data_type, output_data_type)
        || !((shape.m as u64) * (shape.n as u64)).is_multiple_of(4)
    {
        return false;
    }
    let group_size = shape.b_group_size.unwrap_or(0);
    let full_precision = !shape.is_quant();
    let step = if plan.engine == GemmEngine::Mxu && !full_precision {
        group_size
    } else {
        plan.tiling.block_k()
    };
    if step == 0 {
        return false;
    }
    if !shape.a_full_precision
        && plan.engine == GemmEngine::Mxu
        && plan.split_k > (shape.k / plan.tiling.block_k()).max(1)
    {
        return false;
    }
    let mut align = if plan.engine == GemmEngine::Mxu || full_precision {
        step
    } else {
        step.max(group_size)
    };
    if shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant && shape.b_bits == Some(4) {
        align = align.max(2 * group_size);
    }
    align = align.max(ACTIVATION_SCALE_GROUP_SIZE).max(group_size);
    plan.split_k.checked_mul(align).is_some_and(|split_alignment| shape.k.is_multiple_of(split_alignment))
}

#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/kernel/matmul/gemm/selection_test.rs"]
mod tests;
