use thiserror::Error;

use super::{GemmEngine, GemmPlan, policy};
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
    output_data_type: DataType,
    supports_mxu: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub(super) enum GemmPlanError {
    #[error("MXU engine is not available for this GEMM")]
    MxuUnavailable,
    #[error("quantized GEMM requires transposed contiguous B")]
    UnsupportedQuantLayout,
    #[error("split_k={split_k} is not legal for this GEMM problem")]
    InvalidSplitK {
        split_k: u32,
    },
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
        if self.supports_mxu {
            if !self.shape.a_full_precision {
                return self.finish_plan(GemmEngine::Mxu, select_mxu_quant_tiling(self.shape));
            }
            if let Some(tiling) = select_mxu_tiling(self.shape) {
                return self.finish_plan(GemmEngine::Mxu, tiling);
            }
        }

        self.finish_plan(GemmEngine::Simdgroup, select_simdgroup_tiling(self.shape))
    }

    #[cfg(test)]
    pub(super) fn select_plan_for_engine(
        self,
        engine: GemmEngine,
    ) -> Result<GemmPlan, GemmPlanError> {
        self.validate_engine(engine)?;
        let shape = self.shape;
        let tiling = match engine {
            GemmEngine::Mxu => {
                if shape.is_quant() {
                    select_mxu_quant_tiling(shape)
                } else if shape.b_transpose {
                    policy::mxu_fp_tile(shape.m, shape.n, shape.k)
                } else {
                    policy::mxu_mn_tile(false, shape.m, shape.n)
                }
            },
            GemmEngine::Simdgroup => select_simdgroup_tiling(shape),
        };
        let plan = self.finish_plan(engine, tiling);
        self.validate_split_k(plan)?;
        Ok(plan)
    }

    pub(super) fn validate(
        &self,
        plan: GemmPlan,
    ) -> Result<(), GemmPlanError> {
        self.validate_engine(plan.engine)?;
        self.validate_split_k(plan)
    }

    fn validate_split_k(
        &self,
        plan: GemmPlan,
    ) -> Result<(), GemmPlanError> {
        if !self.split_k_is_legal(plan) {
            return Err(GemmPlanError::InvalidSplitK {
                split_k: plan.split_k,
            });
        }
        Ok(())
    }

    fn validate_engine(
        &self,
        engine: GemmEngine,
    ) -> Result<(), GemmPlanError> {
        if engine == GemmEngine::Mxu && !self.supports_mxu {
            return Err(GemmPlanError::MxuUnavailable);
        }
        if self.shape.is_quant() && (!self.shape.b_transpose || self.shape.b_leading_dimension.is_some()) {
            return Err(GemmPlanError::UnsupportedQuantLayout);
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
        let base_tiles = shape.n.div_ceil(tiling.block_n()) * shape.m.div_ceil(tiling.block_m());
        if base_tiles == 0 || !((shape.m as u64) * (shape.n as u64)).is_multiple_of(4) {
            return 1;
        }
        let Some(step) = outer_block_k(shape, engine, tiling) else {
            return 1;
        };
        let group_size = shape.b_group_size.unwrap_or(0);
        let mut align = if engine == GemmEngine::Mxu || !shape.is_quant() {
            step
        } else {
            step.max(group_size)
        };
        if shape.b_prologue == GemmBPrologueKind::ScaleZeroPointDequant && shape.b_bits == Some(4) {
            align = align.max(2 * group_size);
        }
        let align = align.max(ACTIVATION_SCALE_GROUP_SIZE).max(group_size);
        let target_tiles = policy::split_k_target_tiles(!shape.a_full_precision, tiling, shape.b_bits);
        let mut split_k = (target_tiles / base_tiles).max(1).min((shape.k / align).max(1));
        if !shape.a_full_precision && engine == GemmEngine::Mxu && tiling.block_k() != 0 {
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
            || (self.shape.n.is_multiple_of(4) && self.weights_data_type == self.output_data_type)
    }

    fn split_k_is_legal(
        &self,
        plan: GemmPlan,
    ) -> bool {
        plan.split_k == 1 || plan.split_k == self.select_split_k(plan.engine, plan.tiling)
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
        Some(tiling.block_k())
    }
}

fn select_mxu_tiling(shape: MatmulShape) -> Option<GemmTiling> {
    if !shape.a_full_precision {
        return None;
    }
    match shape.b_prologue {
        GemmBPrologueKind::FullPrecision => Some(if shape.b_transpose {
            policy::mxu_fp_tile(shape.m, shape.n, shape.k)
        } else {
            policy::mxu_mn_tile(false, shape.m, shape.n)
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
    let tiling = policy::mxu_mn_tile(!shape.a_full_precision, shape.m, shape.n);
    if tiling.fits_quant_group_size(shape.b_group_size.unwrap_or(0)) {
        tiling
    } else {
        policy::MXU_DEFAULT_TILE
    }
}

fn select_simdgroup_tiling(shape: MatmulShape) -> GemmTiling {
    if shape.is_quant() {
        policy::simdgroup_quant_tile(shape.m, shape.n, shape.b_group_size.unwrap_or(0))
    } else {
        policy::simdgroup_fp_tile(shape.m, shape.n, shape.k)
    }
}
#[cfg(test)]
#[path = "../../../../../../tests/unit/backends/metal/kernel/matmul/gemm/selection_test.rs"]
mod tests;
