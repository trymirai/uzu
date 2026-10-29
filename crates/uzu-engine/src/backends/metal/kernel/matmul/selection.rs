use metal::MTLGPUFamily;

use super::{
    gemm::{GemmEngine, GemmPlan, GemmProblem, MXU_INT8_MIN_BATCH},
    gemv::GemvSpecialization,
};
use crate::{
    backends::{
        common::kernel::matmul::{ActivationFormat, MatmulShape},
        metal::gpu_types::gemm::{GemmBLayout, GemmTiling},
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatmulDispatch {
    Gemv(GemvSpecialization),
    Gemm(GemmPlan),
}

pub struct MatmulSelection {
    pub activation_format: ActivationFormat,
    pub dispatch: MatmulDispatch,
}

pub struct MatmulSelector {
    pub gpu_core_count: u32,
    pub apple_gpu_family: MTLGPUFamily,
    pub supports_mxu: bool,
    pub data_types: (DataType, DataType, DataType),
}

impl MatmulSelector {
    fn problem(
        &self,
        shape: MatmulShape,
    ) -> GemmProblem {
        GemmProblem::new(shape, self.data_types, self.supports_mxu, self.apple_gpu_family)
    }

    fn prefer_gemm_over_gemv(
        &self,
        shape: MatmulShape,
        plan: GemmPlan,
    ) -> bool {
        let (weights_data_type, input_data_type, output_data_type) = self.data_types;
        if shape.gathered || plan.engine != GemmEngine::Mxu {
            return false;
        }
        if shape.m == MXU_INT8_MIN_BATCH
            && (weights_data_type, input_data_type, output_data_type)
                == (DataType::BF16, DataType::BF16, DataType::BF16)
        {
            return false;
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
        matches!(
            plan.tiling,
            GemmTiling::Tile16x32x256_Simdgroups1x1
                | GemmTiling::Tile16x32x256_Simdgroups1x1x4
                | GemmTiling::Tile16x128x256_Simdgroups1x4
        )
    }

    fn dispatch(
        &self,
        shape: &MatmulShape,
        gemm_b_layout: GemmBLayout,
    ) -> MatmulDispatch {
        let (weights_data_type, input_data_type, output_data_type) = self.data_types;
        let all_bf16 = weights_data_type == DataType::BF16
            && input_data_type == DataType::BF16
            && output_data_type == DataType::BF16;
        let problem = self.problem(*shape);
        if shape.b_is_trellis {
            return MatmulDispatch::Gemm(problem.select_plan());
        }
        if let Some(plan) = problem.small_batch_plan(gemm_b_layout) {
            return MatmulDispatch::Gemm(plan);
        }
        if all_bf16 && let Some(gemv) = GemvSpecialization::select_batched(shape, gemm_b_layout, self.supports_mxu) {
            return MatmulDispatch::Gemv(gemv);
        }
        let gemv = GemvSpecialization::select_shape(
            shape,
            gemm_b_layout,
            weights_data_type,
            input_data_type,
            output_data_type,
            self.gpu_core_count,
            self.apple_gpu_family,
        );
        let plan = problem.select_plan();
        match gemv {
            None => MatmulDispatch::Gemm(plan),
            Some(_) if self.prefer_gemm_over_gemv(*shape, plan) => MatmulDispatch::Gemm(plan),
            Some(gemv) => MatmulDispatch::Gemv(gemv),
        }
    }

    pub fn activation_preparation(
        &self,
        shape: &MatmulShape,
        layout: GemmBLayout,
    ) -> ActivationFormat {
        self.problem(*shape).activation_preparation(layout)
    }

    pub fn select(
        &self,
        shape: &MatmulShape,
        layout: GemmBLayout,
        prepared: ActivationFormat,
    ) -> MatmulSelection {
        if shape.a != ActivationFormat::Bf16 || prepared == ActivationFormat::Bf16 {
            return MatmulSelection {
                activation_format: shape.a,
                dispatch: self.dispatch(shape, layout),
            };
        }
        let prepared_shape = MatmulShape {
            a: prepared,
            ..*shape
        };
        let prepared_plan = self.problem(prepared_shape).prepared_plan(layout);
        if let Some(plan) = prepared_plan
            && (matches!(prepared, ActivationFormat::PackedHalf { .. })
                || (layout == GemmBLayout::MxuTiled && shape.m >= MXU_INT8_MIN_BATCH))
        {
            return MatmulSelection {
                activation_format: prepared,
                dispatch: MatmulDispatch::Gemm(plan),
            };
        }
        let bf16_dispatch = self.dispatch(shape, layout);
        if let Some(plan) = prepared_plan
            && matches!(bf16_dispatch, MatmulDispatch::Gemm(bf16_plan) if !bf16_plan.tiling.is_small_batch_variant())
        {
            return MatmulSelection {
                activation_format: prepared,
                dispatch: MatmulDispatch::Gemm(plan),
            };
        }
        MatmulSelection {
            activation_format: ActivationFormat::Bf16,
            dispatch: bf16_dispatch,
        }
    }
}

#[cfg(test)]
#[path = "../../../../../unit/backends/metal/kernel/matmul/selection_test.rs"]
mod tests;
