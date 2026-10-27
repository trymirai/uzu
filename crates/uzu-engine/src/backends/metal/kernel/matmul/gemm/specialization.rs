use super::{GemmEngine, GemmPlan, error::GemmSpecializationError, selection::register_quant_plan_supported};
use crate::{
    backends::{
        common::{
            gpu_types::gemm::{GemmBPrologueKind, GemmDTransform},
            kernel::{activation_transform::ACTIVATION_SCALE_GROUP_SIZE, matmul::MatmulShape},
        },
        metal::gpu_types::gemm::{GemmAPrologueKind, GemmAlignment, GemmBLayout, GemmTiling},
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct GemmSpecialization {
    pub(super) weights_data_type: DataType,
    pub(super) tiling: GemmTiling,
    pub(super) use_mxu: bool,
    pub(super) output_transform: GemmDTransform,
    pub(super) alignment: GemmAlignment,
    pub(super) transpose_b: bool,
    pub(super) a_prologue: GemmAPrologueKind,
    pub(super) b_prologue: GemmBPrologueKind,
    pub(super) bits_per_b: Option<u32>,
    pub(super) b_group_size: Option<u32>,
    pub(super) signed_codes: bool,
    pub(super) gemm_b_layout: GemmBLayout,
    pub(super) a_group_size: Option<u32>,
    pub(super) float_partials: bool,
}

impl GemmSpecialization {
    pub(super) fn from_plan(
        plan: GemmPlan,
        shape: MatmulShape,
        gemm_b_layout: GemmBLayout,
        weights_data_type: DataType,
        output_transform: GemmDTransform,
        alignment: GemmAlignment,
        a_prologue: GemmAPrologueKind,
        a_group_size: Option<u32>,
    ) -> Result<Self, GemmSpecializationError> {
        if plan.tiling.is_register_quant_variant() && !register_quant_plan_supported(shape, gemm_b_layout, plan) {
            return Err(GemmSpecializationError::UnsupportedRegisterQuant);
        }
        let specialization = Self {
            weights_data_type,
            tiling: plan.tiling,
            use_mxu: plan.engine == GemmEngine::Mxu,
            output_transform,
            alignment,
            transpose_b: shape.b_transpose,
            a_prologue,
            b_prologue: shape.b_prologue,
            bits_per_b: shape.b_bits,
            b_group_size: shape.b_group_size,
            signed_codes: shape.signed_codes,
            gemm_b_layout,
            a_group_size,
            float_partials: false,
        };
        specialization.validate()?;
        Ok(specialization)
    }

    pub(super) fn validate(&self) -> Result<(), GemmSpecializationError> {
        let valid_activation_group = match self.a_prologue {
            GemmAPrologueKind::FullPrecision => self.a_group_size.is_none(),
            GemmAPrologueKind::Int8Symmetric => self.a_group_size == Some(ACTIVATION_SCALE_GROUP_SIZE),
        };
        if !valid_activation_group {
            return Err(GemmSpecializationError::InvalidAGroupSize {
                a_group_size: self.a_group_size,
            });
        }
        if self.use_mxu != self.tiling.is_mxu_variant() {
            return Err(GemmSpecializationError::TilingUseMxuMismatch {
                tiling: self.tiling,
                use_mxu: self.use_mxu,
            });
        }
        if self.b_prologue != GemmBPrologueKind::FullPrecision && !self.transpose_b {
            return Err(GemmSpecializationError::QuantizedRequiresTransposedB);
        }
        Ok(())
    }
}
