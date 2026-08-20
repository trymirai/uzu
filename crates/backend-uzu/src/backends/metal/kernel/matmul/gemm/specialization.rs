use super::{GemmEngine, GemmPlan, error::GemmSpecializationError};
use crate::{
    backends::common::{
        gpu_types::gemm::{GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
        kernel::matmul::MatmulShape,
    },
    data_type::DataType,
};

const STAGE_WEIGHT_SCALE_MIN_GROUPS: u32 = 6;

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
    pub(super) a_group_size: Option<u32>,
    pub(super) stage_weight_scales: bool,
    pub(super) hoist_operand_addressing: bool,
}

impl GemmSpecialization {
    pub(super) fn from_plan(
        plan: GemmPlan,
        shape: MatmulShape,
        weights_data_type: DataType,
        output_transform: GemmDTransform,
        alignment: GemmAlignment,
        a_prologue: GemmAPrologueKind,
        a_group_size: Option<u32>,
    ) -> Result<Self, GemmSpecializationError> {
        let use_tuned_addressing = shape.is_integer_quantized() || plan.split_k > 1;
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
            a_group_size,
            stage_weight_scales: if use_tuned_addressing {
                plan.should_stage_weight_scales(shape)
            } else {
                true
            },
            hoist_operand_addressing: use_tuned_addressing && plan.should_hoist_operand_addressing(shape),
        };
        specialization.validate()?;
        Ok(specialization)
    }

    pub(super) fn validate(&self) -> Result<(), GemmSpecializationError> {
        let valid_activation_group = match self.a_prologue {
            GemmAPrologueKind::FullPrecision => self.a_group_size.is_none(),
            GemmAPrologueKind::Int8Symmetric => matches!(self.a_group_size, Some(32 | 64 | 128)),
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
        if self.use_mxu
            && self.b_prologue != GemmBPrologueKind::FullPrecision
            && let Some(group_size) = self.b_group_size
            && !self.tiling.fits_quant_group_size(group_size)
        {
            return Err(GemmSpecializationError::MxuQuantTileTooLarge {
                tiling: self.tiling,
                group_size,
            });
        }
        if !self.use_mxu
            && let Some(group_size) = self.b_group_size
        {
            let simdgroup_block_k = self.tiling.simdgroup_block_k();
            if simdgroup_block_k > group_size {
                return Err(GemmSpecializationError::SimdgroupKExceedsGroupSize {
                    simdgroup_k: simdgroup_block_k,
                    group_size,
                });
            }
        }
        if self.b_prologue != GemmBPrologueKind::FullPrecision && !self.transpose_b {
            return Err(GemmSpecializationError::QuantizedRequiresTransposedB);
        }
        Ok(())
    }
}

impl GemmPlan {
    pub(super) fn should_stage_weight_scales(
        self,
        shape: MatmulShape,
    ) -> bool {
        if shape.b_bits == Some(4) && shape.b_group_size == Some(32) && self.tiling.block_m() <= 32 {
            return false;
        }
        if self.tiling == GemmTiling::Tile32x64x256_Simdgroups2x2 {
            return shape
                .b_group_size
                .is_none_or(|group_size| shape.k / self.split_k / group_size >= STAGE_WEIGHT_SCALE_MIN_GROUPS);
        }
        true
    }

    pub(super) fn should_hoist_operand_addressing(
        self,
        shape: MatmulShape,
    ) -> bool {
        if matches!(shape.b_prologue, GemmBPrologueKind::ScaleBiasDequant | GemmBPrologueKind::ScaleZeroPointDequant) {
            return true;
        }
        self.tiling != GemmTiling::Tile128x128x256_Simdgroups4x4
    }
}
