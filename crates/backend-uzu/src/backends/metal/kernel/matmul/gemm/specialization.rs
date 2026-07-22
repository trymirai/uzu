use super::error::GemmSpecializationError;
use crate::{
    backends::common::gpu_types::gemm::{
        GemmAPrologueKind, GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling,
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
    pub(super) group_size: Option<u32>,
}

impl GemmSpecialization {
    pub(super) fn validate(&self) -> Result<(), GemmSpecializationError> {
        if self.use_mxu != self.tiling.is_mxu_variant() {
            return Err(GemmSpecializationError::TilingUseMxuMismatch {
                tiling: self.tiling,
                use_mxu: self.use_mxu,
            });
        }
        if self.use_mxu
            && self.b_prologue != GemmBPrologueKind::FullPrecision
            && let Some(group_size) = self.group_size
            && !self.tiling.fits_quant_group_size(group_size)
        {
            return Err(GemmSpecializationError::MxuQuantTileTooLarge {
                tiling: self.tiling,
                group_size,
            });
        }
        if !self.use_mxu
            && let Some(group_size) = self.group_size
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
