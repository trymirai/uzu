use super::error::GemmSpecializationError;
use crate::{
    backends::common::gpu_types::gemm::{GemmAlignment, GemmBPrologueKind, GemmDTransform, GemmTiling},
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) weights_data_type: DataType,
    pub(crate) tiling: GemmTiling,
    pub(crate) use_mxu: bool,
    pub(crate) output_transform: GemmDTransform,
    pub(crate) alignment: GemmAlignment,
    pub(crate) transpose_b: bool,
    pub(crate) b_prologue: GemmBPrologueKind,
    pub(crate) bits_per_b: Option<u32>,
    pub(crate) group_size: Option<u32>,
}

impl GemmSpecialization {
    pub(crate) fn validate(&self) -> Result<(), GemmSpecializationError> {
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
