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
        let a_is_int8 = self.a_prologue == GemmAPrologueKind::Int8Symmetric;
        let b_ok_for_int8 = matches!(self.b_prologue, GemmBPrologueKind::ScaleSymmetricDequant);
        let bits_ok = matches!(self.bits_per_b, Some(4) | Some(8));
        if a_is_int8 && !(self.use_mxu && bits_ok && b_ok_for_int8 && self.transpose_b) {
            return Err(GemmSpecializationError::Int8ActivationUnsupported {
                use_mxu: self.use_mxu,
                bits: self.bits_per_b,
                b_prologue: self.b_prologue,
                transpose_b: self.transpose_b,
            });
        }
        Ok(())
    }
}
