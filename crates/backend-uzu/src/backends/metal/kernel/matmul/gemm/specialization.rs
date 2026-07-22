use super::error::GemmSpecializationError;
use crate::{
    backends::common::gpu_types::gemm::{GemmAlignment, GemmDTransform, GemmTiling, MXU_BLOCK_K, WeightsKey},
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GemmSpecialization {
    pub(crate) weights_data_type: DataType,
    pub(crate) tiling: GemmTiling,
    pub(crate) output_transform: GemmDTransform,
    pub(crate) alignment: GemmAlignment,
    pub(crate) transpose_b: bool,
    pub(crate) weights: WeightsKey,
}

impl GemmSpecialization {
    pub(crate) fn validate(&self) -> Result<(), GemmSpecializationError> {
        let use_mxu = self.tiling.block_k() == MXU_BLOCK_K;
        if use_mxu
            && let Some(group_size) = self.weights.group_size()
            && !self.tiling.fits_quant_group_size(group_size)
        {
            return Err(GemmSpecializationError::MxuQuantTileTooLarge {
                tiling: self.tiling,
                group_size,
            });
        }
        if !use_mxu && let Some(group_size) = self.weights.group_size() {
            let simdgroup_block_k = self.tiling.simdgroup_block_k();
            if simdgroup_block_k > group_size {
                return Err(GemmSpecializationError::SimdgroupKExceedsGroupSize {
                    simdgroup_k: simdgroup_block_k,
                    group_size,
                });
            }
        }
        // Mirrors the gemm.metal CONSTRAINT gating which quantized variants exist:
        // quantized => TRANSPOSE_B && (tiling != Tile64x64x16 || GROUP_SIZE == 16).
        if let Some(group_size) = self.weights.group_size() {
            let tiling_instantiated = self.tiling != GemmTiling::Tile64x64x16_Simdgroups2x2 || group_size == 16;
            if !self.transpose_b || !tiling_instantiated {
                return Err(GemmSpecializationError::QuantizedVariantNotInstantiated {
                    tiling: self.tiling,
                    group_size,
                    transpose_b: self.transpose_b,
                });
            }
        }
        Ok(())
    }
}
