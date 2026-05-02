use crate::backends::metal::kernel::unified_matmul::gemm::{
    GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTile,
    UnifiedGemmSpecializationError, WeightsStorageFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UnifiedGemmSpecialization {
    pub(crate) tile: GemmTile,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) compute: GemmComputeKind,
    pub(crate) output: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) weights_storage: WeightsStorageFormat,
}

impl UnifiedGemmSpecialization {
    pub(crate) const fn new(
        tile: GemmTile,
        input_prologue: GemmInputPrologueKind,
        compute: GemmComputeKind,
        output: GemmOutputTransformKind,
        alignment: GemmAlignment,
        weights_storage: WeightsStorageFormat,
    ) -> Self {
        Self {
            tile,
            input_prologue,
            compute,
            output,
            alignment,
            weights_storage,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), UnifiedGemmSpecializationError> {
        self.tile.validate()?;

        let group_size = self.weights_storage.group_size();
        if group_size != 0 && self.tile.threadgroup_k > group_size {
            return Err(UnifiedGemmSpecializationError::ThreadgroupKExceedsGroupSize {
                threadgroup_k: self.tile.threadgroup_k,
                group_size,
            });
        }

        Ok(())
    }
}
