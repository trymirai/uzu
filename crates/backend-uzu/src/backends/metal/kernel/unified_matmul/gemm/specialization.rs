use crate::backends::metal::kernel::unified_matmul::gemm::{
    GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTile, GemmWeightPrologueKind,
    QuantizedStorageFormat, UnifiedGemmSpecializationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct UnifiedGemmSpecialization {
    pub(crate) tile: GemmTile,
    pub(crate) input_prologue: GemmInputPrologueKind,
    pub(crate) weight_prologue: GemmWeightPrologueKind,
    pub(crate) compute: GemmComputeKind,
    pub(crate) output: GemmOutputTransformKind,
    pub(crate) alignment: GemmAlignment,
    pub(crate) quantized_storage: Option<QuantizedStorageFormat>,
}

impl UnifiedGemmSpecialization {
    pub(crate) const fn full_precision_simdgroup(
        tile: GemmTile,
        output: GemmOutputTransformKind,
        alignment: GemmAlignment,
    ) -> Self {
        Self {
            tile,
            input_prologue: GemmInputPrologueKind::FullPrecision,
            weight_prologue: GemmWeightPrologueKind::FullPrecision,
            compute: GemmComputeKind::SimdgroupMma,
            output,
            alignment,
            quantized_storage: None,
        }
    }

    pub(crate) const fn full_precision_mxu_mma(
        tile: GemmTile,
        output: GemmOutputTransformKind,
        alignment: GemmAlignment,
    ) -> Self {
        Self {
            tile,
            input_prologue: GemmInputPrologueKind::FullPrecision,
            weight_prologue: GemmWeightPrologueKind::FullPrecision,
            compute: GemmComputeKind::MxuMma,
            output,
            alignment,
            quantized_storage: None,
        }
    }

    pub(crate) const fn quantized_simdgroup(
        tile: GemmTile,
        weight_prologue: GemmWeightPrologueKind,
        output: GemmOutputTransformKind,
        alignment: GemmAlignment,
        quantized_storage: QuantizedStorageFormat,
    ) -> Self {
        Self {
            tile,
            input_prologue: GemmInputPrologueKind::FullPrecision,
            weight_prologue,
            compute: GemmComputeKind::SimdgroupMma,
            output,
            alignment,
            quantized_storage: Some(quantized_storage),
        }
    }

    pub(crate) const fn quantized_mxu_mma(
        tile: GemmTile,
        weight_prologue: GemmWeightPrologueKind,
        output: GemmOutputTransformKind,
        alignment: GemmAlignment,
        quantized_storage: QuantizedStorageFormat,
    ) -> Self {
        Self {
            tile,
            input_prologue: GemmInputPrologueKind::FullPrecision,
            weight_prologue,
            compute: GemmComputeKind::MxuMma,
            output,
            alignment,
            quantized_storage: Some(quantized_storage),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), UnifiedGemmSpecializationError> {
        self.tile.validate()?;

        match (self.weight_prologue, self.quantized_storage) {
            (GemmWeightPrologueKind::FullPrecision, None) => {},
            (GemmWeightPrologueKind::MlxDequant | GemmWeightPrologueKind::AwqDequant, Some(storage)) => {
                storage.validate()?;
                if self.tile.threadgroup_k > storage.group_size {
                    return Err(UnifiedGemmSpecializationError::ThreadgroupKExceedsGroupSize {
                        threadgroup_k: self.tile.threadgroup_k,
                        group_size: storage.group_size,
                    });
                }
            },
            (GemmWeightPrologueKind::FullPrecision, Some(_)) => {
                return Err(UnifiedGemmSpecializationError::UnexpectedQuantizedStorage);
            },
            (GemmWeightPrologueKind::MlxDequant | GemmWeightPrologueKind::AwqDequant, None) => {
                return Err(UnifiedGemmSpecializationError::MissingQuantizedStorage);
            },
        }

        Ok(())
    }
}
