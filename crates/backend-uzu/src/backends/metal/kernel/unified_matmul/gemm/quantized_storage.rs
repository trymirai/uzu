use crate::backends::{
    common::gpu_types::unified_gemm::{GemmWeightPrologueKind, QuantizedMetadataKind},
    metal::kernel::unified_matmul::gemm::UnifiedGemmSpecializationError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WeightsStorageFormat {
    FullPrecision,
    QuantizedMLXScaleBias {
        bits_per_weight: u8,
        group_size: u32,
    },
    QuantizedAwqScaleZeroPoint {
        bits_per_weight: u8,
        group_size: u32,
    },
}

impl WeightsStorageFormat {
    pub(crate) const fn weight_prologue(self) -> GemmWeightPrologueKind {
        match self {
            Self::FullPrecision => GemmWeightPrologueKind::FullPrecision,
            Self::QuantizedMLXScaleBias {
                ..
            } => GemmWeightPrologueKind::MlxDequant,
            Self::QuantizedAwqScaleZeroPoint {
                ..
            } => GemmWeightPrologueKind::AwqDequant,
        }
    }

    pub(crate) const fn bits_per_weight(self) -> u32 {
        match self {
            Self::FullPrecision => 0,
            Self::QuantizedMLXScaleBias {
                bits_per_weight,
                ..
            }
            | Self::QuantizedAwqScaleZeroPoint {
                bits_per_weight,
                ..
            } => bits_per_weight as u32,
        }
    }

    pub(crate) const fn group_size(self) -> u32 {
        match self {
            Self::FullPrecision => 0,
            Self::QuantizedMLXScaleBias {
                group_size,
                ..
            }
            | Self::QuantizedAwqScaleZeroPoint {
                group_size,
                ..
            } => group_size,
        }
    }

    pub(crate) const fn metadata_kind(self) -> QuantizedMetadataKind {
        match self {
            Self::FullPrecision
            | Self::QuantizedMLXScaleBias {
                ..
            } => QuantizedMetadataKind::MlxScaleBias,
            Self::QuantizedAwqScaleZeroPoint {
                ..
            } => QuantizedMetadataKind::AwqScaleZeroPoint,
        }
    }

    pub(crate) fn validate(self) -> Result<(), UnifiedGemmSpecializationError> {
        let (bits_per_weight, group_size) = match self {
            Self::FullPrecision => return Ok(()),
            Self::QuantizedMLXScaleBias {
                bits_per_weight,
                group_size,
            }
            | Self::QuantizedAwqScaleZeroPoint {
                bits_per_weight,
                group_size,
            } => (bits_per_weight, group_size),
        };

        if !(1..=8).contains(&bits_per_weight) {
            return Err(UnifiedGemmSpecializationError::UnsupportedBitsPerWeight(bits_per_weight));
        }
        if group_size == 0 {
            return Err(UnifiedGemmSpecializationError::ZeroGroupSize);
        }
        Ok(())
    }
}
