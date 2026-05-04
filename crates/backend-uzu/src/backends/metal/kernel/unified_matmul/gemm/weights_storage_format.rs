use crate::{
    DataType,
    backends::common::gpu_types::{
        QuantizationMode, QuantizedFormat,
        unified_gemm::GemmWeightPrologueKind,
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WeightsStorageFormat {
    FullPrecision,
    Quantized {
        format: QuantizedFormat,
        mode: QuantizationMode,
        group_size: u32,
    },
}

impl WeightsStorageFormat {
    pub(crate) const fn weight_prologue(self) -> GemmWeightPrologueKind {
        match self {
            Self::FullPrecision => GemmWeightPrologueKind::FullPrecision,
            Self::Quantized {
                format: QuantizedFormat::MLX,
                ..
            } => GemmWeightPrologueKind::MlxDequant,
            Self::Quantized {
                format: QuantizedFormat::AWQ,
                ..
            } => GemmWeightPrologueKind::AwqDequant,
        }
    }

    pub(crate) fn bits_per_weight(self) -> u32 {
        match self {
            Self::FullPrecision => 0,
            Self::Quantized {
                mode,
                ..
            } => DataType::from(mode).size_in_bits() as u32,
        }
    }

    pub(crate) const fn group_size(self) -> u32 {
        match self {
            Self::FullPrecision => 0,
            Self::Quantized {
                group_size,
                ..
            } => group_size,
        }
    }

}
