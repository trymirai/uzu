use crate::backends::{
    common::gpu_types::unified_gemm::GemmWeightPrologueKind,
    metal::kernel::unified_matmul::gemm::QuantizedFormat,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WeightsStorageFormat {
    FullPrecision,
    Quantized(QuantizedFormat),
}

impl WeightsStorageFormat {
    pub(crate) const fn quantized(self) -> Option<QuantizedFormat> {
        match self {
            Self::FullPrecision => None,
            Self::Quantized(format) => Some(format),
        }
    }

    pub(crate) const fn weight_prologue(self) -> GemmWeightPrologueKind {
        match self {
            Self::FullPrecision => GemmWeightPrologueKind::FullPrecision,
            Self::Quantized(QuantizedFormat::MLX(_)) => GemmWeightPrologueKind::MlxDequant,
            Self::Quantized(QuantizedFormat::AWQ(_)) => GemmWeightPrologueKind::AwqDequant,
        }
    }

    pub(crate) const fn bits_per_weight(self) -> u32 {
        match self {
            Self::FullPrecision => 0,
            Self::Quantized(format) => format.params().bits.as_u32(),
        }
    }

    pub(crate) const fn group_size(self) -> u32 {
        match self {
            Self::FullPrecision => 0,
            Self::Quantized(format) => format.params().group_size.get(),
        }
    }
}
