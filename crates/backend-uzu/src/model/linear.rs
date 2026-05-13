use crate::{
    DataType,
    backends::common::{
        Storage,
        gpu_types::{QuantizationMode, unified_gemm::GemmWeightPrologueKind},
    },
};

/// Layout of the weights for a linear layer, parameterised over the storage
/// carrier `S` (e.g. `Borrowed<'a, _>` at encode time, `Owned<_>` for loaded
/// weights, `Schema` for config-only views).
pub enum LinearWeights<S: Storage> {
    FullPrecision {
        weights: S::Buffer,
    },
    ScaleBias {
        weights: S::Buffer,
        scales: S::Buffer,
        biases: S::Buffer,
        mode: QuantizationMode,
        group_size: u32,
    },
    ScaleZeroPoint {
        weights: S::Buffer,
        scales: S::Buffer,
        zero_points: S::Buffer,
        mode: QuantizationMode,
        group_size: u32,
    },
}

impl<S: Storage> LinearWeights<S> {
    pub fn weight_prologue(&self) -> GemmWeightPrologueKind {
        match self {
            Self::FullPrecision {
                ..
            } => GemmWeightPrologueKind::FullPrecision,
            Self::ScaleBias {
                ..
            } => GemmWeightPrologueKind::ScaleBiasDequant,
            Self::ScaleZeroPoint {
                ..
            } => GemmWeightPrologueKind::ScaleZeroPointDequant,
        }
    }

    pub fn bits_per_weight(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::ScaleBias {
                mode, ..
            }
            | Self::ScaleZeroPoint {
                mode, ..
            } => DataType::from(*mode).size_in_bits() as u32,
        }
    }

    pub fn group_size(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::ScaleBias {
                group_size, ..
            }
            | Self::ScaleZeroPoint {
                group_size, ..
            } => *group_size,
        }
    }
}
