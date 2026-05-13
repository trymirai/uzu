use crate::{
    DataType,
    backends::common::{
        Storage,
        gpu_types::{QuantizationMode, unified_gemm::GemmWeightPrologueKind},
    },
};

pub(crate) enum GemmWeights<S: Storage> {
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

impl<S: Storage> GemmWeights<S> {
    pub(crate) fn weight_prologue(&self) -> GemmWeightPrologueKind {
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

    pub(crate) fn bits_per_weight(&self) -> u32 {
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

    pub(crate) fn group_size(&self) -> u32 {
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
