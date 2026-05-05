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
    Mlx {
        weights: S::Buffer,
        scales: S::Buffer,
        biases: S::Buffer,
        mode: QuantizationMode,
        group_size: u32,
    },
    Awq {
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
            Self::Mlx {
                ..
            } => GemmWeightPrologueKind::MlxDequant,
            Self::Awq {
                ..
            } => GemmWeightPrologueKind::AwqDequant,
        }
    }

    pub(crate) fn bits_per_weight(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::Mlx {
                mode, ..
            }
            | Self::Awq {
                mode, ..
            } => DataType::from(*mode).size_in_bits() as u32,
        }
    }

    pub(crate) fn group_size(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::Mlx {
                group_size, ..
            }
            | Self::Awq {
                group_size, ..
            } => *group_size,
        }
    }
}
