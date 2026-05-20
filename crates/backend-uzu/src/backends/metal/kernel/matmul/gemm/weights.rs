use crate::{
    DataType,
    backends::common::{
        Allocation, Backend,
        gpu_types::{QuantizationMode, gemm::GemmWeightPrologueKind},
    },
};

#[allow(dead_code)]
pub enum GemmWeights<'a, B: Backend> {
    FullPrecision {
        weights: &'a Allocation<B>,
    },
    ScaleBias {
        weights: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        biases: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
    ScaleZeroPoint {
        weights: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
}

impl<B: Backend> GemmWeights<'_, B> {
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
                mode,
                ..
            }
            | Self::ScaleZeroPoint {
                mode,
                ..
            } => DataType::from(*mode).size_in_bits() as u32,
        }
    }

    pub fn group_size(&self) -> u32 {
        match self {
            Self::FullPrecision {
                ..
            } => 0,
            Self::ScaleBias {
                group_size,
                ..
            }
            | Self::ScaleZeroPoint {
                group_size,
                ..
            } => *group_size,
        }
    }
}
