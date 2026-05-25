use crate::{
    DataType,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend,
        gpu_types::{QuantizationMode, gemm::GemmBPrologueKind},
    },
};

pub enum MatmulB<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    FullPrecision {
        b: &'a TB,
    },
    ScaleBiasDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        biases: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
    ScaleZeroPointDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
}

impl<B: Backend, TB: AsBufferRangeRef> MatmulB<'_, B, TB> {
    pub fn b_prologue(&self) -> GemmBPrologueKind {
        match self {
            Self::FullPrecision {
                ..
            } => GemmBPrologueKind::FullPrecision,
            Self::ScaleBiasDequant {
                ..
            } => GemmBPrologueKind::ScaleBiasDequant,
            Self::ScaleZeroPointDequant {
                ..
            } => GemmBPrologueKind::ScaleZeroPointDequant,
        }
    }

    pub fn bits_per_b(&self) -> Option<u32> {
        match self {
            Self::FullPrecision {
                ..
            } => None,
            Self::ScaleBiasDequant {
                mode,
                ..
            }
            | Self::ScaleZeroPointDequant {
                mode,
                ..
            } => Some(DataType::from(*mode).size_in_bits() as u32),
        }
    }

    pub fn group_size(&self) -> Option<u32> {
        match self {
            Self::FullPrecision {
                ..
            } => None,
            Self::ScaleBiasDequant {
                group_size,
                ..
            }
            | Self::ScaleZeroPointDequant {
                group_size,
                ..
            } => Some(*group_size),
        }
    }
}
