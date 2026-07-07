use crate::{
    backends::common::{
        Allocation, Backend, BufferArg,
        gpu_types::{QuantizationMode, gemm::GemmBPrologueKind},
    },
    data_type::DataType,
};

pub enum MatmulB<'a, B: Backend, TB: BufferArg<'a, B> = &'a Allocation<B>> {
    FullPrecision {
        b: TB,
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
    ScaleSymmetricDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
}

impl<'a, B: Backend, TB: BufferArg<'a, B>> MatmulB<'a, B, TB> {
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
            Self::ScaleSymmetricDequant {
                ..
            } => GemmBPrologueKind::ScaleSymmetricDequant,
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
            }
            | Self::ScaleSymmetricDequant {
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
            }
            | Self::ScaleSymmetricDequant {
                group_size,
                ..
            } => Some(*group_size),
        }
    }
}
