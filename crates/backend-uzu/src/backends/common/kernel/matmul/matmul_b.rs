use crate::{
    backends::common::{
        Allocation, AsBufferRangeRef, Backend,
        gpu_types::{QuantizationMode, gemm::GemmBPrologueKind},
    },
    data_type::DataType,
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
    ScaleSymmetricDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
    },
    LloydMaxDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        codebook: &'a Allocation<B>,
        bias_indices: &'a Allocation<B>,
        bias_codebook: &'a Allocation<B>,
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
            Self::ScaleSymmetricDequant {
                ..
            } => GemmBPrologueKind::ScaleSymmetricDequant,
            Self::LloydMaxDequant {
                ..
            } => GemmBPrologueKind::LloydMaxDequant,
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
            }
            | Self::LloydMaxDequant {
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
            }
            | Self::LloydMaxDequant {
                group_size,
                ..
            } => Some(*group_size),
        }
    }
}
