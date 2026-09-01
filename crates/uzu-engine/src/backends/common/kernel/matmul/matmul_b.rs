use crate::{
    backends::common::{
        Allocation, Backend, BufferArg,
        gpu_types::{QuantizationMode, gemm::GemmBPrologueKind},
        microfloat::{MicrofloatFormat, MicrofloatMetadata},
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulBKind {
    Dense,
    Integer,
    Mxfp4,
}

pub enum MatmulB<'a, B: Backend, TB: BufferArg<'a, B> = &'a Allocation<B>> {
    FullPrecision {
        b: TB,
    },
    Microfloat {
        codes: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        outer_scales: &'a Allocation<B>,
        metadata: MicrofloatMetadata,
    },
    ScaleBiasDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        biases: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    ScaleZeroPointDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    ScaleSymmetricDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
}

impl<'a, B: Backend, TB: BufferArg<'a, B>> MatmulB<'a, B, TB> {
    pub fn kind(&self) -> MatmulBKind {
        match self {
            Self::FullPrecision {
                ..
            } => MatmulBKind::Dense,
            Self::Microfloat {
                metadata,
                ..
            } => match metadata.encoding.format {
                MicrofloatFormat::Mxfp4 => MatmulBKind::Mxfp4,
            },
            Self::ScaleBiasDequant {
                ..
            }
            | Self::ScaleZeroPointDequant {
                ..
            }
            | Self::ScaleSymmetricDequant {
                ..
            } => MatmulBKind::Integer,
        }
    }

    pub fn b_prologue(&self) -> GemmBPrologueKind {
        match self {
            Self::FullPrecision {
                ..
            }
            | Self::Microfloat {
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
            Self::Microfloat {
                metadata,
                ..
            } => Some(metadata.encoding.bits),
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
            Self::Microfloat {
                metadata,
                ..
            } => Some(metadata.encoding.group_size),
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

    pub fn signed_codes(&self) -> bool {
        match self {
            Self::FullPrecision {
                ..
            }
            | Self::Microfloat {
                ..
            } => false,
            Self::ScaleBiasDequant {
                signed_codes,
                ..
            }
            | Self::ScaleZeroPointDequant {
                signed_codes,
                ..
            }
            | Self::ScaleSymmetricDequant {
                signed_codes,
                ..
            } => *signed_codes,
        }
    }
}
