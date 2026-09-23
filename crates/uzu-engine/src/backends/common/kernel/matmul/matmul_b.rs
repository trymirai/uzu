use super::{QuantParams, QuantParamsLayout};
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
        params: QuantParams,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    ScaleZeroPointDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        params: QuantParams,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    ScaleSymmetricDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        params: QuantParams,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
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

    pub fn signed_codes(&self) -> bool {
        match self {
            Self::FullPrecision {
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

    pub fn quant_params_layout(&self) -> Option<QuantParamsLayout> {
        self.quant_params().map(QuantParams::layout)
    }

    pub fn quant_params(&self) -> Option<QuantParams> {
        match self {
            Self::FullPrecision {
                ..
            } => None,
            Self::ScaleBiasDequant {
                params,
                ..
            }
            | Self::ScaleZeroPointDequant {
                params,
                ..
            }
            | Self::ScaleSymmetricDequant {
                params,
                ..
            } => Some(*params),
        }
    }
}
