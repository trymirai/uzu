use super::QuantParamsLayout;
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
        params_layout: QuantParamsLayout,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    ScaleZeroPointDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        zero_points: &'a Allocation<B>,
        params_layout: QuantParamsLayout,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
    ScaleSymmetricDequant {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        params_layout: QuantParamsLayout,
        mode: QuantizationMode,
        group_size: u32,
        signed_codes: bool,
    },
}

impl<'a, B: Backend, TB: BufferArg<'a, B>> MatmulB<'a, B, TB> {
    fn quant_params(&self) -> Option<(QuantParamsLayout, u32, &Allocation<B>)> {
        match self {
            Self::FullPrecision {
                ..
            } => None,
            Self::ScaleBiasDequant {
                params_layout,
                group_size,
                scales,
                ..
            }
            | Self::ScaleZeroPointDequant {
                params_layout,
                group_size,
                scales,
                ..
            }
            | Self::ScaleSymmetricDequant {
                params_layout,
                group_size,
                scales,
                ..
            } => Some((*params_layout, *group_size, scales)),
        }
    }

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
        self.quant_params().map(|(layout, _, _)| layout)
    }

    pub fn quant_params_stride(
        &self,
        params_data_type: DataType,
        k: u32,
    ) -> u32 {
        let Some((params_layout, group_size, scales)) = self.quant_params() else {
            return 0;
        };
        let groups = k.div_ceil(group_size);
        let element_size = params_data_type.size_in_bytes();
        let elements = scales.size() / element_size;
        let columns = (elements / groups as usize) as u32;
        params_layout.row_stride(columns, groups)
    }
}
