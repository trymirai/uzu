use crate::{
    backends::common::{
        Allocation, Backend, BufferArg,
        gpu_types::{
            QuantizationMode,
            gemm::{QuantBits, QuantGroupSize, QuantPrologue, WeightsKey},
        },
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
    /// `None` when the operand's bit width or group size is outside the set the kernels
    /// are instantiated for, i.e. no GEMM/GEMV variant can serve this B at all.
    pub fn weights_key(&self) -> Option<WeightsKey> {
        let (b_prologue, mode, group_size) = match self {
            Self::FullPrecision {
                ..
            } => return Some(WeightsKey::FullPrecision),
            Self::ScaleBiasDequant {
                mode,
                group_size,
                ..
            } => (QuantPrologue::ScaleBiasDequant, mode, group_size),
            Self::ScaleZeroPointDequant {
                mode,
                group_size,
                ..
            } => (QuantPrologue::ScaleZeroPointDequant, mode, group_size),
            Self::ScaleSymmetricDequant {
                mode,
                group_size,
                ..
            } => (QuantPrologue::ScaleSymmetricDequant, mode, group_size),
        };
        Some(WeightsKey::Quant {
            b_prologue,
            bits: QuantBits::new(DataType::from(*mode).size_in_bits() as u32)?,
            group_size: QuantGroupSize::new(*group_size)?,
        })
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
