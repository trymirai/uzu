use super::trellis_format::{TRELLIS_BLOCK_K, TrellisConfig};
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
    /// QTIP bitshift-trellis weights: `b` is one bit tape per row and `scales`
    /// one scale per row. There are no per-group scales, biases or zero points
    /// and no stored codes — the weights are hashed out of `config` plus the
    /// tape, and the decode lands directly in int8.
    ///
    /// Packing, codebook construction and the decode oracle live in
    /// [`trellis_format`](super::trellis_format); the device-side decode lives
    /// in `metal/kernel/matmul/common/trellis_decode.h`.
    ///
    /// Requires int8 activations and the MXU: the GEMM decodes a
    /// `BLOCK_N x TRELLIS_BLOCK_K` int8 weight block into threadgroup memory and
    /// feeds the shipped integer schedule from there.
    Trellis {
        b: &'a Allocation<B>,
        scales: &'a Allocation<B>,
        config: TrellisConfig,
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
            Self::Trellis {
                ..
            } => GemmBPrologueKind::Trellis,
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
            // The width of the operand the MXU multiplies, which is what every
            // consumer of this actually asks about. A trellis tape has no
            // addressable per-weight code width — the rate is `k + (L - k*V) /
            // cols` and the states overlap — but the decode's output is int8, so
            // reporting 8 is what routes this to the int8 MMA path.
            Self::Trellis {
                ..
            } => Some(8),
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
            // Not a quantization group: there is one scale per row. This is the
            // staging block K, reported here so the shipped integer schedule and
            // split-K policy — both written against `outer_block_k() ==
            // GROUP_SIZE` — apply unchanged. See `TRELLIS_BLOCK_K`.
            Self::Trellis {
                ..
            } => Some(TRELLIS_BLOCK_K),
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
            // The decode emits two's-complement int8; there is no stored code
            // whose signedness could differ.
            Self::Trellis {
                ..
            } => true,
        }
    }
}
