use crate::{
    DataType,
    backends::common::{
        Allocation, AsBufferRangeRef, Backend,
        gpu_types::{QuantizationMode, gemm::GemmWeightPrologueKind},
    },
};

/// Backend-agnostic description of how B is laid out for the unified GEMM
/// kernel. Mirrors `MatmulB` after destructuring + tagging with the kernel-
/// binary's [`GemmWeightPrologueKind`].
///
/// `TB` is the buffer type for the full-precision weights operand; defaults to
/// [`Allocation<B>`] but can be specialized to KV-cache buffer types so long
/// as `&TB: AsBufferRangeRef`. Quantized variants always use [`Allocation<B>`].
#[allow(dead_code)]
pub enum GemmWeights<'a, B: Backend, TB: AsBufferRangeRef = Allocation<B>> {
    FullPrecision {
        weights: &'a TB,
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

impl<B: Backend, TB: AsBufferRangeRef> GemmWeights<'_, B, TB> {
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
