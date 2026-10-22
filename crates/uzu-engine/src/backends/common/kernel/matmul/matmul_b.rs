use super::{QuantParams, QuantParamsLayout, WeightsLayout};
use crate::{
    backends::common::{
        BufferRef,
        gpu_types::{gemm::GemmBPrologueKind, quantization::QuantizationMode},
    },
    data_type::DataType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrellisFormat {
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: Option<u32>,
}

/// Readable bytes past the last code row (the GEMM loads whole words).
#[cfg(backend = "metal")]
pub const TRELLIS_CODES_TAIL_BYTES: usize = 16;

impl TrellisFormat {
    /// Bytes per code row: restart rows pad to 16 bytes, connected rows to a byte.
    pub fn code_row_bytes(
        self,
        columns: u32,
    ) -> u32 {
        let block_columns = self.restart_columns.unwrap_or(columns);
        let block_bits = 16 + (block_columns / self.vector_width - 1) * self.transition_bits;
        let bytes = (columns / block_columns * block_bits).div_ceil(8);
        if self.restart_columns.is_some() {
            bytes.next_multiple_of(16)
        } else {
            bytes
        }
    }
}

pub enum MatmulB<TB: BufferRef> {
    FullPrecision {
        b: TB,
    },
    Quantized(QuantizedB<TB>),
    Trellis {
        codes: TB,
        row_scales: TB,
        codebook: TB,
        format: TrellisFormat,
    },
}

pub struct QuantizedB<TB: BufferRef> {
    pub codes: TB,
    pub scales: TB,
    pub correction: QuantizedCorrection<TB>,
    pub params: QuantParams,
    pub mode: QuantizationMode,
    pub group_size: u32,
    pub signed_codes: bool,
    pub weights_layout: WeightsLayout<TB::Backend>,
}

#[derive(Clone, Copy)]
pub enum QuantizedCorrection<T> {
    Symmetric,
    Biases(T),
    ZeroPoints(T),
}

impl<T> QuantizedCorrection<T> {
    pub fn as_ref(&self) -> QuantizedCorrection<&T> {
        match self {
            Self::Symmetric => QuantizedCorrection::Symmetric,
            Self::Biases(biases) => QuantizedCorrection::Biases(biases),
            Self::ZeroPoints(zero_points) => QuantizedCorrection::ZeroPoints(zero_points),
        }
    }

    pub fn biases(&self) -> Option<&T> {
        match self {
            Self::Biases(biases) => Some(biases),
            Self::Symmetric | Self::ZeroPoints(_) => None,
        }
    }

    pub fn zero_points(&self) -> Option<&T> {
        match self {
            Self::ZeroPoints(zero_points) => Some(zero_points),
            Self::Symmetric | Self::Biases(_) => None,
        }
    }
}

impl<TB: BufferRef> QuantizedB<TB> {
    pub fn bits(&self) -> u32 {
        DataType::from(self.mode).size_in_bits() as u32
    }

    pub fn prologue(&self) -> GemmBPrologueKind {
        match self.correction {
            QuantizedCorrection::Symmetric => GemmBPrologueKind::ScaleSymmetricDequant,
            QuantizedCorrection::Biases(_) => GemmBPrologueKind::ScaleBiasDequant,
            QuantizedCorrection::ZeroPoints(_) => GemmBPrologueKind::ScaleZeroPointDequant,
        }
    }

    pub fn biases(&self) -> Option<TB> {
        self.correction.biases().copied()
    }

    pub fn zero_points(&self) -> Option<TB> {
        self.correction.zero_points().copied()
    }

    pub fn zero_point_strides(&self) -> super::QuantParamsStrides {
        self.params.zero_point_strides(self.mode)
    }
}

impl<TB: BufferRef> MatmulB<TB> {
    pub fn is_trellis(&self) -> bool {
        matches!(self, Self::Trellis { .. })
    }

    pub fn quantized(&self) -> Option<&QuantizedB<TB>> {
        match self {
            Self::FullPrecision {
                ..
            }
            | Self::Trellis {
                ..
            } => None,
            Self::Quantized(quantized) => Some(quantized),
        }
    }

    pub fn b_prologue(&self) -> GemmBPrologueKind {
        self.quantized().map_or(GemmBPrologueKind::FullPrecision, QuantizedB::prologue)
    }

    pub fn bits_per_b(&self) -> Option<u32> {
        self.quantized().map(QuantizedB::bits)
    }

    pub fn group_size(&self) -> Option<u32> {
        self.quantized().map(|quantized| quantized.group_size)
    }

    pub fn signed_codes(&self) -> bool {
        self.quantized().is_some_and(|quantized| quantized.signed_codes)
    }

    pub fn weights_layout(&self) -> WeightsLayout<TB::Backend> {
        self.quantized().map_or_else(Default::default, |quantized| quantized.weights_layout)
    }

    pub fn quant_params_layout(&self) -> Option<QuantParamsLayout> {
        self.quantized().map(|quantized| quantized.params.layout())
    }
}
