use crate::backends::metal::kernel::unified_matmul::gemm::QuantizationParams;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum QuantizedFormat {
    MLX(QuantizationParams),
    AWQ(QuantizationParams),
}

impl QuantizedFormat {
    pub(crate) const fn params(self) -> QuantizationParams {
        match self {
            Self::MLX(params) | Self::AWQ(params) => params,
        }
    }
}
