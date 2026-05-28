use crate::backends::common::gpu_types::{gemm::GemmDTransform, matmul::GemvTiling};

/// Host-side GEMV configuration: the tile layout plus the output transforms
/// the pipeline is specialized for. SCALE is intentionally omitted —
/// `precompile_configs` only covers the common `ab_scale == 1.0` case; scaled
/// pipelines are created on demand at encode time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GemvSpecialization {
    pub tiling: GemvTiling,
    pub is_accumulate: bool,
    pub is_bias: bool,
    pub is_hadamard: bool,
}

impl GemvSpecialization {
    pub fn precompile_configs(data_type: crate::DataType) -> &'static [Self] {
        use crate::DataType;
        match data_type {
            DataType::BF16 => &[
                Self {
                    tiling: GemvTiling::Standard,
                    is_accumulate: false,
                    is_bias: false,
                    is_hadamard: false,
                },
                Self {
                    tiling: GemvTiling::Standard,
                    is_accumulate: false,
                    is_bias: true,
                    is_hadamard: false,
                },
                Self {
                    tiling: GemvTiling::Wide,
                    is_accumulate: false,
                    is_bias: false,
                    is_hadamard: false,
                },
                Self {
                    tiling: GemvTiling::Wide,
                    is_accumulate: false,
                    is_bias: true,
                    is_hadamard: false,
                },
                Self {
                    tiling: GemvTiling::Wide,
                    is_accumulate: false,
                    is_bias: false,
                    is_hadamard: true,
                },
                Self {
                    tiling: GemvTiling::Wide,
                    is_accumulate: false,
                    is_bias: true,
                    is_hadamard: true,
                },
            ],
            DataType::F16 => &[
                Self {
                    tiling: GemvTiling::Wide,
                    is_accumulate: false,
                    is_bias: false,
                    is_hadamard: false,
                },
                Self {
                    tiling: GemvTiling::Wide,
                    is_accumulate: false,
                    is_bias: false,
                    is_hadamard: true,
                },
            ],
            _ => &[],
        }
    }

    pub fn output_transform(&self) -> GemmDTransform {
        let mut transform = GemmDTransform::empty();
        transform.set(GemmDTransform::ACCUMULATE, self.is_accumulate);
        transform.set(GemmDTransform::BIAS, self.is_bias);
        transform.set(GemmDTransform::RHT, self.is_hadamard);
        transform
    }

    pub fn select(
        input_dimension: u32,
        output_dimension: u32,
        is_accumulate: bool,
        is_bias: bool,
        is_hadamard: bool,
    ) -> Self {
        let narrow = output_dimension < 4;
        let tiling = if is_hadamard {
            if narrow {
                GemvTiling::WideNarrow
            } else {
                GemvTiling::Wide
            }
        } else if input_dimension <= 64 {
            if narrow {
                GemvTiling::SmallInputNarrow
            } else {
                GemvTiling::SmallInput
            }
        } else if input_dimension >= 16 * output_dimension {
            if narrow {
                GemvTiling::SplitInputNarrow
            } else {
                GemvTiling::SplitInput
            }
        } else if output_dimension >= 4096 {
            if narrow {
                GemvTiling::WideNarrow
            } else {
                GemvTiling::Wide
            }
        } else if narrow {
            GemvTiling::StandardNarrow
        } else {
            GemvTiling::Standard
        };

        Self {
            tiling,
            is_accumulate,
            is_bias,
            is_hadamard,
        }
    }
}
