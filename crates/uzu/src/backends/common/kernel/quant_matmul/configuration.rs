use crate::{DataType, backends::common::Backend, config::QuantizationMode};

use super::{
    QuantizedMatmulError, QuantizedMatmulType,
    variant::{MatrixMatrixFamily, MatrixVectorFamily},
};

#[derive(Debug, Clone, Copy)]
pub struct QuantizedMatmulConfiguration {
    pub data_type: DataType,
    pub group_size: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub mode: QuantizationMode,
    pub quantization_type: QuantizedMatmulType,
    pub weights_transposed: bool,
}

impl QuantizedMatmulConfiguration {
    pub fn validate<B: Backend>(&self) -> Result<(), QuantizedMatmulError<B>> {
        if !matches!(self.data_type, DataType::F16 | DataType::BF16 | DataType::F32) {
            return Err(QuantizedMatmulError::UnsupportedDataType(self.data_type));
        }

        if !matches!(self.group_size, 32 | 64 | 128) {
            return Err(QuantizedMatmulError::UnsupportedGroupSize(self.group_size));
        }

        Ok(())
    }

    pub(super) fn matrix_vector_family(&self) -> MatrixVectorFamily {
        if self.weights_transposed {
            if self.output_dim % 8 == 0 && self.input_dim % 512 == 0 {
                MatrixVectorFamily::GemvFast
            } else {
                MatrixVectorFamily::Gemv
            }
        } else {
            MatrixVectorFamily::VectorMatrix
        }
    }

    pub(super) fn matrix_matrix_family(
        &self,
        bits: usize,
    ) -> MatrixMatrixFamily {
        if self.weights_transposed {
            let aligned_n = self.output_dim % 32 == 0;
            let use_64x64 = aligned_n
                && self.data_type == DataType::BF16
                && matches!(self.group_size, 64 | 128)
                && matches!(bits, 4 | 8);
            if use_64x64 {
                MatrixMatrixFamily::GemmTransposed64x64
            } else if aligned_n {
                MatrixMatrixFamily::GemmTransposedAlignedN
            } else {
                MatrixMatrixFamily::GemmTransposedUnalignedN
            }
        } else if self.input_dim % 32 == 0 {
            MatrixMatrixFamily::GemmAlignedK
        } else {
            MatrixMatrixFamily::GemmUnalignedK
        }
    }
}
