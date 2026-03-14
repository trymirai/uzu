use std::fmt;

use crate::{
    DataType,
    backends::common::{
        Backend, Kernels,
        kernel::{
            QuantizedMatmulGemmKernel, QuantizedMatmulGemmTransposed64x64Kernel, QuantizedMatmulGemmTransposedKernel,
            QuantizedMatmulGemvFastKernel, QuantizedMatmulGemvKernel, QuantizedMatmulVectorMatrixKernel,
        },
    },
};

use super::QuantizedMatmulError;

pub(super) enum EncodableVariant<B: Backend> {
    Gemv(<B::Kernels as Kernels>::QuantizedMatmulGemvKernel),
    GemvFast(<B::Kernels as Kernels>::QuantizedMatmulGemvFastKernel),
    VectorMatrix(<B::Kernels as Kernels>::QuantizedMatmulVectorMatrixKernel),
    Gemm(<B::Kernels as Kernels>::QuantizedMatmulGemmKernel),
    GemmTransposed(<B::Kernels as Kernels>::QuantizedMatmulGemmTransposedKernel),
    GemmTransposed64x64(<B::Kernels as Kernels>::QuantizedMatmulGemmTransposed64x64Kernel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum KernelKey {
    MatrixVector(MatrixVectorFamily),
    MatrixMatrix(MatrixMatrixFamily),
}

impl fmt::Display for KernelKey {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::MatrixVector(MatrixVectorFamily::Gemv) => write!(f, "matrix_vector_gemv"),
            Self::MatrixVector(MatrixVectorFamily::GemvFast) => write!(f, "matrix_vector_gemv_fast"),
            Self::MatrixVector(MatrixVectorFamily::VectorMatrix) => write!(f, "matrix_vector_vector_matrix"),
            Self::MatrixMatrix(MatrixMatrixFamily::GemmAlignedK) => write!(f, "matrix_matrix_gemm_aligned_k"),
            Self::MatrixMatrix(MatrixMatrixFamily::GemmUnalignedK) => write!(f, "matrix_matrix_gemm_unaligned_k"),
            Self::MatrixMatrix(MatrixMatrixFamily::GemmTransposedAlignedN) => {
                write!(f, "matrix_matrix_gemm_transposed_aligned_n")
            },
            Self::MatrixMatrix(MatrixMatrixFamily::GemmTransposedUnalignedN) => {
                write!(f, "matrix_matrix_gemm_transposed_unaligned_n")
            },
            Self::MatrixMatrix(MatrixMatrixFamily::GemmTransposed64x64) => {
                write!(f, "matrix_matrix_gemm_transposed_64x64")
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MatrixVectorFamily {
    Gemv,
    GemvFast,
    VectorMatrix,
}

impl MatrixVectorFamily {
    pub(super) fn create_kernel<B: Backend>(
        self,
        context: &B::Context,
        data_type: DataType,
        group_size: i32,
        bits: i32,
        use_mlx_quant: bool,
    ) -> Result<EncodableVariant<B>, QuantizedMatmulError<B>> {
        let use_zero_points = !use_mlx_quant;
        match self {
            Self::Gemv => Ok(EncodableVariant::Gemv(
                <B::Kernels as Kernels>::QuantizedMatmulGemvKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
            Self::GemvFast => Ok(EncodableVariant::GemvFast(
                <B::Kernels as Kernels>::QuantizedMatmulGemvFastKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
            Self::VectorMatrix => Ok(EncodableVariant::VectorMatrix(
                <B::Kernels as Kernels>::QuantizedMatmulVectorMatrixKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum MatrixMatrixFamily {
    GemmAlignedK,
    GemmUnalignedK,
    GemmTransposedAlignedN,
    GemmTransposedUnalignedN,
    GemmTransposed64x64,
}

impl MatrixMatrixFamily {
    pub(super) fn create_kernel<B: Backend>(
        self,
        context: &B::Context,
        data_type: DataType,
        group_size: i32,
        bits: i32,
        use_mlx_quant: bool,
    ) -> Result<EncodableVariant<B>, QuantizedMatmulError<B>> {
        let use_zero_points = !use_mlx_quant;
        match self {
            Self::GemmAlignedK => Ok(EncodableVariant::Gemm(
                <B::Kernels as Kernels>::QuantizedMatmulGemmKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                    true,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
            Self::GemmUnalignedK => Ok(EncodableVariant::Gemm(
                <B::Kernels as Kernels>::QuantizedMatmulGemmKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                    false,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
            Self::GemmTransposedAlignedN => Ok(EncodableVariant::GemmTransposed(
                <B::Kernels as Kernels>::QuantizedMatmulGemmTransposedKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                    true,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
            Self::GemmTransposedUnalignedN => Ok(EncodableVariant::GemmTransposed(
                <B::Kernels as Kernels>::QuantizedMatmulGemmTransposedKernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                    false,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
            Self::GemmTransposed64x64 => Ok(EncodableVariant::GemmTransposed64x64(
                <B::Kernels as Kernels>::QuantizedMatmulGemmTransposed64x64Kernel::new(
                    context,
                    data_type,
                    group_size,
                    bits,
                    use_zero_points,
                    use_mlx_quant,
                )
                .map_err(QuantizedMatmulError::BackendError)?,
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RuntimeVariant {
    MatrixVector,
    MatrixMatrix,
}
