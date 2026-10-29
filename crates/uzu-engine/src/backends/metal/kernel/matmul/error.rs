use thiserror::Error;

use crate::{
    backends::{
        common::{gpu_types::gemm::GemmDTransform, kernel::matmul::MatmulOutputError},
        metal::{
            error::MetalError,
            kernel::matmul::gemm::{GemmPlanError, GemmSpecializationError},
        },
    },
    data_type::DataType,
};

#[derive(Debug, Error)]
pub enum MatmulMetalError {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported D-transform op {bit:?} on path {path}")]
    UnsupportedDOp {
        bit: GemmDTransform,
        path: &'static str,
    },
    #[error("Unsupported layout on path {path}")]
    UnsupportedLayout {
        path: &'static str,
    },
    #[error("Incompatible A operand for {path}: {reason}")]
    IncompatibleA {
        path: &'static str,
        reason: &'static str,
    },
    #[error("gathered readout requires the GEMV path, but shape (m={m}, n={n}) routes to GEMM")]
    UnsupportedGather {
        m: u32,
        n: u32,
    },
    #[error(transparent)]
    Output(#[from] MatmulOutputError),
    #[error(transparent)]
    Plan(#[from] GemmPlanError),
    #[error(transparent)]
    Specialization(#[from] GemmSpecializationError),
    #[error("Backend error: {0}")]
    BackendError(#[from] MetalError),
}
