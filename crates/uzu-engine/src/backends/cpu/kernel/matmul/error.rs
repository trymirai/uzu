use thiserror::Error;

use crate::{
    backends::{common::kernel::matmul::MatmulOutputError, cpu::error::CpuError},
    data_type::DataType,
};

#[derive(Debug, Error)]
pub enum MatmulCpuError {
    #[error("Unsupported data type: {0:?}")]
    UnsupportedDataType(DataType),
    #[error("Unsupported layout on path {path}")]
    UnsupportedLayout {
        path: &'static str,
    },
    #[error("Incompatible A operand for {path}: {reason}")]
    IncompatibleA {
        path: &'static str,
        reason: &'static str,
    },
    #[error(transparent)]
    Output(#[from] MatmulOutputError),
    #[error("Backend error: {0}")]
    BackendError(#[from] CpuError),
}
