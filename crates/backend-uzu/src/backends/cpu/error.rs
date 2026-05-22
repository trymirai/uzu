use std::sync::mpsc::RecvError;

use thiserror::Error;

use crate::backends::{common::kernel::matmul::MatmulError, cpu::Cpu};

#[derive(Debug, Error)]
pub enum CpuError {
    #[error("Not supported")]
    NotSupported,
    #[error("Command buffer execution failed")]
    CommandBufferExecutionFailed(RecvError),
    #[error("Matmul error: {0}")]
    Matmul(#[source] Box<MatmulError<Cpu>>),
}

impl From<MatmulError<Cpu>> for CpuError {
    fn from(value: MatmulError<Cpu>) -> Self {
        match value {
            MatmulError::BackendError(e) => e,
            other => CpuError::Matmul(Box::new(other)),
        }
    }
}
