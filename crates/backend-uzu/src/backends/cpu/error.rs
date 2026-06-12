use std::{error::Error as StdError, sync::mpsc::RecvError};

use thiserror::Error;

use crate::backends::{common::kernel::matmul::MatmulError, cpu::Cpu};

#[derive(Debug, Error)]
pub enum CpuError {
    #[error("Not supported")]
    NotSupported,
    #[error("Command buffer execution failed")]
    CommandBufferExecutionFailed(#[from] RecvError),
    #[error("Kernel dispatch failed: {0}")]
    KernelDispatchFailed(#[source] Box<dyn StdError + Send + Sync + 'static>),
}

impl From<MatmulError<Cpu>> for CpuError {
    fn from(value: MatmulError<Cpu>) -> Self {
        match value {
            MatmulError::BackendError(e) => e,
            other => CpuError::KernelDispatchFailed(Box::new(other)),
        }
    }
}
