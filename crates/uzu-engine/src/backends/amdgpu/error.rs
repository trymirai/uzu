use std::error::Error as StdError;

use thiserror::Error;

use crate::backends::{amdgpu::Amdgpu, common::kernel::matmul::MatmulError};

#[derive(Debug, Error)]
pub enum AmdgpuError {
    #[error("Not supported")]
    NotSupported,
    #[error("HIP runtime is not available: {0}")]
    RuntimeUnavailable(String),
    #[error("No AMD GPU found")]
    NoDevice,
    #[error("{call} failed with {code}: {message}")]
    Hip {
        call: &'static str,
        code: i32,
        message: String,
    },
    #[error("Kernel {0} is not available on AMDGPU")]
    KernelUnavailable(String),
    #[error("Kernel dispatch failed: {0}")]
    KernelDispatchFailed(#[source] Box<dyn StdError + Send + Sync + 'static>),
}

impl From<MatmulError<Amdgpu>> for AmdgpuError {
    fn from(value: MatmulError<Amdgpu>) -> Self {
        match value {
            MatmulError::BackendError(e) => e,
            other => AmdgpuError::KernelDispatchFailed(Box::new(other)),
        }
    }
}
