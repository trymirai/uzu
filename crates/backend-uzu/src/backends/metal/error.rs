use std::error::Error as StdError;

use thiserror::Error;

use crate::backends::{
    common::kernel::matmul::MatmulError,
    metal::{Metal, kernel::matmul::gemm::GemmSpecializationError},
};

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Cannot open device")]
    CannotOpenDevice,
    #[error("Cannot start gpu capture {0}")]
    CannotStartGpuCapture(String),
    #[error("Cannot create library: {0}")]
    CannotCreateLibrary(String),
    #[error("Cannot create command queue")]
    CannotCreateCommandQueue,
    #[error("Cannot create command Metal 4 queue")]
    CannotCreateCommandQueueMtl4,
    #[error("Cannot read model weights metadata: {0}")]
    CannotReadModelWeightsMetadata(#[from] std::io::Error),
    #[error("Cannot create buffer")]
    CannotCreateBuffer,
    #[error("Cannot create command buffer")]
    CannotCreateCommandBuffer,
    #[error("Command buffer execution failed: {0}")]
    CommandBufferExecutionFailed(String),
    #[error("Cannot create event")]
    CannotCreateEvent,
    #[error("Cannot create function: {0}")]
    CannotCreateFunction(String),
    #[error("Cannot create pipeline state: {0}")]
    CannotCreatePipelineState(String),
    #[error("Kernel {kernel} was not compiled for request {request}")]
    UnsupportedKernelVariant {
        kernel: &'static str,
        request: String,
    },
    #[error("Can not allocate buffer with size={0}")]
    SparseBufferAlloc(usize),
    #[error("Can not allocate heap with size={0} and page size={1}")]
    SparseHeapAlloc(usize, usize),
    #[error("Kernel dispatch failed: {0}")]
    KernelDispatchFailed(#[source] Box<dyn StdError + Send + Sync + 'static>),
}

impl From<MatmulError<Metal>> for MetalError {
    fn from(value: MatmulError<Metal>) -> Self {
        match value {
            MatmulError::BackendError(e) => e,
            other => MetalError::KernelDispatchFailed(Box::new(other)),
        }
    }
}

impl From<GemmSpecializationError> for MetalError {
    fn from(value: GemmSpecializationError) -> Self {
        MetalError::KernelDispatchFailed(Box::new(value))
    }
}
