use std::{error::Error as StdError, sync::mpsc::RecvError};

use thiserror::Error;

use crate::backends::{
    common::kernel::matmul::MatmulError,
    metal::{Metal, kernel::matmul::gemm::GemmSpecializationError},
};

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("Cannot open device")]
    CannotOpenDevice,
    #[error("Cannot create residency set: {0}")]
    CannotCreateResidencySet(String),
    #[error("Cannot start gpu capture {0}")]
    CannotStartGpuCapture(String),
    #[error("Cannot create library: {0}")]
    CannotCreateLibrary(String),
    #[error("Cannot create compiler: {0}")]
    CannotCreateCompiler(String),
    #[error("Cannot create command queue")]
    CannotCreateCommandQueue,
    #[error("Cannot create event")]
    CannotCreateEvent,
    #[error("Cannot create buffer")]
    CannotCreateBuffer,
    #[error("Cannot create heap")]
    CannotCreateHeap,
    #[error("Cannot resolve counter heap")]
    CannotResolveCounterHeap,
    #[error("Timestamp {0} was not written by the GPU")]
    UnwrittenTimestamp(usize),
    #[error("Cannot create command buffer")]
    CannotCreateCommandBuffer,
    #[error("Cannot create argument table: {0}")]
    CannotCreateArgumentTable(String),
    #[error("Error waiting for command buffer: {0}")]
    CommandBufferWait(RecvError),
    #[error("Command buffer execution failed: {0}")]
    CommandBufferExecution(String),
    #[error("Cannot create pipeline state for {function_name}: {error}")]
    CannotCreatePipelineState {
        function_name: String,
        error: String,
    },
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
