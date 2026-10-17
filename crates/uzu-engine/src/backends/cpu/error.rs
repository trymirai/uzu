use thiserror::Error;

#[derive(Debug, Error)]
pub enum CpuError {
    #[error("Not supported")]
    NotSupported,
    #[error("Command buffer execution failed")]
    CommandBufferExecutionFailed,
}
