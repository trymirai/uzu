use thiserror::Error;

#[derive(Debug, Error)]
pub enum CpuError {
    #[error("not supported")]
    NotSupported,
}
