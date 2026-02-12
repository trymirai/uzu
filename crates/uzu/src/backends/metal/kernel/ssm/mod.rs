use crate::backends::metal::MTLError;

#[derive(Debug, thiserror::Error)]
pub enum SSMKernelError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

pub mod conv1d_scan;
pub mod ssd_prefill;

pub use conv1d_scan::{Conv1dPackArguments, Conv1dScanArguments, Conv1dScanKernel};
pub use ssd_prefill::{SSDPrefillArguments, SSDPrefillKernels, SSDPrefillMode};
