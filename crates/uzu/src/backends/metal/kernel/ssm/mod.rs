use crate::backends::metal::{KernelDataType, MTLError};

#[derive(Debug, thiserror::Error)]
pub enum SSMKernelError {
    #[error("Metal error: {0}")]
    MetalError(#[from] MTLError),
    #[error("Function not found: {0}")]
    FunctionNotFound(String),
}

fn fn_suffix(dt: KernelDataType) -> &'static str {
    dt.function_name_suffix()
}

pub mod conv1d_scan;
pub mod ssd_update;

pub use conv1d_scan::{
    Conv1dPackArguments, Conv1dScanArguments, Conv1dScanKernel,
};
pub use ssd_update::{SSDUpdateArguments, SSDUpdateKernel};
