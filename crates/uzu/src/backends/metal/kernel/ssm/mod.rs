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

pub mod conv1d_forward;
pub mod conv1d_swish_forward;
pub mod conv1d_update;
pub mod segsum;
pub mod encodable;
pub mod ssd_update;
pub mod ssd_update_no_z;
pub mod ssm_update;

pub use conv1d_forward::{Conv1dForwardArguments, Conv1dForwardKernel};
pub use conv1d_swish_forward::{
    Conv1dSwishForwardArguments, Conv1dSwishForwardKernel,
};
pub use conv1d_update::{Conv1dUpdateArguments, Conv1dUpdateKernel};
pub use segsum::{
    Cumsum1DArguments, Cumsum1DKernel, SegsumFromCumsumArguments,
    SegsumFromCumsumKernel,
};
pub use ssd_update::{SSDUpdateArguments, SSDUpdateKernel};
pub use ssd_update_no_z::{SSDUpdateNoZArguments, SSDUpdateNoZKernel};
pub use ssm_update::{SSMUpdateArguments, SSMUpdateKernel};
pub use encodable::SSMLayerKernelEncodable;
