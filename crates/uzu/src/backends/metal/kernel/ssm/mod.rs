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
pub mod conv1d_scan;
pub mod conv1d_swish_forward;
pub mod conv1d_update;
pub mod dt_decay;
pub mod mamba;
pub mod segsum;
pub mod split_conv_outputs;
pub mod split_inproj;
pub mod ssd_prefill;
pub mod ssd_update;
pub mod ssd_update_no_z;
pub mod ssm_update;

pub use conv1d_forward::{Conv1dForwardArguments, Conv1dForwardKernel};
pub use conv1d_scan::{Conv1dScanArguments, Conv1dScanKernel};
pub use conv1d_swish_forward::{
    Conv1dSwishForwardArguments, Conv1dSwishForwardKernel,
};
pub use conv1d_update::{Conv1dUpdateArguments, Conv1dUpdateKernel};
pub use dt_decay::{DtDecayArguments, DtDecayKernel};
pub(crate) use mamba::MambaMixerEncodable;
pub use segsum::{
    Cumsum1DArguments, Cumsum1DKernel, SegsumFromCumsumArguments,
    SegsumFromCumsumKernel,
};
pub use split_conv_outputs::{
    SplitConvOutputsArguments, SplitConvOutputsKernel,
};
pub use split_inproj::{SplitInProjArguments, SplitInProjKernel};
pub use ssd_prefill::{SSDPrefillArguments, SSDPrefillKernel};
pub use ssd_update::{SSDUpdateArguments, SSDUpdateKernel};
pub use ssd_update_no_z::{SSDUpdateNoZArguments, SSDUpdateNoZKernel};
pub use ssm_update::{SSMUpdateArguments, SSMUpdateKernel};
