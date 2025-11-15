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
pub mod dt_decay;
pub mod mamba;
pub mod segsum;
pub mod split_conv_outputs;
pub mod split_inproj;
pub mod ssd_prefill;
pub mod ssd_update;
pub mod ssm_update;

pub use conv1d_scan::{Conv1dScanArguments, Conv1dScanKernel};
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
pub use ssm_update::{SSMUpdateArguments, SSMUpdateKernel};
