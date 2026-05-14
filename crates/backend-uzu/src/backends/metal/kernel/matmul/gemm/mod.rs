pub(crate) mod fp;
mod kernel;
pub(crate) mod quant;

pub(crate) use kernel::GemmKernel;

pub(crate) use crate::backends::common::{
    gpu_types::gemm::{GemmComputeKind, GemmInputPrologueKind},
    kernel::gemm::GemmDispatch,
};
