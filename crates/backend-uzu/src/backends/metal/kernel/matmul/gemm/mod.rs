#![allow(dead_code)]

pub(crate) mod fp;
mod kernel;
pub(crate) mod quant;

#[allow(unused_imports)]
pub(crate) use kernel::GemmKernel;

#[allow(unused_imports)]
pub(crate) use crate::backends::common::{
    gpu_types::gemm::{
        GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmTilingConfig,
        GemmWeightPrologueKind,
    },
    kernel::gemm::{GemmDispatch, GemmSpecialization, GemmSpecializationError},
};
