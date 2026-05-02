//! Isolated unified GEMM implementation.
#![allow(dead_code)]

mod gemm_tile;
mod quantized_storage;
mod unified_gemm_kernel;
mod unified_gemm_specialization;
mod unified_gemm_specialization_error;

pub(crate) use gemm_tile::GemmTile;
pub(crate) use quantized_storage::{
    GroupSize, QuantizationParams, QuantizedFormat, WeightsStorageFormat,
};
#[allow(unused_imports)]
pub(crate) use unified_gemm_kernel::UnifiedGemmKernel;
pub(crate) use unified_gemm_specialization::UnifiedGemmSpecialization;
pub(crate) use unified_gemm_specialization_error::UnifiedGemmSpecializationError;

#[allow(unused_imports)]
pub(crate) use crate::backends::common::gpu_types::unified_gemm::{
    BitsPerWeight, GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind,
    GemmWeightPrologueKind,
};
