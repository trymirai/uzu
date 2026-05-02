//! Isolated unified GEMM implementation.
#![allow(dead_code)]

mod kernel;
mod quantized_storage;
mod specialization;
mod specialization_error;
mod tile;

#[allow(unused_imports)]
pub(crate) use kernel::UnifiedGemmKernel;
pub(crate) use quantized_storage::WeightsStorageFormat;
pub(crate) use specialization::UnifiedGemmSpecialization;
pub(crate) use specialization_error::UnifiedGemmSpecializationError;
pub(crate) use tile::GemmTile;

#[allow(unused_imports)]
pub(crate) use crate::backends::common::gpu_types::unified_gemm::{
    GemmAlignment, GemmComputeKind, GemmInputPrologueKind, GemmOutputTransformKind, GemmWeightPrologueKind,
    QuantizedMetadataKind,
};
