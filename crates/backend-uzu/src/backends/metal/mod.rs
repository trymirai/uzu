mod backend;
mod buffer;
mod command_buffer;
mod context;
mod dense_buffer;
mod device_profile;
mod error;
mod kernel;
mod metal_extensions;
mod sparse;

use crate::backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE;

const METAL_SIMD_SIZE: u32 = 32;

const _: () = {
    assert!(HADAMARD_TRANSFORM_BLOCK_SIZE == METAL_SIMD_SIZE);
};

pub use backend::Metal;
pub use context::MetalContext;
#[cfg(test)]
pub use device_profile::{DeviceGeneration, DeviceProfile, DeviceSize};
#[cfg(test)]
pub use kernel::matmul::gemm::GemmEngine;
#[cfg(test)]
pub(crate) use kernel::matmul::gemv::{GemvDispatch, GemvSpecialization};
#[cfg(test)]
pub(crate) use kernel::{
    QtipActivationTransformRowCached5120x512MetalKernel, QtipActivationTransformRowCached6144x512MetalKernel,
    QtipActivationTransformRowCached17408x512MetalKernel, QtipActivationTransformRowCachedPadded5120x512MetalKernel,
    QtipActivationTransformRowCachedPadded6144x512MetalKernel,
    QtipActivationTransformRowCachedPadded17408x512MetalKernel, QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit1B64MetalKernel,
    QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit2B64MetalKernel,
    QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit4B64MetalKernel,
    QtipGaussianComputedWalshOneMulA8MxuPairedK2RowSplit8B64MetalKernel,
    QtipGaussianPhysicalQ8V4A8DirectB32BatchRowsMetalKernel,
    QtipGaussianPhysicalQ8V4A8DirectB32MetalKernel, };

// race kernels (qtip_race.metal) + physical kernels needed by the race profile
#[cfg(test)]
pub(crate) use kernel::{
    QtipGaussianPhysicalQ8V2A8DirectK2B16MetalKernel,
    QtipGaussianPhysicalQ8V2A8DirectK2B32BatchRowsMetalKernel,
    QtipGaussianPhysicalQ8V2A8DirectK2B64BatchRowsMetalKernel,
    QtipGaussianPhysicalQ8V2A8DirectK3B16MetalKernel,
    QtipGaussianPhysicalQ8V2A8DirectK3B32BatchRowsMetalKernel,
    QtipGaussianPhysicalQ8V2A8DirectK3B64BatchRowsMetalKernel,
    QtipGaussianPhysicalQ8V4A8DirectB16MetalKernel,
    QtipGaussianPhysicalQ8V4A8DirectB64BatchRowsMetalKernel,
    QtipRowsBatchToBatchRowsBf16MetalKernel,
    QtipRaceV4Pf2Sg4B32MetalKernel,
    QtipRaceV4Pf2Sg2B32MetalKernel,
    QtipRaceV4Pf2Sg4B64MetalKernel,
    QtipRaceV4Pf2Sg2B64MetalKernel,
    QtipRaceV4Pf2Sg8B64MetalKernel,
    QtipRaceV4L15Pf2Sg4B32MetalKernel,
    QtipRaceV4L15Pf2Sg2B32MetalKernel,
    QtipRaceV4L15Pf2Sg2B64MetalKernel,
    QtipRaceV4L15Pf2Sg4B64MetalKernel,
    QtipRaceV4L15Pf2Sg8B64MetalKernel,
    QtipRaceV4L15R2Pf2Sg2B32MetalKernel,
    QtipRaceV4L15T2Pf0Sg2B16MetalKernel,
    QtipRaceV4L15T2Pf2Sg2B16MetalKernel,
    QtipRaceV4L15T4Pf1Sg2B16MetalKernel,
    QtipRaceV4L15Sw22B32MetalKernel,
    QtipRaceV4L15Sw42B32MetalKernel,
    QtipRaceV4L15Sw22sB32MetalKernel,
    QtipRaceV4L15Sw11B32MetalKernel,
    QtipRaceV4L15Sw21B32MetalKernel,
    QtipRaceV4L15Sw22B64MetalKernel,
    QtipRaceV4L15Sw42B64MetalKernel,
    QtipRaceV4L15Sw22sB64MetalKernel,
    QtipRaceV4L15Sw11B64MetalKernel,
    QtipRaceV4L15Sw21B64MetalKernel,
    QtipRaceK3Sw22B32MetalKernel,
    QtipRaceK3Sw42B32MetalKernel,
    QtipRaceK3Sw21B32MetalKernel,
    QtipRaceK3Sw22B64MetalKernel,
    QtipRaceK3Sw42B64MetalKernel,
    QtipRaceK3Sw21B64MetalKernel,
    QtipRaceK2Sw22B32MetalKernel,
    QtipRaceK2Sw42B32MetalKernel,
    QtipRaceK2Sw22B64MetalKernel,
    QtipRaceK2Sw42B64MetalKernel,
    QtipRaceV4L15AsPf1Sg8B32MetalKernel,
    QtipRaceV4L15AsPf1Sg16B32MetalKernel,
    QtipRaceV4L15AsR2Pf1Sg8B32MetalKernel,
    QtipRaceV4L15AsPf2Sg8B32MetalKernel,
    QtipRaceV4L15AsPf1Sg8B64MetalKernel,
    QtipRaceV4L15AsPf1Sg16B64MetalKernel,
    QtipRaceV4L15AsR2Pf1Sg8B64MetalKernel,
    QtipRaceV4L15AsPf2Sg8B64MetalKernel,
    QtipRaceV4AntiPf2Sg2B32MetalKernel,
    QtipRaceV4AntiPf2Sg4B32MetalKernel,
    QtipRaceV4AntiPf2Sg2B64MetalKernel,
    QtipRaceV4AntiPf2Sg4B64MetalKernel,
    QtipRaceV4Sign12Pf2Sg2B32MetalKernel,
    QtipRaceV4Sign12Pf2Sg4B32MetalKernel,
    QtipRaceV4Sign12Pf2Sg2B64MetalKernel,
    QtipRaceV4Sign12Pf2Sg4B64MetalKernel,
    QtipRaceV4AntiT2Pf2Sg2B16MetalKernel,
    QtipRaceV4AntiAsPf1Sg16B64MetalKernel,
    QtipRaceV4Sign14Pf2Sg2B32MetalKernel,
    QtipRaceV4Sign14Pf2Sg4B32MetalKernel,
    QtipRaceV4Sign14Pf2Sg2B64MetalKernel,
    QtipRaceV4Sign14Pf2Sg4B64MetalKernel,
    QtipRaceV4Sign14T2Pf2Sg2B16MetalKernel,
    QtipRaceV4Sign14AsPf1Sg16B64MetalKernel,
    QtipRaceV4Sign12T2Pf2Sg2B16MetalKernel,
    QtipRaceK3L15Pf2Sg2B32MetalKernel,
    QtipRaceK3L15Pf0Sg4B32MetalKernel,
    QtipRaceK3L15Pf2Sg4B64MetalKernel,
    QtipRaceK3L15Pf2Sg2B64MetalKernel,
    QtipRaceK2L15Pf2Sg2B32MetalKernel,
    QtipRaceK2L15Pf0Sg4B32MetalKernel,
    QtipRaceK2L15Pf2Sg2B64MetalKernel,
    QtipRaceK3L15R2Pf2Sg2B32MetalKernel,
    QtipRaceK2L15R2Pf2Sg2B32MetalKernel,
    QtipRaceK3L15T2Pf0Sg2B16MetalKernel,
    QtipRaceK3L15T4Pf0Sg2B16MetalKernel,
    QtipRaceK2L15T2Pf0Sg2B16MetalKernel,
    QtipRaceK2L15T2Pf2Sg2B16MetalKernel,
    QtipRaceK2Pf0Sg4B32MetalKernel,
    QtipRaceK2Pf2Sg4B32MetalKernel,
    QtipRaceK2Pf2Sg2B32MetalKernel,
    QtipRaceK2Pf2Sg2B64MetalKernel,
    QtipRaceK3Pf0Sg4B32MetalKernel,
    QtipRaceK3Pf2Sg4B32MetalKernel,
    QtipRaceK3Pf2Sg2B32MetalKernel,
    QtipRaceK3Pf2Sg4B64MetalKernel,
    QtipRaceK3Pf2Sg2B64MetalKernel,
    };

// race kernels (split-table) + aux kernels needed by the race profiles
#[cfg(test)]
pub(crate) use kernel::{
    QtipFullIncoherenceA8MetalKernel,
    QtipI3S4ReadoutMxuB16MetalKernel,
    QtipI3S4ReadoutMxuB32MetalKernel,
    QtipI3S4ReadoutMxuB64MetalKernel,
    QtipRht32Bf16PaddedMetalKernel,
    QtipD4S4EmbeddingLookupMetalKernel,
    QtipRowsBatchToBatchRowsF32MetalKernel,
};

// race transform kernels (qtip_race_transform.metal)
#[cfg(test)]
pub(crate) use kernel::{
    QtipRaceTransform5120MetalKernel,
    QtipRaceTransform6144MetalKernel,
    QtipRaceTransform17408MetalKernel,
};

// race two-row-fragment kernels
#[cfg(test)]
pub(crate) use kernel::{
    QtipRaceV4R2Pf2Sg2B32MetalKernel,
    QtipRaceV4R2Pf2Sg4B32MetalKernel,
    QtipRaceK2R2Pf2Sg2B32MetalKernel,
    QtipRaceK2R2Pf0Sg2B32MetalKernel,
    QtipRaceK3R2Pf2Sg2B32MetalKernel,
    QtipRaceK3R2Pf0Sg2B32MetalKernel,
    };

// race four-row-fragment kernels

// race transposed 16-token kernels
#[cfg(test)]
pub(crate) use kernel::{
    QtipRaceV4T2Pf2Sg2B16MetalKernel,
    QtipRaceK3T2Pf0Sg2B16MetalKernel,
    QtipRaceK3T4Pf0Sg2B16MetalKernel,
    QtipRaceK2T2Pf0Sg2B16MetalKernel,
    QtipRaceK2T2Pf2Sg2B16MetalKernel,
    };

// race half-table kernels

// race wide-threadgroup probes

// race component-split kernels
#[cfg(test)]
pub(crate) use kernel::{
    QtipRaceV4CsPf2Sg4B32Pass0MetalKernel,
    QtipRaceV4CsPf2Sg4B32Pass1MetalKernel,
    QtipRaceV4CsR2Pf2Sg2B32Pass0MetalKernel,
    QtipRaceV4CsR2Pf2Sg2B32Pass1MetalKernel,
    QtipRaceV4CsPf2Sg2B64Pass0MetalKernel,
    QtipRaceV4CsPf2Sg2B64Pass1MetalKernel,
    QtipRaceV4CsT2Pf0Sg2B16Pass0MetalKernel,
    QtipRaceV4CsT2Pf0Sg2B16Pass1MetalKernel,
    };

// race diagnostics (hot activations)

// race staged-activation kernels
#[cfg(test)]
pub(crate) use kernel::{
    QtipRaceV4AsPf1Sg4B32MetalKernel,
    QtipRaceV4AsPf1Sg8B32MetalKernel,
    QtipRaceV4AsPf1Sg16B32MetalKernel,
    QtipRaceV4AsPf0Sg8B32MetalKernel,
    QtipRaceV4AsR2Pf1Sg4B32MetalKernel,
    QtipRaceV4AsPf1Sg4B64MetalKernel,
    QtipRaceV4AsPf1Sg8B64MetalKernel,
    QtipRaceV4AsPf1Sg16B64MetalKernel,
    QtipRaceV4AsR2Pf1Sg4B64MetalKernel,
    QtipRaceV4AsCsPf1Sg8B32Pass1MetalKernel,
    QtipRaceV4AsCsPf1Sg8B32Pass2MetalKernel,
    QtipRaceV4AsCsPf1Sg16B32Pass1MetalKernel,
    QtipRaceV4AsCsPf1Sg16B32Pass2MetalKernel,
    QtipRaceV4AsCsPf1Sg8B64Pass1MetalKernel,
    QtipRaceV4AsCsPf1Sg8B64Pass2MetalKernel,
    QtipRaceK3AsPf1Sg8B32MetalKernel,
    QtipRaceK3AsPf1Sg16B32MetalKernel,
    QtipRaceK3AsR2Pf1Sg4B32MetalKernel,
    QtipRaceK3AsPf1Sg8B64MetalKernel,
    QtipRaceK3AsPf1Sg16B64MetalKernel,
    QtipRaceK2AsPf1Sg8B32MetalKernel,
    QtipRaceK2AsPf1Sg8B64MetalKernel,
};

// race K-split component split + permute
#[cfg(test)]
pub(crate) use kernel::{
    QtipRacePermuteHalvesMetalKernel,
};
