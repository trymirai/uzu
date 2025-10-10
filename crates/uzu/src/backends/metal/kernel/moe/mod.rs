use crate::backends::metal::KernelDataType;

mod encodable;
pub use encodable::{MoeBlockEncodable, SharedMoeWeights};

// Submodules
mod counts_offsets_fused;
mod experts;
mod finalize;
mod gather;
mod router_topk;
mod scatter;
mod tiles;

// Re-export public items from submodules
pub use counts_offsets_fused::{
    MoeCountsOffsetsFusedArguments, MoeCountsOffsetsFusedError,
    MoeCountsOffsetsFusedKernel,
};
pub use experts::{
    MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel,
    MoeScatterError,
};
pub use finalize::{MoeFinalizeArguments, MoeFinalizeError, MoeFinalizeKernel};
pub use gather::{MoeGatherArguments, MoeGatherError, MoeGatherKernel};
pub use router_topk::{
    MoeRouterTopKArguments, MoeRouterTopKError, MoeRouterTopKKernel,
};
pub use scatter::{
    MoeBlockBasesArguments, MoeScatterArguments, MoeScatterKernels,
    MoeScatterWithMapArguments,
};
pub use tiles::{
    MoePassARowMapArguments, MoePassATileBuildArguments,
    MoePassATileCountsArguments, MoePassATileDispatchArguments,
    MoePassATileKernel, MoePassATileScanArguments, MoeTileCountsArguments,
    MoeTileDispatchArguments, MoeTileError, MoeTileMapBuildArguments,
    MoeTileMapKernel, MoeTileScanArguments,
};

// Common utility functions
pub(crate) fn dtype_suffix(dtype: KernelDataType) -> &'static str {
    match dtype {
        KernelDataType::Float16 => "f16",
        KernelDataType::BFloat16 => "bf16",
        KernelDataType::Float32 => "f32",
    }
}

pub(crate) fn dtype_index(dtype: KernelDataType) -> usize {
    match dtype {
        KernelDataType::Float16 => 0,
        KernelDataType::BFloat16 => 1,
        KernelDataType::Float32 => 2,
    }
}
