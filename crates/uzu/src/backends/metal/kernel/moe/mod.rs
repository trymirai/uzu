use crate::backends::metal::KernelDataType;

// Submodules
mod experts;
mod gather;
mod router_topk;
mod scatter;
mod tiles;
mod experts_single;
pub mod tiles_map;

// Re-export public items from submodules
pub use experts::{
    MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel,
    MoeScatterError,
};
pub use experts_single::{MoeExpertsSingleDecodeArguments, MoeExpertsSingleDecodeKernels};
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
    MoePassATileKernel, MoePassATileScanArguments,
    MoeTileError,
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
