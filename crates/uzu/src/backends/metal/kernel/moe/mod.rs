use crate::backends::metal::KernelDataType;

// Submodules
mod experts;
mod experts_single;
mod experts_two_pass_decode;
mod gather;
mod router_topk;
mod scatter;
pub mod tiles_map;
pub mod tiles_pass_a;

// Re-export public items from submodules
pub use experts::{
    MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments, MoeExpertsTwoPassPrefillKernel, MoeScatterError,
};
pub use experts_single::{MoeExpertsSingleDecodeArguments, MoeExpertsSingleDecodeKernels};
pub use experts_two_pass_decode::MoeExpertsTwoPassDecodeKernels;
pub use gather::{MoeGatherArguments, MoeGatherKernels};
pub use router_topk::{MoeRouterTopKArguments, MoeRouterTopKError, MoeRouterTopKKernel};
pub use scatter::{MoeBlockBasesArguments, MoeScatterArguments, MoeScatterKernels, MoeScatterWithMapArguments};
pub use tiles_pass_a::{
    MoePassARowMapArguments, MoePassATileBuildArguments, MoePassATileCountsArguments, MoePassATileDispatchArguments,
    MoePassATileKernels, MoePassATileScanArguments,
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
