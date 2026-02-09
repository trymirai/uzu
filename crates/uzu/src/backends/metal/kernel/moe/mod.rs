// Submodules
mod experts;
mod experts_single;
mod experts_two_pass_decode;
mod experts_two_pass_prefill;
mod gather;
mod router_topk;
mod scatter;
pub mod tiles_map;
pub mod tiles_pass_a;

// Re-export public items from submodules
pub use experts::{
    MoeExpertsTwoPassArguments
};
pub use experts_single::{
    MoeExpertsSingleDecodeArguments, MoeExpertsSingleDecodeKernels,
};
pub use experts_two_pass_decode::MoeExpertsTwoPassDecodeKernels;
pub use experts_two_pass_prefill::MoeExpertsTwoPassPrefillKernels;
pub use gather::{MoeGatherArguments, MoeGatherKernels};
pub use router_topk::{
    MoeRouterTopKArguments, MoeRouterTopKError, MoeRouterTopKKernel,
};
pub use scatter::{
    MoeBlockBasesArguments, MoeScatterArguments, MoeScatterKernels,
    MoeScatterWithMapArguments,
};
pub use tiles_pass_a::{
    MoePassARowMapArguments, MoePassATileBuildArguments,
    MoePassATileCountsArguments, MoePassATileDispatchArguments,
    MoePassATileKernels, MoePassATileScanArguments,
};
use crate::backends::metal::KernelDataType;

// Common utility functions
pub(crate) fn dtype_suffix(dtype: KernelDataType) -> &'static str {
    match dtype {
        KernelDataType::Float16 => "f16",
        KernelDataType::BFloat16 => "bf16",
        KernelDataType::Float32 => "f32",
    }
}
