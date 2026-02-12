use crate::DataType;

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
pub(crate) fn dtype_suffix(dtype: DataType) -> &'static str {
    match dtype {
        DataType::F16 => "f16",
        DataType::BF16 => "bf16",
        DataType::F32 => "f32",
        _ => panic!("Unsupported data type: {:?}", dtype),
    }
}

pub(crate) fn dtype_index(dtype: DataType) -> usize {
    match dtype {
        DataType::F16 => 0,
        DataType::BF16 => 1,
        DataType::F32 => 2,
        _ => panic!("Unsupported data type: {:?}", dtype),
    }
}
