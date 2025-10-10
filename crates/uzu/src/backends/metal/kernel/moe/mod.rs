use crate::backends::metal::KernelDataType;

mod encodable;
pub use encodable::{MoeBlockEncodable, SharedMoeWeights};

// Submodules
mod bucket_counts;
mod experts;
mod finalize;
mod gather;
mod offsets_scan;
mod router;
mod scatter;
mod tiles;
mod topk;

// Re-export public items from submodules
pub use bucket_counts::{
    MoeBucketCountsArguments, MoeBucketCountsError, MoeBucketCountsKernel,
    encode_moe_bucket_counts, encode_moe_bucket_counts_with_pipelines,
};
pub use experts::{
    MoeExpertsArguments, MoeExpertsError, MoeExpertsFusedKernel,
    MoeExpertsTwoPassArguments, MoeExpertsTwoPassDecodeKernel,
    MoeExpertsTwoPassPrefillKernel, MoeScatterError,
};
pub use finalize::{
    MoeFinalizeArguments, MoeFinalizeError, MoeFinalizeKernel,
    encode_moe_finalize, encode_moe_finalize_with_pipeline,
};
pub use gather::{
    MoeGatherArguments, MoeGatherError, MoeGatherKernel, encode_moe_gather,
    encode_moe_gather_with_pipeline,
};
pub use offsets_scan::{
    MoeOffsetsScanArguments, MoeOffsetsScanError, MoeOffsetsScanKernel,
    encode_moe_offsets_scan, encode_moe_offsets_scan_with_pipeline,
};
pub use router::{
    MoeRouterArguments, MoeRouterError, MoeRouterKernel, RouterEncoderArgs,
    encode_moe_router, encode_moe_router_with_pipeline,
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
    MoeTileMapKernel, MoeTileScanArguments, encode_moe_tile_counts,
    encode_moe_tile_scan,
};
pub use topk::{
    MoeTopKArguments, MoeTopKError, MoeTopKKernel, encode_moe_topk,
    encode_moe_topk_with_pipeline,
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
