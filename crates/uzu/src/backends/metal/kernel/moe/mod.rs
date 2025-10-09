use crate::backends::metal::KernelDataType;

mod encodable;
pub use encodable::{MoeBlockEncodable, SharedMoeWeights};

// Submodules
mod router;
mod topk;
mod bucket_counts;
mod offsets_scan;
mod tiles;
mod gather;
mod experts;
mod scatter;
mod finalize;

// Re-export public items from submodules
pub use router::{
    encode_moe_router, encode_moe_router_with_pipeline, MoeRouterArguments, MoeRouterError,
    MoeRouterKernel, RouterEncoderArgs,
};

pub use topk::{
    encode_moe_topk, encode_moe_topk_with_pipeline, MoeTopKArguments, MoeTopKError,
    MoeTopKKernel,
};

pub use bucket_counts::{
    encode_moe_bucket_counts, encode_moe_bucket_counts_with_pipelines, MoeBucketCountsArguments,
    MoeBucketCountsError, MoeBucketCountsKernel,
};

pub use offsets_scan::{
    encode_moe_offsets_scan, encode_moe_offsets_scan_with_pipeline, MoeOffsetsScanArguments,
    MoeOffsetsScanError, MoeOffsetsScanKernel,
};

pub use tiles::{
    encode_moe_tile_counts, encode_moe_tile_scan, MoePassARowMapArguments,
    MoePassATileBuildArguments, MoePassATileCountsArguments, MoePassATileDispatchArguments,
    MoePassATileKernel, MoePassATileScanArguments, MoeTileCountsArguments,
    MoeTileDispatchArguments, MoeTileError, MoeTileMapBuildArguments, MoeTileMapKernel,
    MoeTileScanArguments,
};

pub use gather::{
    encode_moe_gather, encode_moe_gather_with_pipeline, MoeGatherArguments, MoeGatherError,
    MoeGatherKernel,
};

pub use experts::{
    MoeExpertsArguments, MoeExpertsError, MoeExpertsFusedKernel, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel, MoeScatterError,
};

pub use scatter::{
    MoeBlockBasesArguments, MoeScatterArguments, MoeScatterKernels, MoeScatterWithMapArguments,
};

pub use finalize::{
    encode_moe_finalize, encode_moe_finalize_with_pipeline, MoeFinalizeArguments,
    MoeFinalizeError, MoeFinalizeKernel,
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
