pub mod attention;
mod data_type;
pub mod embedding;
pub mod kv_cache_update;
pub mod layer_norm;
pub mod media_kernels;
pub mod mlp;
pub mod moe;
pub mod pooling;
pub mod quant_matmul;
pub mod rms_norm;
pub mod rope;
pub mod sampling;
pub mod sigmoid;
pub mod ssm;
mod tensor_add_bias;
mod tensor_add_swap;
mod tensor_copy;

pub use attention::{
    AttentionError, AttentionKernel, AttentionKernelVariant,
    AttentionSinglePassArguments, AttentionTwoPassArguments,
    KVCacheUpdateArguments,
};
pub use data_type::KernelDataType;
pub use kv_cache_update::KVCacheUpdate;
pub use layer_norm::{LayerNormArguments, LayerNormError, LayerNormKernel};
pub use moe::{
    MoeBlockBasesArguments, MoeCountsOffsetsFusedArguments,
    MoeCountsOffsetsFusedError, MoeCountsOffsetsFusedKernel,
    MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel,
    MoeFinalizeArguments, MoeFinalizeError, MoeFinalizeKernel,
    MoeGatherArguments, MoeGatherKernel, MoeRouterTopKArguments,
    MoeRouterTopKKernel, MoeScatterArguments, MoeScatterError,
    MoeScatterKernels, MoeScatterWithMapArguments,
};
pub use pooling::PoolingKernel;
pub use rms_norm::{
    QKNormArguments, QKNormTarget, RMSNormArguments, RMSNormError,
    RMSNormKernel, RMSNormKernelType,
};
pub use rope::{RopeError, RopeKernel, RopeKernelArguments};
pub use sampling::{ArgmaxStrategy, SamplingError, SamplingKernel};
pub use sigmoid::SigmoidKernel;
pub use ssm::{
    Conv1dPackArguments, Conv1dScanArguments, Conv1dScanKernel,
    SSDPrefillArguments, SSDPrefillKernel, SSDPrefillMode, SSDUpdateArguments,
    SSDUpdateKernel, SSMKernelError, SplitInProjArguments, SplitInProjKernel,
};
pub use tensor_add_bias::TensorAddBias;
pub use tensor_add_swap::TensorAddSwapKernel;
pub use tensor_copy::TensorCopyKernel;

use super::{MTLContext, metal_extensions};

pub mod media {
    pub use super::media_kernels::{
        ExtractImagePatches, ScalePadNormalizeImage,
    };
}

pub use media_kernels::{
    ExtractImagePatches, ImageParameters, PatchParameters,
    ScalePadNormalizeImage,
};
