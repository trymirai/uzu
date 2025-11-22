pub mod attention;
mod data_type;
pub mod embedding;
pub mod kv_cache_update;
pub mod linear;
pub mod media_kernels;
pub mod mlp;
pub mod moe;
pub mod quant_matmul;
pub mod rms_norm;
pub mod rope;
pub mod sampling;
pub mod ssm;
mod tensor_add_bias;
mod tensor_add_swap;
mod tensor_copy;

pub use attention::{AttentionKernel, AttentionKernelEncodable};
pub use data_type::KernelDataType;
pub use kv_cache_update::KVCacheUpdate;
pub use moe::{
    MoeBlockBasesArguments, MoeBlockEncodable, MoeCountsOffsetsFusedArguments,
    MoeCountsOffsetsFusedError, MoeCountsOffsetsFusedKernel,
    MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernel, MoeExpertsTwoPassPrefillKernel,
    MoeFinalizeArguments, MoeFinalizeError, MoeFinalizeKernel,
    MoeScatterArguments, MoeScatterError, MoeScatterKernels,
    MoeScatterWithMapArguments,
};
pub use rms_norm::{
    QKNormArguments, QKNormKernelEncodable, RMSNormArguments, RMSNormKernel,
    RMSNormKernelEncodable, RMSNormKernelType,
};
pub use rope::{RopeKernel, RopeKernelEncodable};
pub use sampling::{SamplingKernel, SamplingKernelEncodable};
pub(crate) use ssm::MambaMixerEncodable;
pub use ssm::{
    Conv1dPackArguments, Conv1dScanArguments, Conv1dScanKernel,
    SSDPrefillArguments, SSDPrefillKernel, SSDPrefillMode, SSDUpdateArguments,
    SSDUpdateKernel, SSMKernelError, SSMUpdateArguments, SSMUpdateKernel,
    SplitInProjArguments, SplitInProjKernel,
};
pub use tensor_add_bias::TensorAddBias;
pub use tensor_add_swap::TensorAddSwap;
pub use tensor_copy::TensorCopy;

use super::{MTLContext, metal_extensions};
pub use crate::backends::metal::kernel::linear::QuantizedLinearKernelBlock;

pub mod media {
    pub use super::media_kernels::{
        ExtractImagePatches, ScalePadNormalizeImage,
    };
}

pub use media_kernels::{
    ExtractImagePatches, ImageParameters, PatchParameters,
    ScalePadNormalizeImage,
};
