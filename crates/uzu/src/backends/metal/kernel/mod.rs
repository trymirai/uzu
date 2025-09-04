pub mod attention;
mod data_type;
pub mod kv_cache_update;
pub mod linear;
pub mod media_kernels;
pub mod mlp;
pub mod moe;
pub mod quant_matmul;
pub mod rms_norm;
pub mod rope;
pub mod sampling;
mod tensor_add_swap;
mod tensor_copy;

pub use attention::{AttentionKernel, AttentionKernelEncodable};
pub use data_type::KernelDataType;
pub use kv_cache_update::KVCacheUpdate;
pub use moe::{
    MoeBlockBasesArguments, MoeBucketCountsArguments, MoeBucketCountsError,
    MoeBucketCountsKernel, MoeExpertsArguments, MoeExpertsError,
    MoeExpertsKernel, MoeOffsetsScanArguments, MoeOffsetsScanError,
    MoeOffsetsScanKernel, MoeScatterArguments, MoeScatterError,
    MoeScatterKernels, MoeTopKArguments, MoeTopKError, MoeTopKKernel,
};
pub use rms_norm::{
    QKNormArguments, QKNormKernelEncodable, RMSNormArguments, RMSNormKernel,
    RMSNormKernelEncodable, RMSNormKernelType,
};
pub use rope::{RopeKernel, RopeKernelEncodable};
pub use sampling::{SamplingKernel, SamplingKernelEncodable};
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
