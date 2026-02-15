pub mod attention;
use crate::backends::common::kernel::matmul::MatmulKernels;
pub mod dsl {
    include!(concat!(env!("OUT_DIR"), "/dsl.rs"));
}
pub(super) const MTLB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));

pub mod matmul;
pub mod mlp;
pub use mlp::{
    MLP_ACTIVATION_FC_INDEX, MLP_FUSED_FC_INDEX, MLP_HIDDEN_DIM_FC_INDEX, MlpActivationType, MlpFusedConfig,
    make_non_fused_function_constants,
};
pub mod mlp_fused;
pub mod moe;
pub mod quant_matmul;
pub mod ssm;

pub use attention::{AttentionError, AttentionKernel, AttentionKernelVariant};
pub use matmul::{MatmulArguments, MatmulKernel};
pub use moe::{
    MoeBlockBasesArguments, MoeExpertsArguments, MoeExpertsError, MoeExpertsTwoPassArguments,
    MoeExpertsTwoPassDecodeKernels, MoeExpertsTwoPassPrefillKernel, MoeGatherArguments, MoeGatherKernels,
    MoeRouterTopKArguments, MoeRouterTopKKernel, MoeScatterArguments, MoeScatterError, MoeScatterKernels,
    MoeScatterWithMapArguments,
};
pub use ssm::{SSDPrefillArguments, SSDPrefillMode};

impl MatmulKernels for dsl::MetalKernels {
    type FullPrecisionMatmulKernel = MatmulKernel;
    type QuantizedMatmulKernel = quant_matmul::QuantizedMatmulKernel;
}
