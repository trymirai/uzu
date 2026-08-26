use std::{convert::Infallible, marker::PhantomData};

use crate::backends::common::Backend;

pub mod activation_transform;
pub mod attention;
pub mod delta_net_chunked_prefill;
pub mod delta_net_tree_verify;
pub mod gated_act_mul;
pub mod matmul;
pub mod radix_top_k_small;

pub use activation_transform::ActivationTransform;
pub use attention::{AttentionArguments, AttentionKernel, AttentionKernelConfig};
pub use gated_act_mul::{GatedActMul, GatedActMulSettings};

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub trait Kernels: Sized {
    type Backend: Backend<Kernels = Self>;

    autogen_kernels!();
    type AttentionKernel: attention::AttentionKernel<Backend = Self::Backend>;
    type DeltaNetChunkedPrefill: delta_net_chunked_prefill::DeltaNetChunkedPrefill<Backend = Self::Backend>;
    type DeltaNetTreeVerify: delta_net_tree_verify::DeltaNetTreeVerify<Backend = Self::Backend>;
    type MatmulKernel: matmul::MatmulKernel<Backend = Self::Backend>;
    type RadixTopKSmall: radix_top_k_small::RadixTopKSmall<Backend = Self::Backend>;
}

pub struct Unsupported<B: Backend> {
    never: Infallible,
    _marker: PhantomData<fn() -> B>,
}

#[cfg(test)]
#[path = "../../../../unit/backends/common/kernel/mod.rs"]
mod tests;
