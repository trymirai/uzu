use std::convert::Infallible;

use crate::backends::{common::Kernels, cpu::Cpu};

mod activation;
pub(crate) mod activation_transform;
mod attention;
mod embedding;
mod gated_act_mul;
mod gdn;
mod logit_transform;
mod matmul;
mod moe;
mod normalization;
mod pooling;
mod radix_top_k_small;
mod sampling;
mod short_conv;
mod softmax;
mod ssm;
mod tensor_add_bias;
mod tensor_add_scale;
mod tensor_add_swap;
mod tensor_copy;
mod weaver;

use super::error::CpuError;

include!(concat!(env!("OUT_DIR"), "/cpu.rs"));

pub struct CpuKernels;

impl Kernels for CpuKernels {
    type Backend = Cpu;

    autogen_kernels!();
    type AttentionGemmCore = Infallible;
    type DeltaNetChunkedPrefill = Infallible;
    type DeltaNetTreeVerify = Infallible;
    type MatmulKernel = matmul::MatmulCpuKernel;
    type QtipSExactKernel = QtipSExactCpuKernel;
    type RadixTopKSmall = radix_top_k_small::CpuRadixTopKSmall;
}

pub struct QtipSExactCpuKernel;

impl crate::backends::common::kernel::qtip_s_exact::QtipSExactKernel<Cpu> for QtipSExactCpuKernel {
    fn new(_context: &<Cpu as crate::backends::common::Backend>::Context) -> Result<Self, CpuError> {
        panic!("the physical QTIP S checkpoint is Metal-only")
    }

    fn encode_qtip_gaussian(
        &self,
        _arguments: crate::backends::common::kernel::qtip_s_exact::QtipGaussianArguments<'_, Cpu>,
        _encoder: &mut crate::backends::common::Encoder<Cpu>,
    ) -> Result<crate::backends::common::Allocation<Cpu>, CpuError> {
        unreachable!()
    }

    fn encode_d4_s4_embedding(
        &self,
        _arguments: crate::backends::common::kernel::qtip_s_exact::D4S4EmbeddingArguments<'_, Cpu>,
        _encoder: &mut crate::backends::common::Encoder<Cpu>,
    ) {
        unreachable!()
    }

    fn encode_i3_s4_readout(
        &self,
        _arguments: crate::backends::common::kernel::qtip_s_exact::I3S4ReadoutArguments<'_, Cpu>,
        _encoder: &mut crate::backends::common::Encoder<Cpu>,
    ) -> Result<crate::backends::common::Allocation<Cpu>, CpuError> {
        unreachable!()
    }

    fn encode_i3_s4_readout_sparse(
        &self,
        _arguments: crate::backends::common::kernel::qtip_s_exact::I3S4SparseReadoutArguments<'_, Cpu>,
        _encoder: &mut crate::backends::common::Encoder<Cpu>,
    ) -> Result<crate::backends::common::Allocation<Cpu>, CpuError> {
        unreachable!()
    }

    fn encode_residual_merge_hot(
        &self,
        _hot: &crate::backends::common::Allocation<Cpu>,
        _cold: &crate::backends::common::Allocation<Cpu>,
        _token_ids: &crate::backends::common::Allocation<Cpu>,
        _output: &mut crate::backends::common::Allocation<Cpu>,
        _hot_rows: u32,
        _count: u32,
        _encoder: &mut crate::backends::common::Encoder<Cpu>,
    ) {
        unreachable!()
    }
}
