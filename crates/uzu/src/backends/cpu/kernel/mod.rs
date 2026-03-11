#![allow(unused)]
mod activation;
mod attention;
mod audio;
mod embedding;
mod kv_cache_update;
mod layer_norm;
mod mask_update;
mod matmul;
mod mlp;
mod moe;
mod pooling;
mod quant_matmul;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod sigmoid;
mod ssm;
mod tensor_add_bias;
mod tensor_add_swap;
mod tensor_copy;
mod token_copy;

include!(concat!(env!("OUT_DIR"), "/cpu/dsl.rs"));

use crate::{
    DataType,
    backends::{
        common::kernel::matmul::{
            FullPrecisionMatmulArguments, FullPrecisionMatmulKernel as FullPrecisionMatmulKernelTrait, MatmulError,
            MatmulKernels,
        },
        cpu::{Cpu, command_buffer::CpuCommandBuffer, context::CpuContext},
    },
};

pub struct FullPrecisionMatmulCpuKernel;

impl FullPrecisionMatmulKernelTrait for FullPrecisionMatmulCpuKernel {
    type Backend = Cpu;

    fn new(
        #[allow(unused)] context: &CpuContext,
        #[allow(unused)] data_type: DataType,
    ) -> Result<Self, MatmulError<Cpu>> {
        Ok(Self)
    }

    fn encode(
        &mut self,
        #[allow(unused)] context: &CpuContext,
        #[allow(unused)] encoder: &mut CpuCommandBuffer,
        #[allow(unused)] arguments: FullPrecisionMatmulArguments<Cpu>,
    ) {
        encoder.push_command(move || todo!());
    }
}

impl MatmulKernels for CpuKernels {
    type FullPrecisionMatmulKernel = FullPrecisionMatmulCpuKernel;
}
