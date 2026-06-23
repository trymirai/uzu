use crate::backends::{
    common::{
        AsBufferRangeRef, Buffer, Encoder, Kernels,
        kernel::attention_gemm::{AttentionGemmArguments, AttentionGemmBackendBlock, GeneratedAttentionGemmBlock},
    },
    cpu::Cpu,
};

mod activation;
mod attention;
mod audio;
mod delta_net;
mod embedding;
mod gated_act_mul;
mod gdn_tree_verify;
mod hadamard_transform;
mod kv_cache_update;
mod layer_norm;
mod logit_soft_cap;
mod matmul;
mod moe;
mod pooling;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod softmax;
mod ssm;
mod tensor_add_bias;
mod tensor_add_scale;
mod tensor_add_swap;
mod tensor_copy;
mod token_copy;

include!(concat!(env!("OUT_DIR"), "/cpu/dsl.rs"));

pub struct CpuKernels;

impl AttentionGemmBackendBlock for GeneratedAttentionGemmBlock<CpuKernels> {
    type Backend = Cpu;

    fn new(data_type: crate::data_type::DataType) -> Self {
        GeneratedAttentionGemmBlock::new(data_type)
    }

    fn encode<KVBuf: AsBufferRangeRef<Buffer: Buffer<Backend = Cpu>>>(
        &self,
        encoder: &mut Encoder<Cpu>,
        args: AttentionGemmArguments<Cpu, KVBuf>,
    ) -> Result<(), crate::backends::cpu::error::CpuError> {
        self.encode_with_accelerator(false, encoder, args)
    }
}

impl Kernels for CpuKernels {
    type Backend = Cpu;

    autogen_kernels!();
    type AttentionGemmBlock = GeneratedAttentionGemmBlock<CpuKernels>;
    type MatmulKernel = matmul::MatmulCpuKernel;
}
