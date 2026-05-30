use crate::backends::{common::Kernels, cpu::Cpu};

mod activation;
mod attention;
mod audio;
mod delta_net;
mod embedding;
mod hadamard_transform;
mod kv_cache_update;
mod layer_norm;
mod logit_soft_cap;
mod matmul;
mod mlp;
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

impl Kernels for CpuKernels {
    type Backend = Cpu;

    autogen_kernels!();
    type MatmulKernel = matmul::MatmulCpuKernel;
}
