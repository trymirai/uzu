use crate::backends::{common::Kernels, cpu::Cpu};

mod activation;
mod attention;
mod audio;
mod delta_net;
mod embedding;
mod gated_act_mul;
mod gdn_tree_verify;
mod hadamard_transform;
mod logit_soft_cap;
mod matmul;
mod moe;
mod normalization;
mod pooling;
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
