#![allow(unused)]
mod activation;
mod attention;
mod audio;
mod delta_net;
mod embedding;
mod hadamard_transform;
mod kv_cache_update;
mod layer_norm;

mod matmul;
mod mlp;
mod moe;
mod pooling;
mod quant_matmul;
mod rms_norm;
mod rope;
mod sampling;
mod short_conv;
mod ssm;
mod tensor_add_bias;
mod tensor_add_scale;
mod tensor_add_swap;
mod tensor_copy;
mod token_copy;

include!(concat!(env!("OUT_DIR"), "/cpu/dsl.rs"));

/// Public re-export of the NF4-E4M3 CPU reference helpers for the
/// bench/correctness test crate (NF4 GPU kernels are bench-only).
pub use quant_matmul::nf4_e4m3;
