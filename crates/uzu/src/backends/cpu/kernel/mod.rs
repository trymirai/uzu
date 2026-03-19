#![allow(unused)]
mod activation;
mod attention;
mod audio;
mod embedding;
mod hadamard_transform;
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
