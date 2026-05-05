#![allow(unused)]
mod activation;
mod attention;
mod audio;
mod delta_net;
mod embedding;
mod hadamard_transform;
mod kv_cache_update;
mod layer_norm;
mod model_extension;

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
