#[macro_use]
#[path = "../../common/mod.rs"]
mod common;

mod activation_mul_test;
mod attention;
mod audio;
mod full_precision_embedding_test;
mod kv_cache_update_test;
mod mask_update_test;
mod matmul;
mod moe;
mod qk_norm_test;
mod rms_norm_test;
mod rope_test;
mod sampling;
mod tensor_add_bias_test;
mod tensor_add_swap_test;
mod tensor_copy_test;
mod token_copy_test;
