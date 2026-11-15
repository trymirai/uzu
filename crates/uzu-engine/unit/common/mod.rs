#[cfg(not(backend = "cpu"))]
compile_error!("uzu tests require cpu backend");

pub mod assert;
pub(crate) mod attention;
pub mod cold_pool;
pub mod harness;
pub mod helpers;
pub mod matmul;
pub mod metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod path;
pub mod perf;
pub mod proptest;
pub mod util;

#[path = "../bench_cold_pool_test.rs"]
mod bench_cold_pool_test;
#[cfg(not(target_arch = "wasm32"))]
#[path = "../session/model_loading_bench.rs"]
mod model_loading_bench;
#[path = "../safetensors_test.rs"]
mod safetensors_test;

#[path = "../minicpm_eval_test.rs"]
mod minicpm_eval_test;

#[path = "../lfm3_eval_test.rs"]
mod lfm3_eval_test;
#[path = "../liquid_codec_test.rs"]
mod liquid_codec_test;
#[path = "../lfm3_audio_eval_test.rs"]
mod lfm3_audio_eval_test;
#[path = "../lfm3_omni_eval_test.rs"]
mod lfm3_omni_eval_test;
#[path = "../d1_omni_eval_test.rs"]
mod d1_omni_eval_test;

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "../backends/metal/kernel/liquid_codec_test.rs"]
mod liquid_codec_kernel_test;

