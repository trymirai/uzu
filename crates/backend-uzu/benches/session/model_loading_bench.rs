use backend_uzu::session::{ChatSession, config::DecodingConfig};
use proc_macros::uzu_test;
use test_runner::{path::get_test_model_path, perf::run_perf};

#[uzu_test]
fn test_perf_model_loading() {
    let perf_results = run_perf("Model loading", 10, || {
        let model_path = get_test_model_path();
        let _ = ChatSession::new(model_path, DecodingConfig::default());
    });
    perf_results.print();
}
