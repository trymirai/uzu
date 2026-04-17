use uzu::prelude::{ChatSession, DecodingConfig};

use crate::common::{path::get_test_model_path, perf::run_perf};

#[test]
#[ignore]
fn test_perf_model_loading() {
    let perf_results = run_perf("Model loading", 10, || {
        let model_path = get_test_model_path();
        let _ = ChatSession::new(model_path, DecodingConfig::default());
    });
    perf_results.print();
}
