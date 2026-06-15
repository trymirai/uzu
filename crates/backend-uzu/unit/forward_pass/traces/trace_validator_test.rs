use std::path::PathBuf;

use proc_macros::uzu_test;
use test_runner::{for_each_non_cpu_backend, path::get_test_model_path};

use crate::{backends::common::Backend, forward_pass::traces::tests::trace_validator::TraceValidator};

const TRACES_FILE_NAME: &str = "traces.safetensors";

fn get_traces_path() -> PathBuf {
    get_test_model_path().join(TRACES_FILE_NAME)
}

fn test_tracer_internal<B: Backend>() {
    let model_path = get_test_model_path();
    let mut tracer = TraceValidator::<B>::new(&model_path).expect("Failed to create TraceValidator");
    let results = tracer.run().expect("Failed to run tracer");
    for result in results.results.iter() {
        assert!(result.metrics.is_valid(), "{} error:\n{}", result.name, result.metrics.message().as_str());
    }

    let total_token_violations = results.number_of_tokens_violations();
    let allowed_token_violations = results.number_of_allowed_tokens_violations();
    assert!(
        total_token_violations < allowed_token_violations,
        "Too much token violations: {} / {}. Indices: {:?}",
        total_token_violations,
        allowed_token_violations,
        results.tokens_violation_indices
    );
}

#[uzu_test]
#[ignore = "Lalamo 0.10.0 doesn't support exporting traces"] // TODO: this is horrible, should be resolved asap
fn test_tracer() {
    let traces_path = get_traces_path();
    assert!(traces_path.exists(), "Traces file missing at {:?}", traces_path);

    for_each_non_cpu_backend!(|B| {
        test_tracer_internal::<B>();
    })
}
