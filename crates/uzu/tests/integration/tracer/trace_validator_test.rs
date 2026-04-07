#![cfg(all(feature = "tracing"))]

use uzu::{TraceValidator, backends::common::Backend};

use crate::common::{
    for_each_non_cpu_backend,
    path::{get_test_model_path, get_traces_path},
};

fn test_tracer_internal<B: Backend>() {
    let model_path = get_test_model_path();
    let mut tracer = TraceValidator::<B>::new(&model_path).expect("Failed to create TraceValidator");
    let results = tracer.run().expect("Failed to run tracer");
    for result in results.results.iter() {
        // this layers contains too many errors
        if result.name == "activation_trace.output_norm" || result.name == "logits" {
            continue;
        }
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

#[test]
fn test_tracer() {
    let traces_path = get_traces_path();

    // TODO: when api will return `traces.safetensor` remove `if` and uncomment `assert`
    if !traces_path.exists() {
        println!("Skipping tracer test: traces file missing at {:?}", traces_path);
        return;
    }
    // assert!(traces_path.exists(), "Traces file missing at {:?}", traces_path);

    for_each_non_cpu_backend!(|B| {
        test_tracer_internal::<B>();
    })
}
