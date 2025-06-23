mod common;
use uzu::decoder_runner::run_decoder_with_results;

#[test]
#[ignore]
fn decoder_runs_successfully() {
    // Skip the test completely if we're not on a Metal-capable platform.
    if !cfg!(any(target_os = "macos", target_os = "ios")) {
        eprintln!("Skipping decoder test on non-Metal platform");
        return;
    }

    // Build the path to the test model
    let model_path = common::get_test_model_path();

    // Run the decoder and collect the results
    let result = run_decoder_with_results(model_path.to_str().unwrap());

    // Always print the results so they are visible when running with `cargo test -- --nocapture`
    println!("Placement Log:\n{}", result.placement_log);
    println!("Iterations: {}", result.iterations);
    println!("Time per token: {}", result.time_per_token);
    println!("Tokens per second: {}", result.tokens_per_second);
    println!("Success: {}", result.success);
    if let Some(error) = &result.error {
        println!("Error: {}", error);
    }

    // Ensure the decoder executed without errors
    if !result.success {
        panic!(
            "Decoder test failed. Error: {:?}\nPlacement Log:\n{}",
            result.error, result.placement_log
        );
    }

    // Basic sanity checks on the performance numbers
    assert!(result.iterations > 0, "Expected at least one iteration.");
    assert!(result.time_per_token > 0.0, "Time per token should be positive.");
    assert!(
        result.tokens_per_second > 0.0,
        "Tokens per second should be positive."
    );
}
