#![cfg(feature = "tracing")]

mod common;
use std::path::PathBuf;

use uzu::tracer::TraceValidator;

fn build_model_path() -> PathBuf {
    common::get_test_model_path()
}

#[test]
fn test_tracer() {
    let model_path = build_model_path();
    // Ensure traces file present, otherwise skip
    let traces_path = crate::common::get_traces_path();
    if !traces_path.exists() {
        println!(
            "Skipping tracer test: traces file missing at {:?}",
            traces_path
        );
        return;
    }

    let mut tracer = match TraceValidator::new(&model_path) {
        Ok(t) => t,
        Err(e) => {
            println!("Failed to create TraceValidator: {:?}", e);
            return;
        },
    };

    let colored_text = |text: &str, valid: bool| {
        if valid {
            format!("\x1b[32m{}\x1b[0m", text)
        } else {
            format!("\x1b[31m{}\x1b[0m", text)
        }
    };

    let results = match tracer.run() {
        Ok(r) => r,
        Err(e) => {
            println!("Tracer run failed: {:?}", e);
            return;
        },
    };

    for result in results.results.iter() {
        let valid = result.metrics.is_valid();
        let text = colored_text(
            if valid {
                "ok"
            } else {
                "error"
            },
            valid,
        );
        println!("{}: {}", result.name, text);
        if !valid {
            println!(
                "{}",
                colored_text(result.metrics.message().as_str(), false)
            );
        }
    }
    println!("-------------------------");
    println!(
        "number_of_tokens_violations: {}",
        colored_text(
            format!(
                "{} / {}",
                results.number_of_tokens_violations(),
                results.number_of_allowed_tokens_violations()
            )
            .as_str(),
            results.is_valid(),
        )
    );
    println!(
        "tokens_violation_indices: {:?}",
        results.tokens_violation_indices
    );
}
