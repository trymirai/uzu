mod common;

use std::path::PathBuf;

use uzu::session::{ClassificationSession, types::Input};

fn build_model_path() -> PathBuf {
    PathBuf::from("models/modern_bert")
}

#[test]
fn test_classifier_basic() {
    let model_path = build_model_path();

    if !model_path.exists() {
        println!("Skipping test: BERT model not found at {:?}", model_path);
        return;
    }

    let mut session = ClassificationSession::new(model_path)
        .expect("Failed to create classification session");

    let input_text = "This is a test message for classification.".to_string();
    let result = session
        .classify(Input::Text(input_text.clone()))
        .expect("Failed to classify");

    println!("Input: {}", input_text);
    println!("Logits: {:?}", result.logits);
    println!("Probabilities: {:?}", result.probabilities);
    println!("Stats: {:?}", result.stats);

    assert_eq!(result.logits.len(), result.probabilities.len());
    assert!(result.probabilities.len() > 0);

    for (_label, prob) in &result.probabilities {
        assert!(
            *prob >= 0.0 && *prob <= 1.0,
            "Probability {} is out of range [0, 1]",
            prob
        );
    }
}

#[test]
fn test_classifier_multiple_inputs() {
    let model_path = build_model_path();

    if !model_path.exists() {
        println!("Skipping test: BERT model not found at {:?}", model_path);
        return;
    }

    let mut session = ClassificationSession::new(model_path)
        .expect("Failed to create classification session");

    let test_inputs = vec![
        "Hello world".to_string(),
        "This is a longer message with more tokens to test the classifier."
            .to_string(),
        "Short".to_string(),
        "This is a test message for moderation.".to_string(),
    ];

    for input_text in test_inputs {
        let result = session
            .classify(Input::Text(input_text.clone()))
            .expect("Failed to classify");

        println!("\nInput: {}", input_text);
        println!("Result: {:?}", result.probabilities);

        assert!(!result.probabilities.is_empty());
        for (_label, prob) in &result.probabilities {
            assert!(*prob >= 0.0 && *prob <= 1.0);
        }
    }
}
