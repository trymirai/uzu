use std::path::PathBuf;

use uzu::session::{ClassificationSession, types::Input};

#[test]
fn test_bert_classifier_validation() {
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models/bert");

    if !model_path.exists() {
        println!("Skipping test: BERT model not found at {:?}", model_path);
        return;
    }

    let mut session = ClassificationSession::new(model_path)
        .expect("Failed to create classification session");

    let test_cases = vec![
        ("This is a bad idea", 0.605469),
        ("This is a good thing", 0.609375),
    ];

    for (input_text, expected_prob) in test_cases {
        let result = session
            .classify(Input::Text(input_text.to_string()))
            .expect("Failed to classify");

        println!("\nInput: {}", input_text);
        println!("Logits: {:?}", result.logits);
        println!("Probabilities: {:?}", result.probabilities);

        assert_eq!(result.probabilities.len(), 1, "Expected exactly 1 label");

        let label_0_prob = result
            .probabilities
            .get("LABEL_0")
            .expect("LABEL_0 not found in output");

        println!(
            "Expected: {:.6}, Got: {:.6}, Diff: {:.6}",
            expected_prob,
            label_0_prob,
            (label_0_prob - expected_prob).abs()
        );

        let tolerance = 0.02; // Relaxed tolerance for model export variations
        assert!(
            (label_0_prob - expected_prob).abs() < tolerance,
            "Probability mismatch: expected {:.6}, got {:.6}",
            expected_prob,
            label_0_prob
        );
    }
}
