mod common;

use uzu::{
    backends::metal::Metal,
    speculators::{neural_speculator::NeuralSpeculator, speculator::Speculator},
    trie::TrieCreationConfig,
};

// ── stub ──────────────────────────────────────────────────────────────────────

/// Creates a `NeuralSpeculator<Metal>` without loading any GPU model.
/// All fields are left uninitialised.
///
/// # Safety
/// Only use with methods that return before accessing any field:
/// - `trie_creation_config()` — returns a constant.
/// - `speculate(&[])` — empty prefix early return.
///
/// `ManuallyDrop` prevents `Drop` from running on the uninitialised value.
#[allow(invalid_value)]
fn stub() -> std::mem::ManuallyDrop<NeuralSpeculator<Metal>> {
    // SAFETY: no field is accessed by the tested code paths.
    unsafe { std::mem::ManuallyDrop::new(std::mem::MaybeUninit::<NeuralSpeculator<Metal>>::uninit().assume_init()) }
}

// ── constructor error paths (no model required) ───────────────────────────────

#[test]
fn test_new_nonexistent_path_fails() {
    let result = NeuralSpeculator::<Metal>::new(std::path::Path::new("/nonexistent/nowhere"), 4, 8);
    assert!(result.is_err(), "expected Err for a non-existent model path");
}

/// `Llama-3.2-1B-Instruct` has a valid `config.json` but no `pard_token` field —
/// `new()` must fail before touching GPU weights.
#[test]
fn test_new_model_without_pard_token_fails() {
    let model_path = common::get_test_model_path();
    let result = NeuralSpeculator::<Metal>::new(&model_path, 4, 8);
    assert!(
        matches!(result, Err(uzu::session::types::Error::UnsupportedSpeculatorConfigForModel)),
        "expected UnsupportedSpeculatorConfigForModel, got: {:?}",
        result.err()
    );
}

// ── pure-logic methods (stub, no GPU) ────────────────────────────────────────

/// `trie_creation_config` must return `width = 1` (trunk-only linear chain),
/// unlike the trait default of `width = 4`.
#[test]
fn test_trie_creation_config_is_width_1() {
    let spec = stub();
    assert_eq!(spec.trie_creation_config().width, 1);
    assert_ne!(spec.trie_creation_config().width, TrieCreationConfig::default().width);
}

#[test]
fn test_speculate_empty_prefix_returns_empty() {
    let spec = stub();
    assert!(spec.speculate(&[]).is_empty());
}

// ── integration (requires PARD_MODEL_PATH env var) ────────────────────────────
// Run with: PARD_MODEL_PATH=/path/to/pard cargo test -- --ignored

#[test]
#[ignore = "requires PARD_MODEL_PATH env var pointing to a PARD draft model"]
fn test_integration_trie_creation_config_is_width_1() {
    let path = std::env::var("PARD_MODEL_PATH").expect("set PARD_MODEL_PATH");
    let spec = NeuralSpeculator::<Metal>::new(std::path::Path::new(&path), 4, 8).expect("load PARD model");
    assert_eq!(spec.trie_creation_config().width, 1);
}

#[test]
#[ignore = "requires PARD_MODEL_PATH env var pointing to a PARD draft model"]
fn test_integration_speculate_returns_candidates() {
    let path = std::env::var("PARD_MODEL_PATH").expect("set PARD_MODEL_PATH");
    let spec = NeuralSpeculator::<Metal>::new(std::path::Path::new(&path), 4, 8).expect("load PARD model");
    let prefix = &[1u64, 2, 3, 10];
    spec.prepare(prefix);
    let result = spec.speculate(prefix);
    assert!(!result.is_empty());
}
