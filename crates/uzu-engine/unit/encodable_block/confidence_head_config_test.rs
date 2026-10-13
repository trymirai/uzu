//! The head's `config.json` in format 2 and the helpers around it: the file's groups map onto the flat config, a file
//! of another kind or format and a malformed speculator sha256 are refused, the file hash is SHA-256, and the head
//! applies only under samplers that sample.

use std::path::PathBuf;

use uzu_engine_macros::uzu_test;

use super::{ConfidenceHead, ConfidenceHeadConfig, ConfidenceHeadSampling, file_sha256};
use crate::{backends::cpu::Cpu, encodable_block::sampling::SamplingMethod};

const SHA: &str = "6dc095b54b82be959f1c087136284e5df8322b661410df1a51f5395e26e89eeb";

fn config_file(
    sha: &str,
    format: u32,
) -> serde_json::Value {
    serde_json::json!({
        "type": "ConfidenceHeadConfig",
        "format": format,
        "trained_with": {
            "speculator_sha256": sha,
            "tree": {"children": 4, "pool": 512, "max_depth": 15},
            "sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": null}
        },
        "target_width": 5120,
        "local": {"hidden": 64, "pool_hidden": 16, "edge_hidden": 128, "logit_cap": 50.0},
        "attention": {"layers": 4, "heads": 4, "width": 256, "mlp_hidden": 512, "relations": 6},
        "layer_norm_eps": 1e-5
    })
}

fn parse(file: serde_json::Value) -> Result<ConfidenceHeadConfig, serde_json::Error> {
    serde_json::from_value(file)
}

/// Spec (format 2): every grouped field lands in its flat field, and a well-formed file passes the loader's check.
#[uzu_test]
fn config_file_groups_map_onto_the_flat_config() {
    let config = parse(config_file(SHA, 2)).unwrap();
    assert_eq!(config.format, 2);
    assert_eq!(config.speculator_sha256, SHA);
    assert_eq!(
        config.sampling,
        ConfidenceHeadSampling {
            temperature: 1.0,
            top_k: Some(20),
            top_p: Some(0.95),
            min_p: None,
        }
    );
    assert_eq!((config.children, config.pool, config.max_depth), (4, 512, 15));
    assert_eq!(config.model_width, 5120);
    assert_eq!((config.hidden, config.pool_hidden, config.edge_hidden), (64, 16, 128));
    assert_eq!(config.cap, 50.0);
    assert_eq!(config.layer_norm_eps, 1e-5);
    let tree = config.tree.clone().unwrap();
    assert_eq!((tree.layers, tree.heads, tree.width, tree.mlp_hidden, tree.relations), (4, 4, 256, 512, 6));
    ConfidenceHead::<Cpu>::check(&config).unwrap();
}

/// Spec: a head without tree attention leaves `attention` out and runs its local part only.
#[uzu_test]
fn config_file_without_attention_has_no_tree() {
    let mut file = config_file(SHA, 2);
    file.as_object_mut().unwrap().remove("attention");
    assert!(parse(file).unwrap().tree.is_none());
}

/// Spec: the file names its kind; a config of another kind is refused.
#[uzu_test]
fn config_file_of_another_kind_is_refused() {
    let mut file = config_file(SHA, 2);
    file["type"] = serde_json::json!("SpeculatorModelConfig");
    assert!(parse(file).is_err());
}

/// Spec: format 1 (flat, nothing on what the head was trained with) is not read as format 2.
#[uzu_test]
fn config_file_in_the_flat_format_1_is_refused() {
    let file = serde_json::json!({
        "format": 1, "hidden": 16, "pool_hidden": 4, "edge_hidden": 32, "children": 3, "pool": 64, "max_depth": 4,
        "model_width": 48, "cap": 50.0, "layer_norm_eps": 1e-5,
        "tree": {"layers": 2, "heads": 2, "width": 32, "mlp_hidden": 64, "relations": 6}
    });
    assert!(parse(file).is_err());
}

/// Spec: the loader takes format 2 only, and a speculator sha256 of exactly 64 lowercase hex digits.
#[uzu_test]
fn config_check_refuses_other_formats_and_malformed_sha256() {
    let upper = SHA.to_uppercase();
    let long = format!("{SHA}0");
    let non_hex = format!("g{}", &SHA[1..]);
    for (sha, format) in
        [(upper.as_str(), 2), (&SHA[..63], 2), (long.as_str(), 2), (non_hex.as_str(), 2), ("", 2), (SHA, 1), (SHA, 3)]
    {
        let config = parse(config_file(sha, format)).unwrap();
        assert!(ConfidenceHead::<Cpu>::check(&config).is_err(), "sha {sha:?}, format {format}");
    }
}

/// Reference: the SHA-256 test vectors of FIPS 180-2 (empty input, "abc", one million "a", the last spanning many
/// read chunks), independent of the hashing code.
#[uzu_test]
fn file_sha256_matches_the_standard_vectors() {
    let directory = std::env::temp_dir().join(format!("confidence_head_sha256_{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let vectors: [(&str, Vec<u8>, &str); 3] = [
        ("empty", Vec::new(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ("abc", b"abc".to_vec(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        ("million", vec![b'a'; 1_000_000], "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
    ];
    for (name, bytes, digest) in vectors {
        let path: PathBuf = directory.join(name);
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(file_sha256(&path).unwrap(), digest, "{name}");
    }
    std::fs::remove_dir_all(&directory).unwrap();
}

/// Spec: the head predicts the target's sampled choice, so it is off under greedy decoding and under samplers greedy
/// in effect (top-k 1, temperature 0), and on for every other sampler.
#[uzu_test]
fn head_applies_only_under_samplers_that_sample() {
    let stochastic = |temperature: Option<f32>, top_k: Option<u32>| SamplingMethod::Stochastic {
        temperature,
        top_k,
        top_p: Some(0.95),
        min_p: None,
        repetition_penalty: None,
        suffix_repetition_length: None,
    };
    assert!(!ConfidenceHead::<Cpu>::applies_to(&SamplingMethod::Greedy));
    assert!(!ConfidenceHead::<Cpu>::applies_to(&stochastic(Some(1.0), Some(1))));
    assert!(!ConfidenceHead::<Cpu>::applies_to(&stochastic(Some(0.0), Some(20))));
    assert!(ConfidenceHead::<Cpu>::applies_to(&stochastic(Some(1.0), Some(20))));
    assert!(ConfidenceHead::<Cpu>::applies_to(&stochastic(None, None)));
    assert!(ConfidenceHead::<Cpu>::applies_to(&stochastic(Some(0.3), Some(2))));
}
