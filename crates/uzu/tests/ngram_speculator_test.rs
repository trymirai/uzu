use std::io::Write;

use uzu::speculators::{ngram_speculator::NgramSpeculator, speculator::Speculator};

// ── helpers ──────────────────────────────────────────────────────────────────

/// Serialize a minimal `model.bin` into bytes.
///
/// Binary layout (all little-endian):
/// ```text
/// [header] u32 hashtable_size, u32 ngram_k, u32 ngram_n, u32 _pad
/// [keys]   u32[hashtable_size × ngram_k]
/// [values] f32[hashtable_size × ngram_k]
/// [counts] u32[hashtable_size]
/// ```
fn build_model_bin(
    hashtable_size: usize,
    ngram_k: usize,
    ngram_n: usize,
    keys: &[u32],
    values: &[f32],
    counts: &[u32],
) -> Vec<u8> {
    let mut buf = Vec::new();
    // Header
    buf.extend_from_slice(&(hashtable_size as u32).to_le_bytes());
    buf.extend_from_slice(&(ngram_k as u32).to_le_bytes());
    buf.extend_from_slice(&(ngram_n as u32).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // _pad
    // Keys
    for &k in keys {
        buf.extend_from_slice(&k.to_le_bytes());
    }
    // Values
    for &v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    // Counts
    for &c in counts {
        buf.extend_from_slice(&c.to_le_bytes());
    }
    buf
}

/// Write bytes to a temp file and return the path.
fn write_temp_bin(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(bytes).expect("write");
    f
}

// ── load tests ───────────────────────────────────────────────────────────────

#[test]
fn test_load_empty_bytes_fails() {
    let f = write_temp_bin(&[]);
    assert!(NgramSpeculator::load(f.path()).is_err());
}

#[test]
fn test_load_header_only_too_short_fails() {
    // Header claims H=1, K=1 but actual data bytes are missing.
    let bytes = build_model_bin(1, 1, 2, &[], &[], &[]);
    // Truncate to just the header (16 bytes).
    let f = write_temp_bin(&bytes[..16]);
    assert!(NgramSpeculator::load(f.path()).is_err());
}

#[test]
fn test_load_minimal_valid() {
    // H=1, K=1, N=2: one slot, one candidate token.
    let bytes = build_model_bin(1, 1, 2, &[42], &[1.0], &[1]);
    let f = write_temp_bin(&bytes);
    assert!(NgramSpeculator::load(f.path()).is_ok());
}

#[test]
fn test_load_multi_slot() {
    let h = 8;
    let k = 3;
    let keys: Vec<u32> = (0..(h * k) as u32).collect();
    let values: Vec<f32> = vec![0.5; h * k];
    let counts: Vec<u32> = vec![k as u32; h];
    let bytes = build_model_bin(h, k, 2, &keys, &values, &counts);
    let f = write_temp_bin(&bytes);
    assert!(NgramSpeculator::load(f.path()).is_ok());
}

// ── speculate tests ───────────────────────────────────────────────────────────

/// H=1 means all tokens hash to slot 0 regardless of value — simple to test.
fn single_slot_speculator(
    keys: &[u32],
    values: &[f32],
    count: u32,
) -> NgramSpeculator {
    let k = keys.len();
    let bytes = build_model_bin(1, k, 2, keys, values, &[count]);
    let f = write_temp_bin(&bytes);
    NgramSpeculator::load(f.path()).expect("load single-slot speculator")
}

#[test]
fn test_speculate_empty_prefix_returns_empty() {
    let spec = single_slot_speculator(&[7, 13], &[0.8, 0.2], 2);
    assert!(spec.speculate(&[]).is_empty());
}

#[test]
fn test_speculate_hit_returns_candidates() {
    let spec = single_slot_speculator(&[7, 13], &[0.8, 0.2], 2);
    let result = spec.speculate(&[42]);
    assert_eq!(result.len(), 2);
    assert!((result[&7] - 0.8).abs() < 1e-6);
    assert!((result[&13] - 0.2).abs() < 1e-6);
}

#[test]
fn test_speculate_zero_count_returns_empty() {
    // count=0 means no valid entries in the slot.
    let spec = single_slot_speculator(&[7, 13], &[0.8, 0.2], 0);
    assert!(spec.speculate(&[1]).is_empty());
}

#[test]
fn test_speculate_zero_probability_filtered() {
    // Entries with probability 0.0 must not appear in the result.
    let spec = single_slot_speculator(&[7, 13], &[0.8, 0.0], 2);
    let result = spec.speculate(&[1]);
    assert_eq!(result.len(), 1);
    assert!(result.contains_key(&7));
    assert!(!result.contains_key(&13));
}

#[test]
fn test_speculate_partial_count() {
    // k=3 candidates per slot but count=2 → only first 2 entries returned.
    let spec = single_slot_speculator(&[7, 13, 99], &[0.5, 0.4, 0.1], 2);
    let result = spec.speculate(&[1]);
    assert_eq!(result.len(), 2);
    assert!(result.contains_key(&7));
    assert!(result.contains_key(&13));
    assert!(!result.contains_key(&99));
}

#[test]
fn test_speculate_token_zero_as_valid_key() {
    // Token ID 0 is a valid token and must be returned.
    let spec = single_slot_speculator(&[0, 5], &[0.9, 0.1], 2);
    let result = spec.speculate(&[1]);
    assert!(result.contains_key(&0));
}

// ── determinism ───────────────────────────────────────────────────────────────

#[test]
fn test_speculate_is_deterministic() {
    let spec = single_slot_speculator(&[7, 13], &[0.8, 0.2], 2);
    let r1 = spec.speculate(&[100, 200, 300]);
    let r2 = spec.speculate(&[100, 200, 300]);
    assert_eq!(r1, r2);
}

#[test]
fn test_different_prefixes_may_collide_with_h1() {
    // With H=1 every prefix maps to the same slot → all return the same candidates.
    let spec = single_slot_speculator(&[42], &[1.0], 1);
    let r1 = spec.speculate(&[1]);
    let r2 = spec.speculate(&[999]);
    assert_eq!(r1, r2);
}

// ── ngram_n = 1 (context_len = 0) ────────────────────────────────────────────

#[test]
fn test_ngram_n1_ignores_prefix_content() {
    // ngram_n=1 → context_len=0 → hash is always xxh3_64([]) % H.
    // With H=1 this is trivially slot 0 for any prefix.
    let bytes = build_model_bin(1, 1, 1, &[55], &[1.0], &[1]);
    let f = write_temp_bin(&bytes);
    let spec = NgramSpeculator::load(f.path()).unwrap();

    let r_short = spec.speculate(&[1]);
    let r_long = spec.speculate(&[1, 2, 3, 4, 5]);
    assert_eq!(r_short, r_long);
}
