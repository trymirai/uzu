use uzu::speculators::{ngram_speculator::NGramSpeculator, speculator::Speculator};

/// Build a minimal valid KN binary with one table (bigram, hashtable_size=1, top_k=1).
fn build_minimal_kn_binary(ngram_n: u32, hashtable_size: u32, top_k: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    let max_order: u32 = 1;
    let discount: f32 = 0.1;
    buf.extend_from_slice(&max_order.to_le_bytes());
    buf.extend_from_slice(&discount.to_le_bytes());

    // Table header
    let mut table = Vec::new();
    let ngram_pad: u32 = u32::MAX;
    table.extend_from_slice(&hashtable_size.to_le_bytes());
    table.extend_from_slice(&top_k.to_le_bytes());
    table.extend_from_slice(&ngram_n.to_le_bytes());
    table.extend_from_slice(&ngram_pad.to_le_bytes());

    // Tags (u64 each)
    for _ in 0..hashtable_size {
        table.extend_from_slice(&0u64.to_le_bytes());
    }
    // Keys (u32 each)
    for i in 0..hashtable_size * top_k {
        table.extend_from_slice(&(i + 100).to_le_bytes());
    }
    // Values (f32 each)
    for _ in 0..hashtable_size * top_k {
        table.extend_from_slice(&(0.5f32).to_le_bytes());
    }
    // Counts (u32 each)
    for _ in 0..hashtable_size {
        table.extend_from_slice(&1u32.to_le_bytes());
    }
    // Continuation dist: length=0
    table.extend_from_slice(&0u32.to_le_bytes());

    // Write table_len + table
    let table_len = table.len() as u64;
    buf.extend_from_slice(&table_len.to_le_bytes());
    buf.extend_from_slice(&table);

    buf
}

#[test]
fn test_load_minimal_kn_binary() {
    let bin = build_minimal_kn_binary(2, 4, 2);
    let spec = NGramSpeculator::new(bin);
    let result = spec.speculate(&[42]);
    // Should return something (may or may not match depending on hash)
    assert!(result.len() <= 2);
}

#[test]
fn test_speculate_empty_prefix() {
    let bin = build_minimal_kn_binary(2, 4, 2);
    let spec = NGramSpeculator::new(bin);
    // Empty prefix should not panic
    let result = spec.speculate(&[]);
    assert!(result.len() <= 2);
}

#[test]
fn test_temperature_positive() {
    let bin = build_minimal_kn_binary(2, 4, 2);
    let spec = NGramSpeculator::new_with_temperature(bin, Some(0.5));
    let result = spec.speculate(&[42]);
    assert!(result.len() <= 2);
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn test_temperature_zero_panics() {
    let bin = build_minimal_kn_binary(2, 4, 2);
    NGramSpeculator::new_with_temperature(bin, Some(0.0));
}

#[test]
#[should_panic(expected = "temperature must be positive")]
fn test_temperature_negative_panics() {
    let bin = build_minimal_kn_binary(2, 4, 2);
    NGramSpeculator::new_with_temperature(bin, Some(-1.0));
}

#[test]
#[should_panic(expected = "hashtable_size must be > 0")]
fn test_hashtable_size_zero_panics() {
    let bin = build_minimal_kn_binary(2, 0, 1);
    NGramSpeculator::new(bin);
}

#[test]
#[should_panic(expected = "top_k must be > 0")]
fn test_top_k_zero_panics() {
    let bin = build_minimal_kn_binary(2, 4, 0);
    NGramSpeculator::new(bin);
}

#[test]
#[should_panic(expected = "ngram_n must be >= 1")]
fn test_ngram_n_zero_panics() {
    let bin = build_minimal_kn_binary(0, 4, 2);
    NGramSpeculator::new(bin);
}

#[test]
#[should_panic]
fn test_truncated_file_panics() {
    let mut bin = build_minimal_kn_binary(2, 4, 2);
    bin.truncate(bin.len() - 10);
    NGramSpeculator::new(bin);
}
