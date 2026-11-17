use uzu_engine_macros::uzu_test;

use crate::{
    config::stream_op::ngram_injection::NGramHashConfig,
    encodable_block::stream_op::ngram_injection::{hash, num_heads},
};

const EOS: u32 = 7;
const NGRAM_SIZE: u32 = 3;
const HEADS_PER_NGRAM: u32 = 2;
const MULTIPLIERS: [u64; 3] = [27696880803969, 9426225827651, 20661530663711];
const HEAD_VOCAB_SIZES: [u32; 4] = [1031, 1033, 1039, 1049];
const TOKENS: [u32; 11] = [151935, 42, 9001, EOS, 5, 151000, 77, 3, EOS, EOS, 1234];
// lalamo `NGramHashConfig.ngram_ids` with an all-EOS history, vocab 151936, PLE layer 1, seed 1234.
const EXPECTED_IDS: [[u32; 4]; 11] = [
    [802, 1111, 2883, 3854],
    [900, 1477, 3035, 3350],
    [356, 1160, 2207, 3447],
    [902, 1892, 2187, 3904],
    [388, 1383, 2988, 3696],
    [217, 1520, 2805, 3272],
    [607, 1929, 2852, 3738],
    [331, 1469, 2076, 3870],
    [974, 1253, 2177, 3660],
    [625, 1870, 2996, 4100],
    [612, 1931, 2251, 3353],
];

fn hasher() -> NGramHashConfig {
    NGramHashConfig {
        multipliers: MULTIPLIERS.into(),
        head_vocab_sizes: HEAD_VOCAB_SIZES.into(),
        ngram_size: NGRAM_SIZE,
        heads_per_ngram: HEADS_PER_NGRAM,
        eos_token_id: EOS,
        token_vocab_size: 69420,
    }
}

#[uzu_test]
fn test_ngram_hasher_matches_lalamo() {
    let num_heads = num_heads(&hasher());
    let mut ids = vec![0; TOKENS.len() * num_heads as usize];
    hash(&hasher(), &[], &TOKENS, None, &mut ids);
    assert_eq!(ids, EXPECTED_IDS.iter().flatten().copied().collect::<Vec<_>>());
}

#[uzu_test]
fn test_ngram_hasher_history_matches_full_sequence() {
    let num_heads = num_heads(&hasher());
    let split = 5;
    let mut ids = vec![0; (TOKENS.len() - split) * num_heads as usize];
    hash(&hasher(), &TOKENS[..split], &TOKENS[split..], None, &mut ids);
    assert_eq!(ids, EXPECTED_IDS[split..].iter().flatten().copied().collect::<Vec<_>>());
}

#[uzu_test]
fn test_ngram_hasher_follows_parents() {
    let num_heads = num_heads(&hasher()) as usize;
    // A trie with two branches below the first token: the flat sequence and a different one.
    let tokens = [TOKENS[0], TOKENS[1], 4242, TOKENS[2], 4343];
    let parents = [-1, 0, 0, 1, 2];
    let mut ids = vec![0; tokens.len() * num_heads];
    hash(&hasher(), &[], &tokens, Some(&parents), &mut ids);
    let node = |index: usize| &ids[index * num_heads..(index + 1) * num_heads];

    assert_eq!(node(0), EXPECTED_IDS[0]);
    assert_eq!(node(1), EXPECTED_IDS[1]);
    assert_eq!(node(3), EXPECTED_IDS[2]);

    let mut branch_ids = vec![0; 3 * num_heads];
    hash(&hasher(), &[], &[TOKENS[0], 4242, 4343], None, &mut branch_ids);
    assert_eq!(node(2), &branch_ids[num_heads..2 * num_heads]);
    assert_eq!(node(4), &branch_ids[2 * num_heads..]);
}
