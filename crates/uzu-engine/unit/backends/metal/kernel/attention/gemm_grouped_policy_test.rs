use uzu_engine_macros::uzu_test;

use super::{MaskKind, choose_splits, should_encode};

fn splits(
    cores: u32,
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
) -> u32 {
    let (gqa, rows, groups) = if head_dim == 128 {
        (4, 64, 8)
    } else {
        (6, 32, 4)
    };
    let threadgroups = (gqa * suffix_length).div_ceil(rows) * groups;
    choose_splits(head_dim, suffix_length, kv_length, threadgroups, 32, cores)
}

#[uzu_test]
fn measured_and_fallback_boundaries() {
    for (cores, head_dim, suffix, kv, expected) in [
        (40, 256, 16, 5_120, 6),
        (40, 256, 32, 32_768, 10),
        (40, 128, 64, 32_768, 10),
        (40, 256, 17, 262_144, 10),
        (40, 256, 128, 262_144, 3),
        (40, 256, 16, 128, 4),
        (10, 256, 32, 262_144, 3),
        (10, 256, 128, 262_144, 1),
    ] {
        assert_eq!(splits(cores, head_dim, suffix, kv), expected);
    }
}

#[uzu_test]
fn should_encode_boundaries() {
    for (head_dim, mask, suffix_length, kv_length, expected) in [
        (128, MaskKind::Causal, 16, 1_024, true),
        (128, MaskKind::Causal, 32, 1_023, false),
        (256, MaskKind::Causal, 16, 1_023, false),
        (256, MaskKind::Causal, 8, 150_001, false),
        (256, MaskKind::Causal, 1_024, 1_024, true),
        (256, MaskKind::None, 1_024, 1_024, false),
        (256, MaskKind::Trie, 65, 1_024, false),
    ] {
        assert_eq!(should_encode(head_dim, mask, suffix_length, kv_length), expected);
    }
}
