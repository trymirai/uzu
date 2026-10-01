use uzu_engine_macros::uzu_test;

use super::{MaskKind, SplitGeometry, choose_splits, should_encode};

fn geometry(head_dim: u32) -> SplitGeometry {
    let (num_q_heads, block_rows, num_groups) = if head_dim == 128 {
        (32, 64, 8)
    } else {
        (24, 32, 4)
    };
    SplitGeometry {
        head_dim,
        num_q_heads,
        num_groups,
        block_rows,
        block_k: 32,
    }
}

fn splits(
    cores: u32,
    head_dim: u32,
    suffix_length: u32,
    kv_length: u32,
) -> u32 {
    choose_splits(geometry(head_dim), suffix_length, kv_length, cores)
}

#[uzu_test]
fn measured_and_fallback_boundaries() {
    for (cores, head_dim, suffix, kv, expected) in [
        (40, 256, 2, 16_384, 18),
        (40, 256, 8, 16_384, 9),
        (40, 256, 12, 61_440, 6),
        (40, 256, 16, 5_120, 6),
        (40, 256, 32, 32_768, 10),
        (40, 128, 64, 32_768, 10),
        (40, 256, 17, 262_144, 10),
        (40, 256, 128, 16_383, 3),
        (40, 256, 16, 128, 4),
        (10, 256, 32, 262_144, 3),
        (10, 256, 128, 16_383, 1),
    ] {
        assert_eq!(splits(cores, head_dim, suffix, kv), expected);
    }
}

#[uzu_test]
fn should_encode_boundaries() {
    for (head_dim, mask, suffix_length, kv_length, expected) in [
        (256, MaskKind::Causal, 1, 150_001, false),
        (256, MaskKind::Causal, 2, 1_023, false),
        (256, MaskKind::Causal, 2, 1_024, true),
        (128, MaskKind::Trie, 15, 1_024, true),
        (128, MaskKind::Causal, 16, 1_024, true),
        (128, MaskKind::Causal, 32, 1_023, false),
        (256, MaskKind::Causal, 16, 1_023, false),
        (256, MaskKind::Causal, 8, 150_001, true),
        (256, MaskKind::Causal, 1_024, 1_024, true),
        (256, MaskKind::None, 1_024, 1_024, false),
        (256, MaskKind::Trie, 65, 1_024, false),
    ] {
        assert_eq!(should_encode(head_dim, mask, suffix_length, kv_length), expected);
    }
}

#[uzu_test]
fn long_prefill_split_selection_boundaries() {
    for (suffix, kv, expected) in [
        (1024, 16_383, 1),
        (64, 61_440, 5),
        (1024, 16_384, 8),
        (256, 61_440, 30),
        (1024, 20_480, 10),
        (1024, 102_400, 10),
    ] {
        assert_eq!(splits(40, 256, suffix, kv), expected, "suffix={suffix}, kv={kv}");
    }

    let many_heads = SplitGeometry {
        num_q_heads: 240,
        ..geometry(256)
    };
    assert_eq!(choose_splits(many_heads, 1024, 61_440, 40), 1, "fewer than two prefill splits fit");
}
