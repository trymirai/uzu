use uzu_engine_macros::uzu_test;

use super::{MaskKind, SplitGeometry, choose_splits, int8_splits, select_head_dim_slices};

fn geometry(head_dim: u32) -> SplitGeometry {
    let (num_q_heads, query_rows_per_threadgroup, num_groups) = if head_dim == 128 {
        (32, 64, 8)
    } else {
        (24, 32, 4)
    };
    SplitGeometry {
        head_dim,
        num_q_heads,
        num_groups,
        query_rows_per_threadgroup,
        block_k: 32,
    }
}

#[uzu_test]
fn measured_and_fallback_boundaries() {
    for (cores, head_dim, suffix, kv, query_rows, use_mxu, expected) in [
        (40, 256, 2, 16_384, 32, true, 18),
        (40, 256, 8, 16_384, 32, true, 9),
        (40, 256, 12, 61_440, 32, true, 6),
        (40, 256, 16, 5_120, 32, true, 6),
        (40, 256, 32, 32_768, 32, true, 10),
        (40, 128, 64, 32_768, 64, true, 10),
        (40, 256, 17, 262_144, 32, true, 10),
        (40, 256, 128, 16_383, 32, true, 3),
        (40, 256, 16, 128, 32, true, 4),
        (10, 256, 32, 262_144, 32, true, 3),
        (10, 256, 128, 16_383, 32, true, 1),
        (10, 256, 1, 61_440, 8, false, 10),
    ] {
        let geometry = SplitGeometry {
            query_rows_per_threadgroup: query_rows,
            ..geometry(head_dim)
        };
        assert_eq!(choose_splits(geometry, suffix, kv, cores, use_mxu), expected);
    }
}

#[uzu_test]
fn should_encode_boundaries() {
    for (head_dim, mask, suffix, kv, use_mxu, expected) in [
        (256, MaskKind::Causal, 1, 0, true, None),
        (256, MaskKind::Causal, 2, 0, true, Some(4)),
        (128, MaskKind::Trie, 64, 0, true, Some(2)),
        (128, MaskKind::Causal, 65, 0, true, None),
        (256, MaskKind::Causal, 1_024, 0, true, Some(4)),
        (256, MaskKind::Causal, 1_025, 0, true, None),
        (256, MaskKind::None, 1_024, 0, true, None),
        (256, MaskKind::Trie, 65, 0, true, None),
        (256, MaskKind::Causal, 1, 16_383, false, None),
        (256, MaskKind::Causal, 1, 16_384, false, Some(8)),
        (256, MaskKind::Causal, 2, 16_384, false, Some(4)),
        (256, MaskKind::Causal, 4, 16_384, false, Some(2)),
        (128, MaskKind::Causal, 1, 16_384, false, Some(4)),
        (128, MaskKind::Causal, 512, 30_719, false, None),
        (128, MaskKind::Causal, 512, 30_720, false, Some(1)),
        (256, MaskKind::Causal, 3, 61_440, false, None),
        (256, MaskKind::None, 1, 61_440, false, None),
        (256, MaskKind::Causal, 512, 61_439, false, None),
        (256, MaskKind::Causal, 512, 61_440, false, Some(2)),
    ] {
        assert_eq!(select_head_dim_slices(head_dim, mask, suffix, kv, use_mxu), expected);
    }
}

#[uzu_test]
fn long_prefill_split_selection_boundaries() {
    for (suffix, kv, cores, use_mxu, expected) in [
        (1024, 16_383, 40, true, 1),
        (64, 61_440, 40, true, 5),
        (1024, 16_384, 40, true, 8),
        (256, 61_440, 40, true, 30),
        (1024, 20_480, 40, true, 10),
        (1024, 102_400, 40, true, 10),
        (512, 61_440, 10, false, 16),
    ] {
        assert_eq!(choose_splits(geometry(256), suffix, kv, cores, use_mxu), expected, "suffix={suffix}, kv={kv}");
    }

    let many_heads = SplitGeometry {
        num_q_heads: 240,
        ..geometry(256)
    };
    assert_eq!(choose_splits(many_heads, 1024, 61_440, 40, true), 1, "fewer than two prefill splits fit");
}

#[uzu_test]
fn int8_small_suffix_split_cap() {
    for (suffix, kv, expected) in
        [(1, 30_720, 16), (1, 102_400, 16), (4, 61_440, 16), (8, 61_440, 62), (16, 61_440, 62)]
    {
        assert_eq!(int8_splits(32, 256, suffix, kv), expected, "suffix={suffix}, kv={kv}");
    }
}
