use uzu_engine_macros::uzu_test;

use super::{int8_simd_splits, int8_splits};

#[uzu_test]
fn int8_small_suffix_split_cap() {
    for (suffix, kv, expected) in
        [(1, 30_720, 16), (1, 102_400, 16), (4, 61_440, 16), (8, 61_440, 62), (16, 61_440, 62)]
    {
        assert_eq!(int8_splits(32, 256, suffix, kv), expected, "suffix={suffix}, kv={kv}");
    }
}

#[uzu_test]
fn int8_simd_split_occupancy() {
    for (prefix, cores, expected) in [
        (0, 32, 1),
        (4096, 32, 8),
        (8192, 32, 8),
        (16384, 32, 8),
        (32768, 32, 8),
        (131072, 32, 8),
        (32768, 10, 3),
        (32768, 0, 1),
    ] {
        assert_eq!(int8_simd_splits(256, 256, 8, prefix + 8, 8, 6, cores), expected);
    }
    assert_eq!(int8_simd_splits(32, 256, 1, 32769, 4, 2, 32), 32);
    assert_eq!(int8_simd_splits(16, 256, 1, 32769, 2, 2, 32), 64);
    assert_eq!(int8_simd_splits(256, 256, 128, 32768, 8, 6, 32), int8_splits(256, 256, 128, 32768));
}
