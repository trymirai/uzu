use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn fp_policy_cases() {
    #[rustfmt::skip]
    let cases = [
        (DeviceTier::Large,    1, 12288, 1536, true,  tile(8, 8, 1)),
        (DeviceTier::Large,    1, 12288,  512, true,  tile(8, 4, 1)),
        (DeviceTier::Large,    1, 12288,  128, true,  tile(8, 1, 1)),
        (DeviceTier::Large,    1, 12288, 1536, false, tile(8, 1, 1)),
        (DeviceTier::Large,    1,  1536,  256, true,  tile(8, 1, 1)),
        (DeviceTier::Small,    1,  1536,  256, true,  tile(8, 2, 1)),
        (DeviceTier::Large,    8, 12288, 1536, true,  tile(8, 1, 4)),
        (DeviceTier::Large,    8,  1536, 6144, true,  tile(8, 4, 4)),
        (DeviceTier::Large,    1,  1536, 12288, true, tile(8, 8, 4)),
        (DeviceTier::Small,    1,  1536, 12288, true, tile(8, 8, 1)),
        (DeviceTier::SmallG13, 1, 262144, 1536, true, tile(8, 1, 4)),
        (DeviceTier::SmallG13, 1,   5120, 1536, true, tile(8, 8, 1)),
    ];

    for (tier, m, n, k, aligned, expected) in cases {
        assert_eq!(fp_tile(m, n, k, aligned, tier), expected, "tier={tier:?} m={m} n={n} k={k}");
    }
}

#[uzu_test]
fn quant_policy_cases() {
    #[rustfmt::skip]
    let cases = [
        (DeviceTier::Large,    1,    256, 1536, 4, false, qtile(2, 1)),
        (DeviceTier::Large,    1,   2048, 1536, 4, false, qtile(2, 2)),
        (DeviceTier::Large,    1, 262144, 1536, 4, false, DEFAULT_TILE),
        (DeviceTier::Small,    1,   1536,  256, 4, false, qtile(4, 4)),
        (DeviceTier::SmallG14, 1,   2048, 1536, 4, false, qtile(8, 2)),
        (DeviceTier::SmallG13, 1,    256, 1536, 4, false, qtile(8, 2)),
        (DeviceTier::SmallG13, 1,   1536,  256, 4, false, qtile(4, 8)),
        (DeviceTier::Large,    2,   2048, 1536, 4, false, DEFAULT_TILE),
        (DeviceTier::Large,    1,   2048, 1536, 8, false, DEFAULT_TILE),
        (DeviceTier::Large,    1,   2560, 9216, 4, true,  qtile(4, 8)),
        (DeviceTier::Small,    1,   2560, 9216, 4, true,  DEFAULT_TILE),
    ];

    for (tier, m, n, k, bits, has_rht, expected) in cases {
        assert_eq!(quant_tile(m, n, k, bits, has_rht, tier), expected, "tier={tier:?} m={m} n={n} k={k} bits={bits}");
    }
}
