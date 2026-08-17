use proc_macros::uzu_test;

use super::*;
use crate::backends::metal::device_profile::DeviceGeneration;

const LARGE: DeviceProfile = DeviceProfile::new(40, DeviceGeneration::Apple9);
const SMALL_APPLE9: DeviceProfile = DeviceProfile::new(10, DeviceGeneration::Apple9);
const SMALL_APPLE8: DeviceProfile = DeviceProfile::new(10, DeviceGeneration::Apple8);
const SMALL_LEGACY: DeviceProfile = DeviceProfile::new(8, DeviceGeneration::Legacy);
const SMALL_M5: DeviceProfile = DeviceProfile::new(10, DeviceGeneration::M5Plus);
/// 32-core M1 Max: Legacy generation on a Large die, so the G13 rules must not fire.
const LARGE_LEGACY: DeviceProfile = DeviceProfile::new(32, DeviceGeneration::Legacy);

#[uzu_test]
fn fp_policy_cases() {
    #[rustfmt::skip]
    let cases = [
        (LARGE,        1, 12288, 1536, true,  tile(8, 8, 1)),
        (LARGE,        1, 12288, 1536, false, tile(8, 1, 1)),
        (LARGE,        1,  1536,  256, true,  tile(8, 1, 1)),
        (SMALL_APPLE9, 1,  1536,  256, true,  tile(8, 2, 1)),
        (LARGE,        8, 12288, 1536, true,  tile(8, 1, 4)),
        (LARGE,        8,     3,  128, true,  tile(8, 1, 1)),
        (LARGE,        1,  1536, 12288, true, tile(8, 8, 4)),
        (SMALL_LEGACY, 1, 262144, 1536, true, tile(8, 1, 4)),
        (LARGE_LEGACY, 1, 262144, 1536, true, tile(8, 8, 1)),
        (LARGE_LEGACY, 1,   6144, 1536, true, tile(8, 8, 1)),
    ];

    for (profile, m, n, k, aligned, expected) in cases {
        assert_eq!(fp_tile(m, n, k, aligned, profile), expected, "profile={profile:?} m={m} n={n} k={k}");
    }
}

#[uzu_test]
fn quant_policy_cases() {
    #[rustfmt::skip]
    let cases = [
        (LARGE,        1,    256, 1536, 4, false, qtile(2, 1)),
        (LARGE,        1, 262144, 1536, 4, false, DEFAULT_TILE),
        (SMALL_APPLE9, 1,   1536,  256, 4, false, qtile(4, 4)),
        (SMALL_APPLE8, 1,   2048, 1536, 4, false, qtile(8, 2)),
        (SMALL_LEGACY, 1,    256, 1536, 4, false, qtile(8, 2)),
        (SMALL_LEGACY, 1,   1536,  256, 4, false, qtile(4, 8)),
        (LARGE,        2,   2048, 1536, 4, false, DEFAULT_TILE),
        (LARGE,        1,   2048, 1536, 8, false, DEFAULT_TILE),
        (LARGE,        1,   2560, 9216, 4, true,  qtile(4, 8)),
        // Small M5 dies classified as SmallApple9 before the split; keep those rows.
        (SMALL_M5,     1,   1536,  256, 4, false, qtile(4, 4)),
        (SMALL_M5,     1,    256, 1536, 4, false, qtile(4, 2)),
    ];

    for (profile, m, n, k, bits, has_rht, expected) in cases {
        assert_eq!(
            quant_tile(m, n, k, bits, has_rht, profile),
            expected,
            "profile={profile:?} m={m} n={n} k={k} bits={bits}"
        );
    }
}
