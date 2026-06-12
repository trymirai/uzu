use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_tier_detection() {
    assert_eq!(device_tier_for(5, true, true), DeviceTier::Small); // A18 Pro
    assert_eq!(device_tier_for(8, false, false), DeviceTier::SmallG13); // M1
    assert_eq!(device_tier_for(10, true, false), DeviceTier::SmallG14); // M2
    assert_eq!(device_tier_for(19, true, false), DeviceTier::SmallG14); // M2 Pro
    assert_eq!(device_tier_for(20, true, true), DeviceTier::Small); // M4 Pro
    assert_eq!(device_tier_for(20, false, true), DeviceTier::Small); // Prefer newest supported family
    assert_eq!(device_tier_for(40, true, true), DeviceTier::Large); // M3/M4 Max
    assert_eq!(device_tier_for(32, false, false), DeviceTier::Large); // M1 Max stays large
}
