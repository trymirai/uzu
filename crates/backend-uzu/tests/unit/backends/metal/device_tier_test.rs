use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_tier_detection() {
    assert_eq!(device_tier_for(40, true, true), DeviceTier::Large);
    assert_eq!(device_tier_for(20, true, true), DeviceTier::SmallApple9);
    assert_eq!(device_tier_for(10, true, false), DeviceTier::SmallApple8);
    assert_eq!(device_tier_for(8, false, false), DeviceTier::SmallLegacy);
}
