use backend_uzu_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_profile_detection() {
    assert_eq!(classify_device(40, true, true, false), DeviceProfile::new(40, DeviceGeneration::Apple9));
    assert_eq!(classify_device(40, true, true, true), DeviceProfile::new(40, DeviceGeneration::M5Plus));
    assert_eq!(classify_device(20, true, true, false), DeviceProfile::new(20, DeviceGeneration::Apple9));
    assert_eq!(classify_device(10, true, false, false), DeviceProfile::new(10, DeviceGeneration::Apple8));
    assert_eq!(classify_device(8, false, false, false), DeviceProfile::new(8, DeviceGeneration::Legacy));
}

#[uzu_test]
fn size_derives_from_core_count() {
    // M1 Max ships in two bins that straddle the cutoff, on identical silicon.
    assert_eq!(classify_device(24, false, false, false).size(), DeviceSize::Small);
    assert_eq!(classify_device(32, false, false, false).size(), DeviceSize::Large);
    assert_eq!(classify_device(8, false, false, false).gpu_core_count(), 8);
}
