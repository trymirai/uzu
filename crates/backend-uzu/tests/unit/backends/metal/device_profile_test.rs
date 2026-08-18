use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_profile_detection() {
    assert_eq!(classify_device(40, true, true, false), DeviceProfile::new(DeviceSize::Large, DeviceGeneration::Apple9));
    assert_eq!(classify_device(40, true, true, true), DeviceProfile::new(DeviceSize::Large, DeviceGeneration::M5Plus));
    assert_eq!(classify_device(20, true, true, false), DeviceProfile::new(DeviceSize::Small, DeviceGeneration::Apple9));
    assert_eq!(
        classify_device(10, true, false, false),
        DeviceProfile::new(DeviceSize::Small, DeviceGeneration::Apple8)
    );
    assert_eq!(
        classify_device(8, false, false, false),
        DeviceProfile::new(DeviceSize::Small, DeviceGeneration::Legacy)
    );
}
