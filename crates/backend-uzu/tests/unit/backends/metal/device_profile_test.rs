use proc_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_profile_detection() {
    let cases = [
        ("Apple M4 Max", 40, true, true, false, DeviceIdentity::Other, GpuFamily::Apple9, DeviceSize::Large),
        ("Apple M5 Max", 40, true, true, true, DeviceIdentity::M5Max, GpuFamily::M5Plus, DeviceSize::Large),
        ("Apple M3 Pro", 20, true, true, false, DeviceIdentity::Other, GpuFamily::Apple9, DeviceSize::Small),
        ("Apple M2", 10, true, false, false, DeviceIdentity::M2, GpuFamily::Apple8, DeviceSize::Small),
        ("Apple M1", 8, false, false, false, DeviceIdentity::M1, GpuFamily::Legacy, DeviceSize::Small),
    ];
    for (name, cores, apple8, apple9, mxu, identity, family, size) in cases {
        let profile = classify_device(name, cores, apple8, apple9, mxu);
        assert_eq!(profile.identity(), identity);
        assert_eq!(profile.gpu_family(), family);
        assert_eq!(profile.size(), size);
        assert_eq!(profile.supports_mxu(), mxu);
    }
}

#[uzu_test]
fn family_detection_does_not_use_mxu_as_generation_proxy() {
    let m5_without_mxu = classify_device("Apple M5", 12, true, true, false);
    let m4_with_mxu = classify_device("Apple M4", 10, true, true, true);

    assert_eq!(m5_without_mxu.gpu_family(), GpuFamily::M5Plus);
    assert!(!m5_without_mxu.supports_mxu());
    assert_eq!(m4_with_mxu.gpu_family(), GpuFamily::Apple9);
    assert!(m4_with_mxu.supports_mxu());
}

#[uzu_test]
fn measured_route_identities_remain_distinct() {
    let cases = [
        ("Apple M1", DeviceIdentity::M1),
        ("Apple M2", DeviceIdentity::M2),
        ("Apple M2 Pro", DeviceIdentity::M2Pro),
        ("Apple M3 Max", DeviceIdentity::M3Max),
        ("Apple M4", DeviceIdentity::M4),
        ("Apple M4 Pro", DeviceIdentity::M4Pro),
        ("Apple M5 Max", DeviceIdentity::M5Max),
    ];
    for (name, expected) in cases {
        assert_eq!(classify_device(name, 16, true, true, false).identity(), expected);
    }
}

#[uzu_test]
fn unmeasured_identity_does_not_inherit_a_tuned_route() {
    assert_eq!(classify_device("Apple M1 Ultra", 64, false, false, false).identity(), DeviceIdentity::Other);
    assert_eq!(classify_device("Apple M6", 12, true, true, true).identity(), DeviceIdentity::Other);
}

#[uzu_test]
fn core_count_is_reduced_to_device_size() {
    assert_eq!(classify_device("Apple M1 Max", 24, false, false, false).size(), DeviceSize::Small);
    assert_eq!(classify_device("Apple M1 Max", 32, false, false, false).size(), DeviceSize::Large);
}
