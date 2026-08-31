use uzu_engine_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_profile_detection() {
    let cases = [
        ("Apple A18 Pro", 6, false, DeviceIdentity::A18Pro, GpuFamily::Apple9, DeviceSize::Small),
        ("Apple M3 Ultra", 80, false, DeviceIdentity::M3Ultra, GpuFamily::Apple9, DeviceSize::Large),
        ("Apple M4 Max", 40, false, DeviceIdentity::M4Max, GpuFamily::Apple9, DeviceSize::Large),
        ("Apple M5 Max", 40, true, DeviceIdentity::M5Max, GpuFamily::M5Plus, DeviceSize::Large),
        ("Apple M5 Ultra", 80, true, DeviceIdentity::M5Ultra, GpuFamily::M5Plus, DeviceSize::Large),
        ("Apple M3 Pro", 20, false, DeviceIdentity::M3Pro, GpuFamily::Apple9, DeviceSize::Small),
        ("Apple M2", 10, false, DeviceIdentity::M2, GpuFamily::Apple8, DeviceSize::Small),
        ("Apple M1", 8, false, DeviceIdentity::M1, GpuFamily::Legacy, DeviceSize::Small),
    ];
    for (name, cores, mxu, identity, family, size) in cases {
        let profile = classify_device(name, cores, mxu);
        assert_eq!(profile.identity(), identity);
        assert_eq!(profile.gpu_family(), family);
        assert_eq!(profile.size(), size);
        assert_eq!(profile.supports_mxu(), mxu);
    }
}

#[uzu_test]
fn family_detection_does_not_use_mxu_as_generation_proxy() {
    let m5_without_mxu = classify_device("Apple M5", 12, false);
    let m4_with_mxu = classify_device("Apple M4", 10, true);

    assert_eq!(m5_without_mxu.gpu_family(), GpuFamily::M5Plus);
    assert!(!m5_without_mxu.supports_mxu());
    assert_eq!(m4_with_mxu.gpu_family(), GpuFamily::Apple9);
    assert!(m4_with_mxu.supports_mxu());
}

#[uzu_test]
fn phone_device_profiles_follow_apple_gpu_families() {
    let cases = [
        ("Apple A14 GPU", DeviceIdentity::A14, GpuFamily::Legacy),
        ("Apple A15 GPU", DeviceIdentity::A15, GpuFamily::Apple8),
        ("Apple A16 GPU", DeviceIdentity::A16, GpuFamily::Apple8),
        ("Apple A17 Pro GPU", DeviceIdentity::A17Pro, GpuFamily::Apple9),
        ("Apple A18 GPU", DeviceIdentity::A18, GpuFamily::Apple9),
        ("Apple A18 Pro GPU", DeviceIdentity::A18Pro, GpuFamily::Apple9),
        ("Apple A19 GPU", DeviceIdentity::A19, GpuFamily::M5Plus),
        ("Apple A19 Pro GPU", DeviceIdentity::A19Pro, GpuFamily::M5Plus),
    ];
    for (name, identity, family) in cases {
        let profile = classify_device(name, 6, false);
        assert_eq!(profile.identity(), identity);
        assert_eq!(profile.gpu_family(), family);
    }
}

#[uzu_test]
fn mac_device_identities_remain_distinct() {
    let cases = [
        ("Apple M1", DeviceIdentity::M1),
        ("Apple M1 Pro", DeviceIdentity::M1Pro),
        ("Apple M1 Max", DeviceIdentity::M1Max),
        ("Apple M1 Ultra", DeviceIdentity::M1Ultra),
        ("Apple M2", DeviceIdentity::M2),
        ("Apple M2 Pro", DeviceIdentity::M2Pro),
        ("Apple M2 Max", DeviceIdentity::M2Max),
        ("Apple M2 Ultra", DeviceIdentity::M2Ultra),
        ("Apple M3", DeviceIdentity::M3),
        ("Apple M3 Pro", DeviceIdentity::M3Pro),
        ("Apple M3 Max", DeviceIdentity::M3Max),
        ("Apple M3 Ultra", DeviceIdentity::M3Ultra),
        ("Apple M4", DeviceIdentity::M4),
        ("Apple M4 Pro", DeviceIdentity::M4Pro),
        ("Apple M4 Max", DeviceIdentity::M4Max),
        ("Apple M5", DeviceIdentity::M5),
        ("Apple M5 Pro", DeviceIdentity::M5Pro),
        ("Apple M5 Max", DeviceIdentity::M5Max),
        ("Apple M5 Ultra", DeviceIdentity::M5Ultra),
    ];
    for (name, expected) in cases {
        assert_eq!(classify_device(name, 16, false).identity(), expected);
    }
}

#[uzu_test]
fn mobile_m_series_names_accept_the_gpu_suffix() {
    assert_eq!(classify_device("Apple M1 GPU", 8, false).identity(), DeviceIdentity::M1);
}

#[uzu_test]
fn untuned_devices_keep_their_identity_and_family() {
    let m1_ultra = classify_device("Apple M1 Ultra", 64, false);
    assert_eq!(m1_ultra.identity(), DeviceIdentity::M1Ultra);
    assert_eq!(m1_ultra.gpu_family(), GpuFamily::Legacy);
}

#[uzu_test]
fn core_count_is_reduced_to_device_size() {
    assert_eq!(classify_device("Apple M1 Max", 24, false).size(), DeviceSize::Small);
    assert_eq!(classify_device("Apple M1 Max", 32, false).size(), DeviceSize::Large);
}
