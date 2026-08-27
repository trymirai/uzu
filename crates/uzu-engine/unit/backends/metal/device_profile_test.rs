use uzu_engine_macros::uzu_test;

use super::*;

#[uzu_test]
fn device_profile_uses_the_newest_supported_family() {
    for (apple8, apple9, apple10, expected) in [
        (false, false, false, GpuTuningTier::Legacy),
        (true, false, false, GpuTuningTier::Apple8),
        (true, true, false, GpuTuningTier::Apple9),
        (true, true, true, GpuTuningTier::Apple10),
    ] {
        let profile = classify_device(8, apple8, apple9, apple10, false);
        assert_eq!(profile.tuning_tier(), expected);
    }
}

#[uzu_test]
fn newest_family_wins_even_if_older_capability_results_are_missing() {
    let profile = classify_device(8, false, false, true, false);

    assert_eq!(profile.tuning_tier(), GpuTuningTier::Apple10);
}

#[uzu_test]
fn core_count_is_reduced_to_device_size() {
    assert_eq!(classify_device(29, true, true, false, false).size(), DeviceSize::Small);
    assert_eq!(classify_device(30, true, true, false, false).size(), DeviceSize::Large);
}

#[uzu_test]
fn mxu_support_is_independent_from_tuning_tier() {
    let legacy_with_mxu = classify_device(8, false, false, false, true);
    assert_eq!(legacy_with_mxu.tuning_tier(), GpuTuningTier::Legacy);
    assert!(legacy_with_mxu.supports_mxu());

    let apple10_without_mxu = classify_device(8, true, true, true, false);
    assert_eq!(apple10_without_mxu.tuning_tier(), GpuTuningTier::Apple10);
    assert_eq!(apple10_without_mxu.size(), DeviceSize::Small);
    assert!(!apple10_without_mxu.supports_mxu());
}
