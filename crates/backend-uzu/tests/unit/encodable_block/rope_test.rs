use super::build_inverse_frequencies;
use crate::config::{ConfigDataType, RoPEConfig, RopeConfigCommon};

#[test]
fn test_rope_inverse_frequency_precompute_supports_scaling_variants() {
    let unscaled = RoPEConfig::Unscaled(common());
    assert_close_slice(&build_inverse_frequencies(&unscaled, 4), &[1.0, 0.01]);

    let linear = RoPEConfig::LinearScalingRoPEConfig {
        common: common(),
        scaling_factor: 2.0,
    };
    assert_close_slice(&build_inverse_frequencies(&linear, 4), &[0.5, 0.005]);

    let llama = RoPEConfig::Llama {
        common: common(),
        scaling_factor: 8.0,
        original_context_length: 8,
        low_frequency_factor: 1.0,
        high_frequency_factor: 4.0,
    };
    assert_close_slice(&build_inverse_frequencies(&llama, 4), &[0.204694867, 0.00125]);

    let yarn = RoPEConfig::YARN {
        common: common(),
        scaling_factor: 4.0,
        original_context_length: 8,
        beta_fast: 32.0,
        beta_slow: 1.0,
        truncate: false,
    };
    assert_close_slice(&build_inverse_frequencies(&yarn, 4), &[1.0, 0.0025]);
}

fn common() -> RopeConfigCommon {
    RopeConfigCommon {
        precision: ConfigDataType::BFloat16,
        base: 10000.0,
        max_sequence_length: 8,
        head_dim: Some(4),
        partial_rotary_dim: None,
    }
}

fn assert_close_slice(
    actual: &[f32],
    expected: &[f32],
) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected) {
        assert!((*actual_value - *expected_value).abs() < 1e-6, "expected {expected_value}, got {actual_value}");
    }
}
