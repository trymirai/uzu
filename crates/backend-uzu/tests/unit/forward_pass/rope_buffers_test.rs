use super::build_rope_values;
use crate::{
    ConfigDataType,
    config::{RoPEConfig, RopeConfigCommon},
};

fn common_rope_config(partial_rotary_dim: Option<usize>) -> RopeConfigCommon {
    RopeConfigCommon {
        precision: ConfigDataType::Float32,
        base: 10000.0,
        max_sequence_length: 3,
        head_dim: Some(8),
        partial_rotary_dim,
    }
}

fn assert_rope_values_match_lalamo_fixture(
    rope_config: RoPEConfig,
    expected: &[(f32, f32)],
) {
    let actual = build_rope_values(&rope_config, 3, 8);
    assert_eq!(actual.len(), expected.len());

    for (index, ((actual_cosine, actual_sine), (expected_cosine, expected_sine))) in
        actual.iter().zip(expected.iter()).enumerate()
    {
        assert!(
            (actual_cosine - expected_cosine).abs() < 1.0e-6,
            "cosine mismatch at index {index}: expected {expected_cosine}, got {actual_cosine}",
        );
        assert!(
            (actual_sine - expected_sine).abs() < 1.0e-6,
            "sine mismatch at index {index}: expected {expected_sine}, got {actual_sine}",
        );
    }
}

#[test]
fn test_rope_buffers_config_generation_matches_lalamo_unscaled_partial_fixture() {
    assert_rope_values_match_lalamo_fixture(
        RoPEConfig::Unscaled(common_rope_config(Some(4))),
        &[
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.540302277, 0.841470957),
            (0.995004177, 0.0998334214),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.540302277, 0.841470957),
            (0.995004177, 0.0998334214),
            (1.0, 0.0),
            (1.0, 0.0),
            (-0.416146845, 0.909297407),
            (0.980066597, 0.198669329),
            (1.0, 0.0),
            (1.0, 0.0),
            (-0.416146845, 0.909297407),
            (0.980066597, 0.198669329),
            (1.0, 0.0),
            (1.0, 0.0),
        ],
    );
}

#[test]
fn test_rope_buffers_config_generation_matches_lalamo_linear_fixture() {
    assert_rope_values_match_lalamo_fixture(
        RoPEConfig::LinearScalingRoPEConfig {
            common: common_rope_config(None),
            scaling_factor: 2.0,
        },
        &[
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.87758255, 0.47942555),
            (0.998750269, 0.0499791689),
            (0.999987483, 0.00499997893),
            (0.999999881, 0.000500000024),
            (0.87758255, 0.47942555),
            (0.998750269, 0.0499791689),
            (0.999987483, 0.00499997893),
            (0.999999881, 0.000500000024),
            (0.540302277, 0.841470957),
            (0.995004177, 0.0998334214),
            (0.999949992, 0.00999983307),
            (0.999999523, 0.000999999931),
            (0.540302277, 0.841470957),
            (0.995004177, 0.0998334214),
            (0.999949992, 0.00999983307),
            (0.999999523, 0.000999999931),
        ],
    );
}

#[test]
fn test_rope_buffers_config_generation_matches_lalamo_llama_fixture() {
    assert_rope_values_match_lalamo_fixture(
        RoPEConfig::Llama {
            common: common_rope_config(None),
            scaling_factor: 8.0,
            original_context_length: 8192,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
        },
        &[
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.540302277, 0.841470957),
            (0.995004177, 0.0998334214),
            (0.999949992, 0.00999983307),
            (1.0, 0.000213607578),
            (0.540302277, 0.841470957),
            (0.995004177, 0.0998334214),
            (0.999949992, 0.00999983307),
            (1.0, 0.000213607578),
            (-0.416146845, 0.909297407),
            (0.980066597, 0.198669329),
            (0.999800026, 0.0199986659),
            (0.999999881, 0.000427215156),
            (-0.416146845, 0.909297407),
            (0.980066597, 0.198669329),
            (0.999800026, 0.0199986659),
            (0.999999881, 0.000427215156),
        ],
    );
}

#[test]
fn test_rope_buffers_config_generation_matches_lalamo_yarn_fixture() {
    assert_rope_values_match_lalamo_fixture(
        RoPEConfig::YARN {
            common: common_rope_config(None),
            scaling_factor: 4.0,
            original_context_length: 8192,
            beta_fast: 32.0,
            beta_slow: 1.0,
            truncate: false,
        },
        &[
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (1.13862944, 0.0),
            (0.615204096, 0.958123624),
            (1.13294101, 0.11367327),
            (1.13859248, 0.00917380489),
            (1.13862932, 0.000350023736),
            (0.615204096, 0.958123624),
            (1.13294101, 0.11367327),
            (1.13859248, 0.00917380489),
            (1.13862932, 0.000350023736),
            (-0.473837048, 1.03535283),
            (1.1159327, 0.226210743),
            (1.13848162, 0.0183470156),
            (1.1386292, 0.000700047414),
            (-0.473837048, 1.03535283),
            (1.1159327, 0.226210743),
            (1.13848162, 0.0183470156),
            (1.1386292, 0.000700047414),
        ],
    );
}
