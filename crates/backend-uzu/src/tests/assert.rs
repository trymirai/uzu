use std::fmt::Display;

use num_traits::Float;

use crate::array::ArrayElement;

pub fn assert_eq_float<T: ArrayElement + Float + Display>(
    expected: &[T],
    actual: &[T],
    eps: f32,
    msg: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "Slices size mismatch: expected {}, actual {}",
        expected.len(),
        actual.len()
    );

    for i in 0..expected.len() {
        if expected[i] == actual[i] {
            continue;
        }

        let expected_value = expected[i].to_f32().unwrap();
        let actual_value = actual[i].to_f32().unwrap();
        let diff = (expected_value - actual_value).abs();
        assert!(
            diff < eps,
            "{}. Mismatch at index {}: expected {}, got {}, diff {} (eps {})",
            msg,
            i,
            expected[i],
            actual[i],
            diff,
            eps
        );
    }
}

/// Absolute `eps` floored by `relative * |expected|`, for low-precision
/// dtypes whose output spacing exceeds a fixed absolute eps at larger
/// magnitudes. This is a relative tolerance, not an ULP comparison.
pub fn assert_eq_float_with_relative<T: ArrayElement + Float + Display>(
    expected: &[T],
    actual: &[T],
    eps: f32,
    relative: f32,
    msg: &str,
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "Slices size mismatch: expected {}, actual {}",
        expected.len(),
        actual.len()
    );

    for i in 0..expected.len() {
        if expected[i] == actual[i] {
            continue;
        }

        let expected_value = expected[i].to_f32().unwrap();
        let actual_value = actual[i].to_f32().unwrap();
        let expected_magnitude = expected_value.abs();
        let diff = (expected_value - actual_value).abs();
        let tolerance = eps.max(relative * expected_magnitude);
        assert!(
            diff < tolerance,
            "{}. Mismatch at index {}: expected {}, got {}, diff {} (eps {}, relative {})",
            msg,
            i,
            expected[i],
            actual[i],
            diff,
            eps,
            relative
        );
    }
}
