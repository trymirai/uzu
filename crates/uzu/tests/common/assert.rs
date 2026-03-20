use std::fmt::Display;

use num_traits::Float;
use uzu::ArrayElement;

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

        let diff = (expected[i] - actual[i]).to_f32().unwrap().abs();
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
