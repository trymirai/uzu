use std::fmt::Display;

use num_traits::Float;

use crate::data_type::DataTypeRepr;

pub fn assert_eq_float<T: DataTypeRepr + Float + Display>(
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

pub fn assert_parity<T: DataTypeRepr + Float>(
    label: &str,
    reference: &[T],
    actual: &[T],
    rel_tol: f64,
    abs_tol: f64,
) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    for (i, (&expected, &actual)) in reference.iter().zip(actual).enumerate() {
        let expected = expected.to_f32().unwrap();
        let actual = actual.to_f32().unwrap();
        let diff = f64::from((expected - actual).abs());
        let tolerance = abs_tol.max(f64::from(expected.abs()) * rel_tol);
        assert!(diff <= tolerance, "{label}[{i}]: expected {expected}, got {actual}, diff {diff} > {tolerance}");
    }
}
