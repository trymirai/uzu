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

fn check_tolerance(
    expected: f32,
    actual: f32,
    rel_tol: f64,
    abs_tol: f64,
) -> bool {
    let diff = (expected - actual).abs() as f64;
    let tol = abs_tol.max(expected.abs() as f64 * rel_tol);
    diff <= tol
}

pub fn assert_parity<T: DataTypeRepr + Float>(
    label: &str,
    reference: &[T],
    actual: &[T],
    rel_tol: f64,
    abs_tol: f64,
) {
    assert_eq!(reference.len(), actual.len(), "{label}: length mismatch");
    let mut errors = 0usize;
    for (i, (&exp, &got)) in reference.iter().zip(actual.iter()).enumerate() {
        let exp_f32 = exp.to_f32().unwrap();
        let got_f32 = got.to_f32().unwrap();
        if !check_tolerance(exp_f32, got_f32, rel_tol, abs_tol) {
            if errors < 5 {
                eprintln!("  {label} idx={i} expected={exp_f32} got={got_f32} diff={}", (exp_f32 - got_f32).abs());
            }
            errors += 1;
        }
    }
    assert_eq!(errors, 0, "{label}: {errors} mismatches");
}
