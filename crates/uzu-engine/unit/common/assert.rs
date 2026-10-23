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

/// Each index where `actual` leaves `reference`'s tolerance, with the expected and actual values.
pub fn parity_mismatches<'a, T: DataTypeRepr + Float>(
    reference: &'a [T],
    actual: &'a [T],
    rel_tol: f64,
    abs_tol: f64,
) -> impl Iterator<Item = (usize, f32, f32)> + 'a {
    let shared = reference.iter().zip(actual).enumerate().filter_map(move |(i, (&exp, &got))| {
        let (exp, got) = (exp.to_f32().unwrap(), got.to_f32().unwrap());
        (!check_tolerance(exp, got, rel_tol, abs_tol)).then_some((i, exp, got))
    });
    let missing = (reference.len().min(actual.len())..reference.len().max(actual.len())).map(move |i| {
        (
            i,
            reference.get(i).map_or(f32::NAN, |value| value.to_f32().unwrap()),
            actual.get(i).map_or(f32::NAN, |value| value.to_f32().unwrap()),
        )
    });
    shared.chain(missing)
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
    for (i, exp_f32, got_f32) in parity_mismatches(reference, actual, rel_tol, abs_tol) {
        if errors < 5 {
            eprintln!("  {label} idx={i} expected={exp_f32} got={got_f32} diff={}", (exp_f32 - got_f32).abs());
        }
        errors += 1;
    }
    assert_eq!(errors, 0, "{label}: {errors} mismatches");
}
