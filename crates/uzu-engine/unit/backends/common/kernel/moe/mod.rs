mod combine_test;
mod gemv_test;
mod router_top_k_test;

use half::bf16;

fn assert_vec_close(
    actual: &[bf16],
    expected: &[bf16],
    tol: f32,
    desc: &str,
) {
    for (idx, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let (af, ef) = (a.to_f32(), e.to_f32());
        assert!((af - ef).abs() <= tol, "{desc} mismatch at idx={idx}: actual={af}, expected={ef}");
    }
}
