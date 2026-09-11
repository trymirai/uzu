use uzu_engine_macros::uzu_test;

use super::unit_interval;

/// Monotonic in `word`, so the extremes are the only values that could leave (0, 1),
/// where the gumbel -ln(-ln u) is +inf (wins every argmax) or -inf (loses every one).
#[uzu_test]
fn unit_interval_stays_open() {
    assert_eq!(unit_interval(u32::MAX), 1.0 - f32::powi(2.0, -24));
    assert_eq!(unit_interval(0), f32::powi(2.0, -24));
    assert_eq!(unit_interval(u8::MAX as u32), f32::powi(2.0, -24));
}
