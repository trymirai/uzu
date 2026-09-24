use uzu_engine_macros::uzu_test;

use super::{decode_e2m1, decode_e8m0};

#[uzu_test]
fn decodes_e2m1_and_e8m0_edges() {
    let expected = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

    for (code, value) in expected.into_iter().enumerate() {
        assert_eq!(decode_e2m1(code as u8), value);
        assert_eq!(decode_e2m1(code as u8 | 0b1000), -value);
    }
    assert_eq!(decode_e2m1(0).to_bits(), 0.0f32.to_bits());
    assert_eq!(decode_e2m1(8).to_bits(), (-0.0f32).to_bits());
    assert_eq!(decode_e8m0(0).to_bits(), 0x0040_0000);
    assert_eq!(decode_e8m0(1), 2.0f32.powi(-126));
    assert_eq!(decode_e8m0(127), 1.0);
    assert_eq!(decode_e8m0(254), 2.0f32.powi(127));
    assert!(decode_e8m0(255).is_nan());
}
