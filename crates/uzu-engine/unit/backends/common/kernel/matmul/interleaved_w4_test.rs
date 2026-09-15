use uzu_engine_macros::uzu_test;

use crate::backends::common::kernel::matmul::interleaved_w4;

#[uzu_test]
fn convert_interleaves_and_signs_w4_codes() {
    let mut ascending = [0x10, 0x32, 0x54, 0x76]; // 0,1,2,3,4,5,6,7
    interleaved_w4::convert(&mut ascending, 4, true);
    assert_eq!(ascending, [0x40, 0x51, 0x62, 0x73]); // 0,4,1,5,2,6,3,7

    let mut unsigned_zeros = [0x88; 4];
    interleaved_w4::convert(&mut unsigned_zeros, 4, false);
    assert_eq!(unsigned_zeros, [0; 4]);
}
