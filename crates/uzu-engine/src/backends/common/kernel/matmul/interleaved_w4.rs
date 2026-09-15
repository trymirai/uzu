use super::parallel_rows;
use crate::backends::common::gpu_types::QuantizationMode;

const W4_BITS: u32 = 4;
const CODES_PER_WORD: usize = 8;
const WORD_BYTES: usize = CODES_PER_WORD * W4_BITS as usize / u8::BITS as usize;

pub const fn can_interleave(row_bytes: usize) -> bool {
    row_bytes.is_multiple_of(WORD_BYTES)
}

pub fn convert(
    codes: &mut [u8],
    row_bytes: usize,
    codes_are_signed: bool,
) {
    assert!(row_bytes > 0, "packed W4 rows must not be empty");
    assert!(can_interleave(row_bytes), "W4 rows must contain whole 8-code words");
    assert!(codes.len().is_multiple_of(row_bytes), "packed W4 allocation must contain whole rows");

    let sign_flip_mask = if codes_are_signed {
        0
    } else {
        QuantizationMode::U4.weight_codes_sign_flip_mask().expect("U4 sign-flip mask")
    };
    let sign_flip_bits = u32::from_le_bytes([sign_flip_mask; WORD_BYTES]);
    let entries = codes.len() * u8::BITS as usize / W4_BITS as usize;
    parallel_rows::for_each_block(codes, row_bytes, entries, |_, rows| {
        for bytes in rows.as_chunks_mut::<WORD_BYTES>().0 {
            let word = u32::from_le_bytes(*bytes) ^ sign_flip_bits;
            *bytes = interleave_nibbles(word).to_le_bytes();
        }
    });
}

#[inline(always)]
fn interleave_nibbles(word: u32) -> u32 {
    // [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 4, 1, 5, 2, 6, 3, 7].
    (word & 0xf000_000f)
        | ((word & 0x000f_0000) >> 12)
        | ((word & 0x0000_00f0) << 4)
        | ((word & 0x00f0_0000) >> 8)
        | ((word & 0x0000_0f00) << 8)
        | ((word & 0x0f00_0000) >> 4)
        | ((word & 0x0000_f000) << 12)
}
