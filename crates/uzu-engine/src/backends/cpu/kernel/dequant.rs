/// Shared CPU dequantization helpers for packed integer weights (4-bit and 8-bit).

/// Unpack a single quantized code from a 32-bit packed word array.
#[inline(always)]
pub unsafe fn read_packed_code(weights_words: *const u32, linear_index: usize, bits: usize) -> u8 {
    let pack_factor = 32 / bits;
    let word_index = linear_index / pack_factor;
    let code_index_in_word = linear_index % pack_factor;
    let bit_offset = code_index_in_word * bits;
    let word = unsafe { weights_words.add(word_index).read_unaligned() };
    let code_mask = (1u32 << bits) - 1;
    ((word >> bit_offset) & code_mask) as u8
}

/// Unpack a zero-point value for 4-bit (two zero points per byte) or 8-bit.
#[inline(always)]
pub unsafe fn read_zero_point(zp_ptr: *const u8, zp_index: usize, bits: usize) -> f32 {
    if bits == 4 {
        let byte_value = unsafe { *zp_ptr.add(zp_index / 2) };
        if zp_index.is_multiple_of(2) {
            (byte_value & 0x0F) as f32
        } else {
            ((byte_value >> 4) & 0x0F) as f32
        }
    } else {
        unsafe { *zp_ptr.add(zp_index) as f32 }
    }
}

/// Dequantize a packed weight element given its scale and optional zero-point.
#[inline(always)]
pub unsafe fn dequantize_weight(
    weights_words: *const u32,
    linear_index: usize,
    bits: usize,
    signed_codes: bool,
    scale: f32,
    zero_point: Option<f32>,
) -> f32 {
    let mut code = unsafe { read_packed_code(weights_words, linear_index, bits) };
    if signed_codes {
        code ^= 1u8 << (bits - 1);
    }
    let quantized_value = f32::from(code);
    let zero_point_offset = if let Some(zp) = zero_point {
        -scale * zp
    } else {
        -scale * (1u32 << (bits - 1)) as f32
    };
    scale * quantized_value + zero_point_offset
}

#[cfg(test)]
mod tests {
    use super::*;
    use uzu_engine_macros::uzu_test;

    #[uzu_test]
    fn test_read_packed_code_4bit() {
        let words = [0x76543210u32, 0xFEDCBA98u32];
        for i in 0..16 {
            let code = unsafe { read_packed_code(words.as_ptr(), i, 4) };
            assert_eq!(code, i as u8);
        }
    }

    #[uzu_test]
    fn test_read_zero_point_4bit() {
        let zp_bytes = [0xA3u8];
        assert_eq!(unsafe { read_zero_point(zp_bytes.as_ptr(), 0, 4) }, 3.0);
        assert_eq!(unsafe { read_zero_point(zp_bytes.as_ptr(), 1, 4) }, 10.0);
    }
}
