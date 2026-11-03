#[inline(always)]
pub unsafe fn read_packed_code(
    weights_words: *const u32,
    linear_index: usize,
    bits: usize,
) -> u8 {
    let pack_factor = 32 / bits;
    let word_index = linear_index / pack_factor;
    let code_index_in_word = linear_index % pack_factor;
    let bit_offset = code_index_in_word * bits;
    let word = unsafe { weights_words.add(word_index).read_unaligned() };
    let code_mask = (1u32 << bits) - 1;
    ((word >> bit_offset) & code_mask) as u8
}

#[inline(always)]
pub unsafe fn read_zero_point(
    zp_ptr: *const u8,
    zp_index: usize,
    bits: usize,
) -> f32 {
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

#[inline(always)]
pub unsafe fn unpack_code(
    weights_words: *const u32,
    linear_index: usize,
    bits: usize,
    signed_codes: bool,
) -> f32 {
    let mut code = unsafe { read_packed_code(weights_words, linear_index, bits) };
    if signed_codes {
        code ^= 1u8 << (bits - 1);
    }
    f32::from(code)
}

#[inline(always)]
pub unsafe fn dequantize_weight(
    weights_words: *const u32,
    linear_index: usize,
    bits: usize,
    signed_codes: bool,
    scale: f32,
    zero_point: Option<f32>,
) -> f32 {
    let quantized_value = unsafe { unpack_code(weights_words, linear_index, bits, signed_codes) };
    let zero_point_offset = if let Some(zp) = zero_point {
        -scale * zp
    } else {
        -scale * (1u32 << (bits - 1)) as f32
    };
    scale * quantized_value + zero_point_offset
}

#[inline(always)]
pub unsafe fn dequantize_weight_with_bias(
    weights_words: *const u32,
    linear_index: usize,
    bits: usize,
    signed_codes: bool,
    scale: f32,
    bias: f32,
) -> f32 {
    let quantized_value = unsafe { unpack_code(weights_words, linear_index, bits, signed_codes) };
    scale * quantized_value + bias
}
