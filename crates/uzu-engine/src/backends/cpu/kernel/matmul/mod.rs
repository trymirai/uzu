mod error;
mod kernel;
mod reference;

pub use error::MatmulCpuError;
pub use kernel::MatmulCpuKernel;

#[inline]
pub(super) unsafe fn unpack_code(
    weights_words: *const u32,
    linear_index: usize,
    bits: usize,
    signed_codes: bool,
) -> f32 {
    let pack_factor = u32::BITS as usize / bits;
    let word_index = linear_index / pack_factor;
    let code_index_in_word = linear_index % pack_factor;
    let bit_offset = code_index_in_word * bits;
    let word = unsafe { weights_words.add(word_index).read_unaligned() };
    let code_mask = (1u32 << bits) - 1;
    let mut code = ((word >> bit_offset) & code_mask) as u8;
    if signed_codes {
        code ^= 1u8 << (bits - 1);
    }
    f32::from(code)
}

#[inline]
pub(super) unsafe fn read_zero_point(
    zero_points: *const u8,
    zero_point_index: usize,
    bits: usize,
) -> f32 {
    if bits == 4 {
        let byte_value = unsafe { *zero_points.add(zero_point_index / 2) };
        if zero_point_index.is_multiple_of(2) {
            (byte_value & 0x0F) as f32
        } else {
            ((byte_value >> 4) & 0x0F) as f32
        }
    } else {
        unsafe { *zero_points.add(zero_point_index) as f32 }
    }
}
