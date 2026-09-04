#[inline]
pub fn decode_e2m1(code: u8) -> f32 {
    const VALUES: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    VALUES[usize::from(code & 0x0f)]
}

#[inline]
pub fn decode_e8m0(exponent: u8) -> f32 {
    match exponent {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        exponent => f32::from_bits(u32::from(exponent) << 23),
    }
}

#[inline]
pub fn decode_mxfp4(
    code: u8,
    exponent: u8,
    outer_scale: f32,
) -> f32 {
    decode_e2m1(code) * decode_e8m0(exponent) * outer_scale
}
