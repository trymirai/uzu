use std::sync::Arc;

pub fn generate(
    name: &str,
    size: usize,
) -> Arc<[u8]> {
    let seed = name.bytes().fold(0u8, |accumulator, byte| accumulator.wrapping_add(byte));
    (0..size)
        .map(|byte_index| {
            let low_bits = byte_index as u8;
            low_bits.wrapping_mul(31).wrapping_add(seed).wrapping_add((byte_index / 7) as u8)
        })
        .collect::<Vec<_>>()
        .into()
}
