use crate::data_type::DataType;

pub fn size_for_shape(
    shape: &[u32],
    data_type: &DataType,
) -> usize {
    debug_assert!(
        shape.len() == 1 || data_type.size_in_bits() >= 8,
        "a sub-byte data type needs a 1-D shape, otherwise row padding is lost"
    );

    let Some(&last_dim) = shape.last() else {
        return data_type.size_in_bytes();
    };

    let padded_bytes_per_row = (last_dim as usize * data_type.size_in_bits()).div_ceil(8);

    shape
        .iter()
        .rev()
        .skip(1)
        .try_fold(padded_bytes_per_row, |bytes, &dim| bytes.checked_mul(dim as usize))
        .expect("tensor byte size exceeds the address space")
}
