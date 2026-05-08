use crate::DataType;

pub fn size_for_shape(
    shape: &[usize],
    data_type: DataType,
) -> usize {
    let Some(last_dim) = shape.last() else {
        return data_type.size_in_bytes();
    };

    let bits_per_row = last_dim * data_type.size_in_bits();
    let padded_bytes_per_row = bits_per_row.div_ceil(8);

    let num_rows: usize = shape.iter().rev().skip(1).product();

    num_rows * padded_bytes_per_row
}
