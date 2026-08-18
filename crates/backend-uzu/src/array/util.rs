use crate::data_type::DataType;

pub fn size_for_shape(
    shape: &[u32],
    data_type: DataType,
) -> usize {
    let Some(last_dim) = shape.last() else {
        return data_type.size_in_bytes();
    };

    let bits_per_row = *last_dim as usize * data_type.size_in_bits();
    let padded_bytes_per_row = bits_per_row.div_ceil(8);

    let num_rows: usize = shape.iter().rev().skip(1).map(|&dim| dim as usize).product();

    num_rows * padded_bytes_per_row
}
