use uzu_engine_macros::kernel;

fn row_bytes<const BITS: u16>(columns: usize) -> usize {
    (columns * BITS as usize).div_ceil(8)
}

#[inline]
fn get_element<const BITS: u16>(
    data: &[u8],
    row_stride: usize,
    row: usize,
    column: usize,
) -> u16 {
    let byte = row * row_stride;
    match BITS {
        4 => {
            let packed = data[byte + column / 2];
            u16::from(if column % 2 == 0 {
                packed & 0xf
            } else {
                packed >> 4
            })
        },
        8 => u16::from(data[byte + column]),
        16 => u16::from_le_bytes([data[byte + column * 2], data[byte + column * 2 + 1]]),
        _ => unreachable!(),
    }
}

#[inline]
fn set_element<const BITS: u16>(
    data: &mut [u8],
    row_stride: usize,
    row: usize,
    column: usize,
    value: u16,
) {
    let byte = row * row_stride;
    match BITS {
        4 => {
            let slot = &mut data[byte + column / 2];
            let shift = (column % 2) * 4;
            *slot = (*slot & !(0xf << shift)) | (((value as u8) & 0xf) << shift);
        },
        8 => data[byte + column] = value as u8,
        16 => data[byte + column * 2..][..2].copy_from_slice(&value.to_le_bytes()),
        _ => unreachable!(),
    }
}

fn transpose_out_of_place<const BITS: u16>(
    source: &[u8],
    output: &mut [u8],
    rows: usize,
    columns: usize,
) {
    let source_stride = row_bytes::<BITS>(columns);
    let output_stride = row_bytes::<BITS>(rows);
    output.fill(0);
    for row in 0..rows {
        for column in 0..columns {
            let value = get_element::<BITS>(source, source_stride, row, column);
            set_element::<BITS>(output, output_stride, column, row, value);
        }
    }
}

fn transpose_in_place<const BITS: u16>(
    data: &mut [u8],
    size: usize,
) {
    let stride = row_bytes::<BITS>(size);
    for row in 0..size {
        for column in row + 1..size {
            let upper = get_element::<BITS>(data, stride, row, column);
            let lower = get_element::<BITS>(data, stride, column, row);
            set_element::<BITS>(data, stride, row, column, lower);
            set_element::<BITS>(data, stride, column, row, upper);
        }
    }
    if BITS == 4 && size % 2 != 0 {
        for row in 0..size {
            data[row * stride + size / 2] &= 0xf;
        }
    }
}

#[kernel(Transpose)]
#[variants(BITS, 4, 8, 16)]
pub fn transpose<const BITS: u16>(
    input: *mut u8,
    #[optional(!in_place)] output: Option<*mut u8>,
    rows: u32,
    cols: u32,
    #[specialize] in_place: bool,
) {
    let (rows, cols) = (rows as usize, cols as usize);
    let input_bytes = rows * row_bytes::<BITS>(cols);
    if in_place {
        let data = unsafe { std::slice::from_raw_parts_mut(input, input_bytes) };
        transpose_in_place::<BITS>(data, rows);
    } else {
        let output_bytes = cols * row_bytes::<BITS>(rows);
        let source = unsafe { std::slice::from_raw_parts(input, input_bytes) };
        let output = unsafe { std::slice::from_raw_parts_mut(output.expect("out-of-place output"), output_bytes) };
        transpose_out_of_place::<BITS>(source, output, rows, cols);
    }
}
