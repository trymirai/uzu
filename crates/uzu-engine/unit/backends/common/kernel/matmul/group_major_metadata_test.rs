use uzu_engine_macros::uzu_test;

use crate::backends::common::kernel::matmul::group_major_metadata::{plane_bytes, row_stride, transpose};

#[uzu_test]
fn row_stride_matches_four_column_loads() {
    assert_eq!(row_stride(1), 4);
    assert_eq!(row_stride(4), 4);
    assert_eq!(row_stride(5), 8);
    assert_eq!(row_stride(12), 12);
}

fn check_value_transpose<T>(
    source: &[T],
    bits: u32,
) where
    T: bytemuck::Pod + Copy + Default,
{
    let columns = 5;
    let groups = 3;
    let source_bytes = bytemuck::cast_slice(source);
    let mut destination = vec![0u8; plane_bytes(columns, groups, bits)];
    destination[..source_bytes.len()].copy_from_slice(source_bytes);

    transpose(&mut destination, columns, groups, bits);

    let stride = row_stride(columns) as usize;
    let mut expected = vec![T::default(); stride * groups as usize];
    for group in 0..groups as usize {
        for column in 0..columns as usize {
            expected[group * stride + column] = source[column * groups as usize + group];
        }
    }
    assert_eq!(destination, bytemuck::cast_slice::<T, u8>(&expected));
}

#[uzu_test]
fn transpose_supports_value_planes() {
    check_value_transpose(&(0..15).map(|value| value as f32).collect::<Vec<_>>(), 32);
    check_value_transpose(&(0..15).map(|value| 0x1000 + value as u16).collect::<Vec<_>>(), 16);
    check_value_transpose(&[0u8, 3, 6, 9, 12, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14], 8);
}

#[uzu_test]
fn transpose_packs_u4_with_odd_group_count() {
    let source = [0x21, 0x03, 0x54, 0x06, 0x87, 0x09, 0xBA, 0x0C, 0xED, 0x0F];
    let mut destination = vec![0u8; plane_bytes(5, 3, 4)];
    destination[..source.len()].copy_from_slice(&source);

    transpose(&mut destination, 5, 3, 4);

    assert_eq!(destination, [0x41, 0xA7, 0x0D, 0, 0x52, 0xB8, 0x0E, 0, 0x63, 0xC9, 0x0F, 0]);
}
