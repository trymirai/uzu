use uzu_engine_macros::uzu_test;

use crate::{
    array::size_for_shape,
    backends::{
        common::{kernel::Transpose, *},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
};

const GUARD: [u8; 8] = [0x5a; 8];

fn source(
    rows: usize,
    columns: usize,
    data_type: DataType,
) -> Vec<u8> {
    let stride = size_for_shape(&[1, columns as u32], data_type);
    let mut data = (0..rows * stride).map(|index| (index as u8).wrapping_mul(37).wrapping_add(11)).collect::<Vec<_>>();
    if data_type == DataType::U4 && !columns.is_multiple_of(2) {
        for row in data.chunks_mut(stride) {
            *row.last_mut().unwrap() |= 0xf0;
        }
    }
    data
}

fn run<B: Backend>(
    input_data: &[u8],
    data_type: DataType,
    rows: usize,
    columns: usize,
    in_place: bool,
) -> Vec<u8> {
    let context = B::Context::new().unwrap();
    let operation = Transpose::<B>::new(&context, data_type, in_place).unwrap();
    let input_bytes = size_for_shape(&[rows as u32, columns as u32], data_type);
    let output_bytes = size_for_shape(&[columns as u32, rows as u32], data_type);
    let mut input_host = input_data.to_vec();
    input_host.extend_from_slice(&GUARD);
    let mut input = alloc_allocation_with_data::<B, u8>(&context, &input_host);
    let mut output = (!in_place).then(|| {
        let mut host = vec![0xcc; output_bytes];
        host.extend_from_slice(&GUARD);
        alloc_allocation_with_data::<B, u8>(&context, &host)
    });
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    operation.encode(&mut input, output.as_mut(), rows as u32, columns as u32, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    let input_after = allocation_to_vec::<B, u8>(&input);
    if in_place {
        assert_eq!(&input_after[input_bytes..], GUARD.as_slice());
        input_after[..input_bytes].to_vec()
    } else {
        assert_eq!(input_after, input_host);
        let output_after = allocation_to_vec::<B, u8>(output.as_ref().unwrap());
        assert_eq!(&output_after[output_bytes..], GUARD.as_slice());
        output_after[..output_bytes].to_vec()
    }
}

fn check(
    data_type: DataType,
    rows: usize,
    columns: usize,
    in_place: bool,
) {
    let input = source(rows, columns, data_type);
    let cpu = run::<Cpu>(&input, data_type, rows, columns, in_place);
    if data_type == DataType::U4 && !rows.is_multiple_of(2) {
        let stride = size_for_shape(&[1, rows as u32], data_type);
        assert!(cpu.chunks(stride).all(|row| row[stride - 1] & 0xf0 == 0));
    }
    for_each_non_cpu_backend!(|B| assert_eq!(run::<B>(&input, data_type, rows, columns, in_place), cpu));
}

#[uzu_test]
fn transpose_backends_match_cpu() {
    for &(rows, columns) in &[(32, 40), (33, 35)] {
        for &data_type in &[DataType::U4, DataType::U8, DataType::BF16] {
            check(data_type, rows, columns, false);
        }
    }
    for &size in &[32, 33] {
        for &data_type in &[DataType::U4, DataType::U8, DataType::BF16] {
            check(data_type, size, size, true);
        }
    }
}
