use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    backends::{
        common::{Backend, Encoder, Kernels, kernel::radix_top_k_small::RadixTopKSmall},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context},
};

const COLUMNS: usize = 1025;
const K: usize = 512;
const ROWS: usize = 3;

fn radix_top_k_small<B: Backend>(input: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let input = alloc_allocation_with_data::<B, f32>(&context, input);
    let mut ids = alloc_allocation::<B, u32>(&context, ROWS * K);
    let mut scores = alloc_allocation::<B, f32>(&context, ROWS * K);
    let kernel = <B::Kernels as Kernels>::RadixTopKSmall::new(&context, COLUMNS as u32).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(&input, &mut ids, &mut scores, ROWS as u32, K as u32, &mut encoder).unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    (allocation_to_vec(&ids), allocation_to_vec(&scores))
}

#[uzu_test]
fn radix_top_k_small_matches_cpu() {
    let mut input = (0..ROWS * COLUMNS).map(|index| ((index * 37 % 101) as f32).sin()).collect::<Vec<_>>();
    input[..4].copy_from_slice(&[f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0]);
    input[4..8].copy_from_slice(&[1.0, 1.0, f32::from_bits(0x7fc0_0001), f32::from_bits(0xffc0_0001)]);

    for_each_non_cpu_backend!(|B| {
        let expected = radix_top_k_small::<Cpu>(&input);
        let actual = radix_top_k_small::<B>(&input);
        assert_eq!(actual.0, expected.0);
        assert_eq!(
            actual.1.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            expected.1.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        );
    });
}
