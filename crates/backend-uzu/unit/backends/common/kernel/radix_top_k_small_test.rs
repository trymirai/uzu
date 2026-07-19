#[cfg(metal_backend)]
use std::time::Instant;

use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

#[cfg(metal_backend)]
use crate::backends::metal::Metal;
use crate::{
    backends::{
        common::{Backend, Encoder, Kernels, kernel::radix_top_k_small::RadixTopKSmall},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context},
};

const TARGET_COLUMNS: usize = 248_320;
const TARGET_K: usize = 512;

fn values(
    rows: usize,
    columns: usize,
) -> Vec<f32> {
    (0..rows * columns).map(|index| ((index * 37 % 101) as f32).sin()).collect()
}

fn radix_top_k_small<B: Backend>(
    input: &[f32],
    rows: usize,
    columns: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let context = create_context::<B>();
    let input = alloc_allocation_with_data::<B, f32>(&context, input);
    let mut ids = alloc_allocation::<B, u32>(&context, rows * k);
    let mut scores = alloc_allocation::<B, f32>(&context, rows * k);
    let kernel = <B::Kernels as Kernels>::RadixTopKSmall::new(&context, columns as u32).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(&input, &mut ids, &mut scores, rows as u32, k as u32, &mut encoder).unwrap();
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    (allocation_to_vec(&ids), allocation_to_vec(&scores))
}

fn reference(
    input: &[f32],
    rows: usize,
    columns: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(rows * k);
    let mut scores = Vec::with_capacity(rows * k);
    for values in input.chunks_exact(columns) {
        let mut row = (0..columns).collect::<Vec<_>>();
        row.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]).then_with(|| left.cmp(&right)));
        for index in row.into_iter().take(k) {
            ids.push(index as u32);
            scores.push(values[index]);
        }
    }
    (ids, scores)
}

fn assert_output(
    actual: &(Vec<u32>, Vec<f32>),
    expected: &(Vec<u32>, Vec<f32>),
    shape: (usize, usize, usize),
) {
    assert_eq!(actual.0, expected.0, "shape={shape:?}");
    assert!(
        actual.1.iter().map(|value| value.to_bits()).eq(expected.1.iter().map(|value| value.to_bits())),
        "shape={shape:?}",
    );
}

#[uzu_test]
fn radix_top_k_small_matches_cpu() {
    for shape @ (rows, columns, k) in
        [(1, 1, 1), (1, 8, 8), (3, 1025, 1), (3, 1025, 511), (3, 1025, 512), (15, TARGET_COLUMNS, TARGET_K)]
    {
        let mut input = values(rows, columns);
        let special = [
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            1.0,
            1.0,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0xffc0_0001),
        ];
        let special_count = special.len().min(input.len());
        input[..special_count].copy_from_slice(&special[..special_count]);
        let expected = reference(&input, rows, columns, k);
        assert_output(&radix_top_k_small::<Cpu>(&input, rows, columns, k), &expected, shape);
        for_each_non_cpu_backend!(|B| {
            let actual = radix_top_k_small::<B>(&input, rows, columns, k);
            assert_output(&actual, &expected, shape);
            if columns == TARGET_COLUMNS {
                for _ in 0..2 {
                    assert_output(&radix_top_k_small::<B>(&input, rows, columns, k), &expected, shape);
                }
            }
        });
    }
}

#[cfg(metal_backend)]
#[uzu_test]
#[ignore = "benchmark"]
fn benchmark_radix_top_k_small() {
    const ROWS: usize = 15;
    const SAMPLES: usize = 50;
    const BATCH: u32 = 16;

    let context = create_context::<Metal>();
    let input = alloc_allocation_with_data::<Metal, f32>(&context, &values(ROWS, TARGET_COLUMNS));
    let mut ids = alloc_allocation::<Metal, u32>(&context, ROWS * TARGET_K);
    let mut scores = alloc_allocation::<Metal, f32>(&context, ROWS * TARGET_K);
    let kernel =
        <<Metal as Backend>::Kernels as Kernels>::RadixTopKSmall::new(&context, TARGET_COLUMNS as u32).unwrap();
    let mut run = || {
        let start = Instant::now();
        let mut encoder = Encoder::new(context.as_ref()).unwrap();
        for _ in 0..BATCH {
            kernel.encode(&input, &mut ids, &mut scores, ROWS as u32, TARGET_K as u32, &mut encoder).unwrap();
        }
        let completed = encoder.end_encoding().submit().wait_until_completed().unwrap();
        (completed.gpu_execution_time().div_f64(BATCH as f64), start.elapsed().div_f64(BATCH as f64))
    };
    for _ in 0..5 {
        run();
    }
    let (mut gpu, mut wall): (Vec<_>, Vec<_>) = (0..SAMPLES).map(|_| run()).unzip();
    gpu.sort_unstable();
    wall.sort_unstable();
    eprintln!("radix_top_k_small gpu={:?} wall={:?}", gpu[SAMPLES / 2], wall[SAMPLES / 2]);
}
