#![cfg(metal_backend)]

use proc_macros::uzu_test;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rstest::rstest;

use super::ActivationsPrepareMetalKernel;
use crate::{
    backends::{
        common::{
            Context, Encoder,
            gpu_types::{ActivationPrepareOps, ActivationScaleStat},
            kernel::{ActivationsPrepareKernel, group_stat, quantize_symmetric_i8, symmetric_divisor},
        },
        metal::{Metal, MetalContext},
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
};

const RHT_BLOCK_SIZE: usize = 32;

fn hadamard(values: &mut [f32; RHT_BLOCK_SIZE]) {
    let mut stride = 1;
    while stride < RHT_BLOCK_SIZE {
        for lane in 0..RHT_BLOCK_SIZE {
            if lane & stride == 0 {
                let left = values[lane];
                let right = values[lane | stride];
                values[lane] = left + right;
                values[lane | stride] = left - right;
            }
        }
        stride <<= 1;
    }

    let scale = 1.0 / (RHT_BLOCK_SIZE as f32).sqrt();
    for value in values {
        *value *= scale;
    }
}

fn reference(
    input: &[f32],
    factors: &[i32],
    rows: usize,
    columns: usize,
    group_size: usize,
    stat: ActivationScaleStat,
) -> (Vec<i8>, Vec<f32>) {
    let groups = columns.div_ceil(group_size);
    let mut values = vec![0i8; rows * columns];
    let mut scales = vec![0.0f32; rows * groups];
    let mut prepared = vec![0.0f32; columns];

    for row in 0..rows {
        for block_start in (0..columns).step_by(RHT_BLOCK_SIZE) {
            let mut block = [0.0f32; RHT_BLOCK_SIZE];
            for lane in 0..RHT_BLOCK_SIZE {
                let column = block_start + lane;
                block[lane] = input[row * columns + column] * factors[column] as f32;
            }
            hadamard(&mut block);
            prepared[block_start..block_start + RHT_BLOCK_SIZE].copy_from_slice(&block);
        }

        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(columns);
            let divisor = symmetric_divisor(group_stat(&prepared[start..end], stat));
            scales[row * groups + group] = divisor;
            for column in start..end {
                values[row * columns + column] = quantize_symmetric_i8(prepared[column], divisor);
            }
        }
    }

    (values, scales)
}

fn run_metal(
    context: &MetalContext,
    input: &[f32],
    factors: &[i32],
    rows: usize,
    columns: usize,
    group_size: usize,
    stat: ActivationScaleStat,
) -> (Vec<i8>, Vec<f32>) {
    let groups = columns.div_ceil(group_size);
    let input = alloc_allocation_with_data::<Metal, f32>(context, input);
    let factors = alloc_allocation_with_data::<Metal, i32>(context, factors);
    let mut values = alloc_allocation::<Metal, i8>(context, rows * columns);
    let mut scales = alloc_allocation::<Metal, f32>(context, rows * groups);
    let ops = ActivationPrepareOps::INPUT_RHT | ActivationPrepareOps::QUANTIZE;
    let kernel = ActivationsPrepareMetalKernel::new(context, DataType::F32, ops, stat).expect("prepare kernel");
    let mut encoder = Encoder::<Metal>::new(context).expect("encoder");

    kernel.encode(
        &input,
        Some(&mut values),
        Some(&mut scales),
        Some(&factors),
        rows as u32,
        columns as u32,
        group_size as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec::<Metal, i8>(&values), allocation_to_vec::<Metal, f32>(&scales))
}

#[rstest]
#[test_attr(uzu_test)]
#[case::absmax_32(5, 96, 32, ActivationScaleStat::AbsMax)]
#[case::absmax_64(8, 256, 64, ActivationScaleStat::AbsMax)]
#[case::rms_128(4, 224, 128, ActivationScaleStat::Rms)]
fn rht_and_quantization_match_cpu(
    #[case] rows: usize,
    #[case] columns: usize,
    #[case] group_size: usize,
    #[case] stat: ActivationScaleStat,
) {
    let context = MetalContext::new().expect("Metal context");
    let mut rng = SmallRng::seed_from_u64(0x5EED_0001 ^ columns as u64);
    let input = (0..rows * columns).map(|_| rng.random_range(-1.0f32..1.0f32)).collect::<Vec<_>>();
    let factors = (0..columns)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect::<Vec<_>>();

    let (actual_values, actual_scales) = run_metal(&context, &input, &factors, rows, columns, group_size, stat);
    let (expected_values, expected_scales) = reference(&input, &factors, rows, columns, group_size, stat);

    for (index, (actual, expected)) in actual_scales.iter().zip(&expected_scales).enumerate() {
        let relative_error = (actual - expected).abs() / expected.abs().max(1e-6);
        assert!(relative_error < 1e-4, "scale {index}: {actual} != {expected}");
    }
    assert!(
        actual_values
            .iter()
            .zip(expected_values)
            .all(|(actual, expected)| (*actual as i32 - expected as i32).abs() <= 1),
        "prepared int8 values differ by more than one level",
    );
}
