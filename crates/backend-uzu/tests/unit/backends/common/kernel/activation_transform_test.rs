use std::fmt::Debug;

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::common::{Backend, Context, Encoder, kernel::ActivationTransform},
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend},
};

const BLOCK_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
enum TransformOrder {
    Input,
    Output,
}

fn reference_transform(
    data: &[f64],
    factors: &[i32],
    channel_count: usize,
    order: TransformOrder,
) -> Vec<f64> {
    let batch_count = data.len() / channel_count;
    let normalization_factor = 1.0 / (BLOCK_SIZE as f64).sqrt();
    let mut result = data.to_vec();

    for batch_index in 0..batch_count {
        let batch_offset = batch_index * channel_count;
        for block_start in (0..channel_count).step_by(BLOCK_SIZE) {
            if matches!(order, TransformOrder::Input) {
                for lane in 0..BLOCK_SIZE {
                    let index = batch_offset + block_start + lane;
                    result[index] *= f64::from(factors[block_start + lane]);
                }
            }

            let mut stride = 1;
            while stride < BLOCK_SIZE {
                for pair_start in (0..BLOCK_SIZE).step_by(stride * 2) {
                    for offset in 0..stride {
                        let index_a = batch_offset + block_start + pair_start + offset;
                        let index_b = index_a + stride;
                        let sum = result[index_a] + result[index_b];
                        let difference = result[index_a] - result[index_b];
                        result[index_a] = sum;
                        result[index_b] = difference;
                    }
                }
                stride *= 2;
            }

            for lane in 0..BLOCK_SIZE {
                let index = batch_offset + block_start + lane;
                result[index] *= normalization_factor;
                if matches!(order, TransformOrder::Output) {
                    result[index] *= f64::from(factors[block_start + lane]);
                }
            }
        }
    }

    result
}

fn run<T: ArrayElement + Float, B: Backend>(
    data: &[T],
    factors: &[i32],
    channel_count: usize,
    order: TransformOrder,
) -> Vec<T> {
    let context = B::Context::new().expect("context");
    let kernel = match order {
        TransformOrder::Input => ActivationTransform::<B>::input_rht(context.as_ref(), T::data_type()),
        TransformOrder::Output => ActivationTransform::<B>::output_rht(context.as_ref(), T::data_type()),
    }
    .expect("activation transform");

    let input = alloc_allocation_with_data::<B, T>(context.as_ref(), data);
    let mut output = alloc_allocation::<B, T>(context.as_ref(), data.len());
    let factors = alloc_allocation_with_data::<B, i32>(context.as_ref(), factors);
    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode_fp(
        &input,
        &mut output,
        &factors,
        channel_count as u32,
        (data.len() / channel_count) as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&output)
}

fn check<T: ArrayElement + Float + Debug>(tolerance: f64) {
    for order in [TransformOrder::Input, TransformOrder::Output] {
        for (batch_count, channel_count) in [(1, 32), (1, 64), (1, 128), (4, 32), (4, 256), (2, 2048)] {
            let data_f64: Vec<f64> =
                (0..batch_count * channel_count).map(|index| ((index as f64) * 0.1).sin() * 2.0).collect();
            let factors: Vec<i32> = (0..channel_count)
                .map(|index| {
                    if index % 3 == 0 {
                        -1
                    } else {
                        1
                    }
                })
                .collect();
            let expected = reference_transform(&data_f64, &factors, channel_count, order);
            let data: Vec<T> = data_f64.iter().map(|&value| T::from(value).unwrap()).collect();

            for_each_backend!(|B| {
                let actual = run::<T, B>(&data, &factors, channel_count, order);
                for (index, (actual_value, &expected_value)) in actual.iter().zip(&expected).enumerate() {
                    let actual_value = actual_value.to_f64().unwrap();
                    let error = (actual_value - expected_value).abs();
                    assert!(
                        error <= (expected_value.abs() * tolerance).max(tolerance),
                        "{order:?} mismatch at {index} for batch={batch_count}, channels={channel_count}: \
                         actual={actual_value}, expected={expected_value}, error={error}"
                    );
                }
            });
        }
    }
}

#[uzu_test]
fn input_and_output_rht_f32() {
    check::<f32>(1e-4);
}

#[uzu_test]
fn input_and_output_rht_bf16() {
    check::<bf16>(0.1);
}
