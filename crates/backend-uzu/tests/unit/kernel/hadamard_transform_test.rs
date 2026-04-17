use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, kernel::HadamardTransformKernel},
};

use crate::uzu_test;

const BLOCK_SIZE: usize = 32;

fn reference_hadamard_transform_mul(
    data: &[f64],
    factors: &[f64],
    channel_count: usize,
) -> Vec<f64> {
    let batch_count = data.len() / channel_count;
    let normalization_factor = 1.0 / (BLOCK_SIZE as f64).sqrt();
    let mut result = data.to_vec();

    for batch_index in 0..batch_count {
        let batch_offset = batch_index * channel_count;

        for block_start in (0..channel_count).step_by(BLOCK_SIZE) {
            for lane in 0..BLOCK_SIZE {
                let index = batch_offset + block_start + lane;
                let factor_index = block_start + lane;
                result[index] *= factors[factor_index];
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
            }
        }
    }

    result
}

struct TestInput<T: ArrayElement + Float> {
    data: Box<[T]>,
    factors: Box<[i32]>,
    channel_count: usize,
    batch_count: usize,
}

fn generate_test_input<T: ArrayElement + Float>(
    batch_count: usize,
    channel_count: usize,
) -> (TestInput<T>, Vec<f64>) {
    let total_elements = batch_count * channel_count;

    let data_f64: Vec<f64> = (0..total_elements).map(|index| ((index as f64) * 0.1).sin() * 2.0).collect();

    let factors_i32: Vec<i32> = (0..channel_count)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();

    let factors_f64: Vec<f64> = factors_i32.iter().map(|&v| v as f64).collect();
    let expected = reference_hadamard_transform_mul(&data_f64, &factors_f64, channel_count);

    let data: Vec<T> = data_f64.iter().map(|&value| T::from(value).unwrap()).collect();

    let input = TestInput {
        data: data.into_boxed_slice(),
        factors: factors_i32.into_boxed_slice(),
        channel_count,
        batch_count,
    };

    (input, expected)
}

fn run_kernel<T: ArrayElement + Float, B: Backend>(input: &TestInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create context");

    let kernel = <<B as Backend>::Kernels as Kernels>::HadamardTransformKernel::new(&context, T::data_type())
        .expect("Failed to create HadamardTransformKernel");

    let total_elements = input.batch_count * input.channel_count;
    let data_array = context.create_array_from(&[total_elements], &input.data, "data");
    let factors_array = context.create_array_from::<i32>(&[input.channel_count], &input.factors, "factors");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        data_array.buffer().borrow_mut().deref_mut(),
        factors_array.buffer().borrow().deref(),
        input.channel_count as u32,
        input.batch_count as u32,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    data_array.as_slice().to_vec()
}

fn test_hadamard_transform<T: ArrayElement + Float + Debug>(tolerance: f64) {
    let test_cases = [(1, 32), (1, 64), (1, 128), (4, 32), (4, 256), (2, 2048)];

    for (batch_count, channel_count) in test_cases {
        let (input, expected) = generate_test_input::<T>(batch_count, channel_count);

        for_each_non_cpu_backend!(|B| {
            let actual = run_kernel::<T, B>(&input);
            assert_eq!(actual.len(), expected.len());

            for (index, (actual_value, &expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
                let actual_f64 = actual_value.to_f64().unwrap();
                let error = (actual_f64 - expected_value).abs();
                let relative_bound = expected_value.abs() * tolerance;
                let absolute_bound = tolerance;
                assert!(
                    error <= relative_bound.max(absolute_bound),
                    "Mismatch at index {index} for batch_count={batch_count}, channel_count={channel_count}: \
                     actual={actual_f64}, expected={expected_value}, error={error}"
                );
            }
        });
    }
}

#[uzu_test]
fn test_f32() {
    test_hadamard_transform::<f32>(1e-4);
}

#[uzu_test]
fn test_f16() {
    test_hadamard_transform::<f16>(0.05);
}

#[uzu_test]
fn test_bf16() {
    test_hadamard_transform::<bf16>(0.1);
}
