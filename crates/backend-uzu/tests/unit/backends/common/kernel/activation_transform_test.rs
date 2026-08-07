use std::fmt::Debug;

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, Context, Encoder, gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE as BLOCK_SIZE,
            kernel::ActivationTransform,
        },
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend},
};

#[derive(Clone, Copy, Debug)]
enum TransformOrder {
    Input,
    Output,
}

fn run<T: ArrayElement + Float, B: Backend>(
    data: &[T],
    factors: &[i32],
    channel_count: usize,
    order: TransformOrder,
    in_place: bool,
) -> Vec<T> {
    let context = B::Context::new().expect("context");
    let kernel = match order {
        TransformOrder::Input => ActivationTransform::<B>::input_rht(context.as_ref(), T::data_type(), in_place),
        TransformOrder::Output => ActivationTransform::<B>::output_rht(context.as_ref(), T::data_type(), in_place),
    }
    .expect("activation transform");

    let mut input = alloc_allocation_with_data::<B, T>(context.as_ref(), data);
    let mut output = alloc_allocation::<B, T>(context.as_ref(), data.len());
    let factors = alloc_allocation_with_data::<B, i32>(context.as_ref(), factors);
    let batch_count = (data.len() / channel_count) as u32;
    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    if in_place {
        kernel.encode_fp_in_place(&mut input, &factors, batch_count, channel_count as u32, &mut encoder);
    } else {
        kernel.encode_fp(&input, &mut output, &factors, batch_count, channel_count as u32, &mut encoder);
    }
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(if in_place {
        &input
    } else {
        &output
    })
}

fn check<T: ArrayElement + Float + Debug>(tolerance: f64) {
    for (order, in_place) in [
        (TransformOrder::Input, false),
        (TransformOrder::Input, true),
        (TransformOrder::Output, false),
        (TransformOrder::Output, true),
    ] {
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
            let data: Vec<T> = data_f64.iter().map(|&value| T::from(value).unwrap()).collect();
            let expected = run::<T, Cpu>(&data, &factors, channel_count, order, in_place);

            for_each_backend!(|B| {
                let actual = run::<T, B>(&data, &factors, channel_count, order, in_place);
                for (index, (actual_value, expected_value)) in actual.iter().zip(&expected).enumerate() {
                    let actual_value = actual_value.to_f64().unwrap();
                    let expected_value = expected_value.to_f64().unwrap();
                    let error = (actual_value - expected_value).abs();
                    assert!(
                        error <= (expected_value.abs() * tolerance).max(tolerance),
                        "{order:?} (in_place={in_place}) mismatch at {index} for batch={batch_count}, \
                         channels={channel_count}: actual={actual_value}, expected={expected_value}, error={error}"
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

mod quantize {
    use proc_macros::uzu_test;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::BLOCK_SIZE;
    use crate::{
        backends::{
            common::{Backend, Context, Encoder, kernel::ActivationTransform},
            cpu::Cpu,
        },
        data_type::DataType,
        tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend},
    };

    fn run<B: Backend>(
        input_data: &[f32],
        factors_data: &[i32],
        rows: usize,
        columns: usize,
        activation_group_size: usize,
        emit_group_sums: bool,
        sum_group_size: Option<usize>,
    ) -> (Vec<i8>, Vec<f32>, Option<Vec<i32>>) {
        let scale_groups = columns / activation_group_size;
        let sum_groups = sum_group_size.map_or(0, |group_size| columns / group_size);
        let context = B::Context::new().expect("context");
        let input = alloc_allocation_with_data::<B, f32>(context.as_ref(), input_data);
        let factors = alloc_allocation_with_data::<B, i32>(context.as_ref(), factors_data);
        let mut values = alloc_allocation::<B, i8>(context.as_ref(), rows * columns);
        let mut scales = alloc_allocation::<B, f32>(context.as_ref(), rows * scale_groups);
        let mut group_sums = emit_group_sums.then(|| alloc_allocation::<B, i32>(context.as_ref(), rows * sum_groups));
        let kernel = ActivationTransform::quantize(
            context.as_ref(),
            DataType::F32,
            activation_group_size,
            sum_group_size.map(|size| size as u32),
        )
        .expect("quantize transform");
        let mut encoder = Encoder::<B>::new(context.as_ref()).expect("encoder");
        kernel.encode_quantize(
            &input,
            &mut values,
            &mut scales,
            group_sums.as_mut(),
            &factors,
            rows as u32,
            columns as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        (allocation_to_vec(&values), allocation_to_vec(&scales), group_sums.as_ref().map(allocation_to_vec))
    }

    fn check_quantize(
        activation_group_size: usize,
        emit_group_sums: bool,
        sum_group_size: Option<usize>,
    ) {
        let rows = 3;
        let columns = 256;
        let mut rng = SmallRng::seed_from_u64(0x5EED_0001);
        let input_data: Vec<f32> = (0..rows * columns).map(|_| rng.random_range(-1.0f32..1.0f32)).collect();
        let factors_data: Vec<i32> = (0..columns)
            .map(|i| {
                if i % 3 == 0 {
                    -1
                } else {
                    1
                }
            })
            .collect();
        let (expected_values, expected_scales, expected_group_sums) = run::<Cpu>(
            &input_data,
            &factors_data,
            rows,
            columns,
            activation_group_size,
            emit_group_sums,
            sum_group_size,
        );

        for_each_backend!(|B| {
            let (actual_values, actual_scales, actual_group_sums) = run::<B>(
                &input_data,
                &factors_data,
                rows,
                columns,
                activation_group_size,
                emit_group_sums,
                sum_group_size,
            );

            for (index, (&actual, &expected)) in actual_scales.iter().zip(&expected_scales).enumerate() {
                let relative_error = (actual - expected).abs() / expected.abs().max(1e-6);
                assert!(relative_error < 1e-3, "scale {index}: {actual} != {expected}");
            }
            for (index, (&actual, &expected)) in actual_values.iter().zip(&expected_values).enumerate() {
                assert!((i32::from(actual) - i32::from(expected)).abs() <= 1, "code {index}: {actual} != {expected}");
            }

            match actual_group_sums {
                Some(actual_group_sums) => {
                    for (group_index, (&actual, &expected)) in
                        actual_group_sums.iter().zip(expected_group_sums.as_ref().expect("CPU group sums")).enumerate()
                    {
                        let sum_group_size = sum_group_size.expect("correction group");
                        let start = group_index * sum_group_size;
                        let sum_from_actual_codes: i32 =
                            actual_values[start..start + sum_group_size].iter().copied().map(i32::from).sum();
                        assert_eq!(actual, sum_from_actual_codes, "group sum {group_index}");
                        assert!(
                            (actual - expected).abs() <= sum_group_size as i32,
                            "group sum {group_index}: {actual} != {expected}"
                        );
                    }
                },
                None => assert!(!emit_group_sums),
            }
        });
    }

    #[uzu_test]
    fn quantize_with_group_sums_matches_cpu() {
        check_quantize(128, true, Some(BLOCK_SIZE));
    }

    #[uzu_test]
    fn quantize_without_group_sums_matches_cpu() {
        check_quantize(128, false, None);
    }

    #[uzu_test]
    fn quantize_compact_scale_g128_sum_g64_matches_cpu() {
        check_quantize(128, true, Some(64));
    }

    #[uzu_test]
    fn quantize_scale_g32_and_g64_match_cpu() {
        check_quantize(32, false, None);
        check_quantize(64, false, None);
    }
}
