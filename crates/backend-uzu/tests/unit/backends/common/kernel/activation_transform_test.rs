use std::fmt::Debug;

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::common::{
        Backend, Context, Encoder, gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE as BLOCK_SIZE, kernel::ActivationTransform,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend},
};

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
            let expected = reference_transform(&data_f64, &factors, channel_count, order);
            let data: Vec<T> = data_f64.iter().map(|&value| T::from(value).unwrap()).collect();

            for_each_backend!(|B| {
                let actual = run::<T, B>(&data, &factors, channel_count, order, in_place);
                for (index, (actual_value, &expected_value)) in actual.iter().zip(&expected).enumerate() {
                    let actual_value = actual_value.to_f64().unwrap();
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

#[cfg(backend = "metal")]
mod quantize {
    use proc_macros::uzu_test;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    use super::BLOCK_SIZE;
    use crate::{
        backends::{
            common::{Backend, Context, Encoder, kernel::ActivationTransform},
            cpu::Cpu,
            metal::{Metal, MetalContext},
        },
        data_type::DataType,
        tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    };

    fn check_quantize(emit_group_sums: bool) {
        let rows = 5usize;
        let columns = 96usize;
        let groups = columns / BLOCK_SIZE;

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

        let metal = MetalContext::new().expect("metal");
        let cpu = <Cpu as Backend>::Context::new().expect("cpu");
        let mut metal_values = alloc_allocation::<Metal, i8>(metal.as_ref(), rows * columns);
        let mut metal_scales = alloc_allocation::<Metal, f32>(metal.as_ref(), rows * groups);
        let mut cpu_values = alloc_allocation::<Cpu, i8>(cpu.as_ref(), rows * columns);
        let mut cpu_scales = alloc_allocation::<Cpu, f32>(cpu.as_ref(), rows * groups);
        let mut metal_group_sums =
            emit_group_sums.then(|| alloc_allocation::<Metal, i32>(metal.as_ref(), rows * groups));
        let mut cpu_group_sums = emit_group_sums.then(|| alloc_allocation::<Cpu, i32>(cpu.as_ref(), rows * groups));
        let metal_input = alloc_allocation_with_data::<Metal, f32>(metal.as_ref(), &input_data);
        let metal_factors = alloc_allocation_with_data::<Metal, i32>(metal.as_ref(), &factors_data);
        let cpu_input = alloc_allocation_with_data::<Cpu, f32>(cpu.as_ref(), &input_data);
        let cpu_factors = alloc_allocation_with_data::<Cpu, i32>(cpu.as_ref(), &factors_data);

        let metal_kernel =
            ActivationTransform::quantize(metal.as_ref(), DataType::F32, emit_group_sums).expect("metal prepare");
        let cpu_kernel =
            ActivationTransform::quantize(cpu.as_ref(), DataType::F32, emit_group_sums).expect("cpu prepare");

        let mut metal_enc = Encoder::<Metal>::new(metal.as_ref()).expect("metal encoder");
        metal_kernel.encode_quantize(
            &metal_input,
            &mut metal_values,
            &mut metal_scales,
            metal_group_sums.as_mut(),
            &metal_factors,
            rows as u32,
            columns as u32,
            &mut metal_enc,
        );
        metal_enc.end_encoding().submit().wait_until_completed().unwrap();

        let mut cpu_enc = Encoder::<Cpu>::new(cpu.as_ref()).expect("cpu encoder");
        cpu_kernel.encode_quantize(
            &cpu_input,
            &mut cpu_values,
            &mut cpu_scales,
            cpu_group_sums.as_mut(),
            &cpu_factors,
            rows as u32,
            columns as u32,
            &mut cpu_enc,
        );
        cpu_enc.end_encoding().submit().wait_until_completed().unwrap();

        let (mv, ms) = (allocation_to_vec::<Metal, i8>(&metal_values), allocation_to_vec::<Metal, f32>(&metal_scales));
        let (cv, cs) = (allocation_to_vec::<Cpu, i8>(&cpu_values), allocation_to_vec::<Cpu, f32>(&cpu_scales));
        for (i, (a, e)) in ms.iter().zip(&cs).enumerate() {
            let rel = (a - e).abs() / e.abs().max(1e-6);
            assert!(rel < 1e-3, "scale {i}: {a} != {e}");
        }
        assert!(mv.iter().zip(&cv).all(|(a, e)| (*a as i32 - *e as i32).abs() <= 1));

        if let (Some(metal_group_sums), Some(cpu_group_sums)) = (&metal_group_sums, &cpu_group_sums) {
            let mrs = allocation_to_vec::<Metal, i32>(metal_group_sums);
            let crs = allocation_to_vec::<Cpu, i32>(cpu_group_sums);
            for (codes, sums, label) in [(&mv, &mrs, "metal"), (&cv, &crs, "cpu")] {
                for row in 0..rows {
                    for group in 0..groups {
                        let start = row * columns + group * BLOCK_SIZE;
                        let expected: i32 = codes[start..start + BLOCK_SIZE].iter().map(|code| *code as i32).sum();
                        assert_eq!(sums[row * groups + group], expected, "{label} group_sum r{row} g{group}");
                    }
                }
            }
            assert!(mrs.iter().zip(&crs).all(|(a, e)| (a - e).abs() <= BLOCK_SIZE as i32));
        }
    }

    #[uzu_test]
    fn quantize_with_group_sums_matches_cpu() {
        check_quantize(true);
    }

    #[uzu_test]
    fn quantize_without_group_sums_matches_cpu() {
        check_quantize(false);
    }
}
