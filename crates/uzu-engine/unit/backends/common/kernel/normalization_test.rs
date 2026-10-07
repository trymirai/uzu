use std::{
    fmt::{Debug, Display},
    mem::size_of,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{ActivationTransform, NormalizationKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_backend, for_each_non_cpu_backend},
    },
};

fn get_output<
    B: Backend,
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
>(
    input: &[InputT],
    scales: Option<&[AffineT]>,
    batch_size: u32,
    element_count: u32,
    epsilon: f32,
    full_layer: bool,
    hadamard_factors: Option<&[i32]>,
    subtract_mean: bool,
    scale_output: Option<f32>,
) -> Vec<OutputT> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::NormalizationKernel::new(
        &context,
        InputT::data_type(),
        AffineT::data_type(),
        OutputT::data_type(),
        DataType::F32,
        false,
        subtract_mean,
        full_layer,
        false,
        false,
        hadamard_factors.is_some(),
        false,
        scale_output.is_some(),
        false,
        scales.is_some(),
    )
    .expect("Failed to create NormalizationKernel");

    let input_buffer = create_buffer_with_data::<B, InputT>(&context, input);
    let scales_buffer = scales.map(|scales| create_buffer_with_data::<B, AffineT>(&context, scales));
    let hadamard_factors_buffer =
        hadamard_factors.map(|hadamard_factors| create_buffer_with_data::<B, i32>(&context, hadamard_factors));
    let mut output_buffer = create_buffer_with_data::<B, OutputT>(&context, &vec![OutputT::zero(); input.len()]);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        Some(&input_buffer),
        scales_buffer.as_ref(),
        None::<&B::GlobalBuffer>,
        &mut output_buffer,
        None::<&mut B::GlobalBuffer>,
        hadamard_factors_buffer.as_ref(),
        batch_size,
        element_count,
        epsilon,
        0.0,
        scale_output.unwrap_or(1.0),
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    buffer_to_vec::<B, OutputT>(&output_buffer)
}

fn test_internal<
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
>(
    has_scales: bool,
    full_layer: bool,
) {
    let batch_size = 2u32;
    let element_count = 64u32;
    let epsilon = 1e-6f32;

    let input: Vec<InputT> =
        (0..(batch_size * element_count)).map(|index| InputT::from(0.5f32 + (index as f32) * 0.01).unwrap()).collect();
    let scales: Vec<AffineT> =
        (0..element_count).map(|index| AffineT::from(1.0f32 + (index as f32) * 0.001).unwrap()).collect();
    let scales = has_scales.then_some(scales);

    let expected = get_output::<Cpu, InputT, AffineT, OutputT>(
        &input,
        scales.as_deref(),
        batch_size,
        element_count,
        epsilon,
        full_layer,
        None,
        false,
        None,
    );

    let eps = if matches!(InputT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(AffineT::data_type(), DataType::F16 | DataType::BF16)
        || matches!(OutputT::data_type(), DataType::F16 | DataType::BF16)
    {
        1e-2
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<B, InputT, AffineT, OutputT>(
            &input,
            scales.as_deref(),
            batch_size,
            element_count,
            epsilon,
            full_layer,
            None,
            false,
            None,
        );
        let message = format!(
            "Normalization kernel test failed with backend={}, has_scales={}, full_layer={}",
            std::any::type_name::<B>(),
            has_scales,
            full_layer,
        );
        assert_eq_float::<OutputT>(&expected, &actual, eps, &message);
    });
}

fn test_normalization<
    InputT: ArrayElement + Float,
    AffineT: ArrayElement + Float,
    OutputT: ArrayElement + Float + Debug + Display,
>() {
    for has_scales in [true, false] {
        for full_layer in [true, false] {
            test_internal::<InputT, AffineT, OutputT>(has_scales, full_layer);
        }
    }
}

// The fused transform has to match plain normalization followed by the standalone input RHT
fn test_hadamard<T: ArrayElement + Float + Debug + Display>() {
    let batch_size = 2u32;
    let element_count = 64u32;
    let epsilon = 1e-6f32;

    let input: Vec<T> =
        (0..(batch_size * element_count)).map(|index| T::from(0.5f32 + (index as f32) * 0.01).unwrap()).collect();
    let scales: Vec<T> = (0..element_count).map(|index| T::from(1.0f32 + (index as f32) * 0.001).unwrap()).collect();
    let hadamard_factors: Vec<i32> = (0..element_count)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect();

    let plain =
        get_output::<Cpu, T, T, T>(&input, Some(&scales), batch_size, element_count, epsilon, true, None, false, None);

    let context = <Cpu as Backend>::Context::new().expect("Failed to create Context");
    let input_rht = ActivationTransform::<Cpu>::input_rht(context.as_ref(), T::data_type(), false)
        .expect("Failed to create ActivationTransform");
    let plain_buffer = create_buffer_with_data::<Cpu, T>(&context, &plain);
    let hadamard_factors_buffer = create_buffer_with_data::<Cpu, i32>(&context, &hadamard_factors);
    let mut expected_buffer = create_buffer::<Cpu, T>(&context, plain.len());
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    input_rht.encode_fp(
        &plain_buffer,
        &mut expected_buffer,
        &hadamard_factors_buffer,
        batch_size,
        element_count,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    let expected = buffer_to_vec::<Cpu, T>(&expected_buffer);

    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-2
    } else {
        1e-5
    };

    for_each_backend!(|B| {
        let actual = get_output::<B, T, T, T>(
            &input,
            Some(&scales),
            batch_size,
            element_count,
            epsilon,
            true,
            Some(&hadamard_factors),
            false,
            None,
        );
        let message = format!("Normalization hadamard kernel test failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps, &message);
    });
}

/// Position of a value in the total order of its storage type, so differences count representable steps.
fn ordinal<T: ArrayElement>(value: T) -> i64 {
    let (bits, sign) = match *bytemuck::bytes_of(&value) {
        [a, b] => (i64::from(u16::from_ne_bytes([a, b])), 1 << 15),
        [a, b, c, d] => (i64::from(u32::from_ne_bytes([a, b, c, d])), 1 << 31),
        _ => unreachable!("normalization storage types are 16 or 32 bits"),
    };
    if bits & sign != 0 {
        -(bits & !sign)
    } else {
        bits
    }
}

// Layer normalization of near-constant rows around ±1000 and of uniform rows against an FP64 two-pass oracle of the
// stored inputs: F32 within relative 2e-6 or absolute 1e-6, 16-bit types within 2 storage steps.
fn test_shifted_variance<T: ArrayElement + Float + Debug>() {
    let epsilon = 1e-5f32;
    // One storage step at 1000, so every row keeps a variance in its type.
    let step = T::epsilon().to_f32().unwrap() * 512.0;
    for element_count in [33u32, 257, 4096] {
        let n = element_count as usize;
        let rows = [1000.0f32, -1000.0, 1000.0].into_iter().enumerate().flat_map(|(row, center)| {
            (0..n).map(move |i| {
                T::from(
                    center
                        + if row == 2 {
                            0.0
                        } else {
                            ((i * 37) % 9) as f32 * step
                        },
                )
                .unwrap()
            })
        });
        let input = rows.collect::<Vec<_>>();
        let expected = input
            .chunks(n)
            .flat_map(|row| {
                let values = row.iter().map(|value| value.to_f64().unwrap()).collect::<Vec<_>>();
                let mean = values.iter().sum::<f64>() / n as f64;
                let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / n as f64;
                values.into_iter().map(move |value| (value - mean) / (variance + f64::from(epsilon)).sqrt())
            })
            .collect::<Vec<_>>();
        for_each_backend!(|B| {
            let actual = get_output::<B, T, T, T>(&input, None, 3, element_count, epsilon, true, None, true, None);
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let (value, rounded) = (actual.to_f64().unwrap(), T::from(expected).unwrap());
                let within = match size_of::<T>() {
                    4 => (value - expected).abs() <= 2e-6 * expected.abs() || (value - expected).abs() <= 1e-6,
                    _ => (ordinal(actual) - ordinal(rounded)).abs() <= 2,
                };
                assert!(
                    within,
                    "{} {:?} length {element_count} element {index}: expected {expected}, actual {value}",
                    std::any::type_name::<B>(),
                    T::data_type()
                );
            }
        });
    }
}

#[uzu_test]
fn test_normalization_shifted_variance() {
    test_shifted_variance::<f32>();
    test_shifted_variance::<f16>();
    test_shifted_variance::<bf16>();
}

// Output scaling multiplies the stored output by the FP32 scalar, not by the scalar rounded to the output type: rows of
// ±1 normalize exactly, so with integer scales every output before scaling is exact.
fn test_scale_output_rounding<T: ArrayElement + Float + Debug>() {
    let input = (0..64)
        .map(|i| {
            T::from(if i % 2 == 0 {
                1.0
            } else {
                -1.0
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let scales = (1..=64).map(|scale| T::from(scale).unwrap()).collect::<Vec<_>>();
    let expected = input
        .iter()
        .zip(&scales)
        .map(|(&x, &scale)| T::from(x.to_f32().unwrap() * scale.to_f32().unwrap() * 0.3).unwrap())
        .collect::<Vec<_>>();
    for_each_backend!(|B| {
        let actual = get_output::<B, T, T, T>(&input, Some(&scales), 1, 64, 0.0, true, None, false, Some(0.3));
        let bits = |values: &[T]| values.iter().map(|&value| ordinal(value)).collect::<Vec<_>>();
        assert_eq!(bits(&actual), bits(&expected), "{} {:?}", std::any::type_name::<B>(), T::data_type());
    });
}

#[uzu_test]
fn test_normalization_scale_output_rounding() {
    test_scale_output_rounding::<f16>();
    test_scale_output_rounding::<bf16>();
}

#[uzu_test]
fn test_normalization_hadamard_f32() {
    test_hadamard::<f32>();
}

#[uzu_test]
fn test_normalization_hadamard_bf16() {
    test_hadamard::<bf16>();
}

#[uzu_test]
fn test_normalization_f32_f32_f32() {
    test_normalization::<f32, f32, f32>();
}

#[uzu_test]
fn test_normalization_bf16_bf16_bf16() {
    test_normalization::<bf16, bf16, bf16>();
}
