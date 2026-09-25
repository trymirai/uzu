use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            kernel::{ActivationTransform, NormalizationKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{
            alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend, for_each_non_cpu_backend,
        },
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
) -> Vec<OutputT> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::NormalizationKernel::new(
        &context,
        InputT::data_type(),
        AffineT::data_type(),
        OutputT::data_type(),
        DataType::F32,
        false,
        false,
        full_layer,
        false,
        false,
        hadamard_factors.is_some(),
        false,
        false,
        false,
        scales.is_some(),
    )
    .expect("Failed to create NormalizationKernel");

    let input_allocation = alloc_allocation_with_data::<B, InputT>(&context, input);
    let scales_allocation = scales.map(|scales| alloc_allocation_with_data::<B, AffineT>(&context, scales));
    let hadamard_factors_allocation =
        hadamard_factors.map(|hadamard_factors| alloc_allocation_with_data::<B, i32>(&context, hadamard_factors));
    let mut output_allocation = alloc_allocation_with_data::<B, OutputT>(&context, &vec![OutputT::zero(); input.len()]);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        Some((&input_allocation, 0)),
        scales_allocation.as_ref(),
        None::<&Allocation<B>>,
        &mut output_allocation,
        None::<(&mut Allocation<B>, usize)>,
        hadamard_factors_allocation.as_ref(),
        batch_size,
        element_count,
        epsilon,
        0.0,
        1.0,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    allocation_to_vec::<B, OutputT>(&output_allocation)
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

    let plain = get_output::<Cpu, T, T, T>(&input, Some(&scales), batch_size, element_count, epsilon, true, None);

    let context = <Cpu as Backend>::Context::new().expect("Failed to create Context");
    let input_rht = ActivationTransform::<Cpu>::input_rht(context.as_ref(), T::data_type(), false)
        .expect("Failed to create ActivationTransform");
    let plain_allocation = alloc_allocation_with_data::<Cpu, T>(&context, &plain);
    let hadamard_factors_allocation = alloc_allocation_with_data::<Cpu, i32>(&context, &hadamard_factors);
    let mut expected_allocation = alloc_allocation::<Cpu, T>(&context, plain.len());
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    input_rht.encode_fp(
        &plain_allocation,
        &mut expected_allocation,
        &hadamard_factors_allocation,
        batch_size,
        element_count,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");
    let expected = allocation_to_vec::<Cpu, T>(&expected_allocation);

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
        );
        let message = format!("Normalization hadamard kernel test failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, eps, &message);
    });
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
