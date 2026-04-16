use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::ActivationKernel},
        cpu::Cpu,
    },
};

use crate::{
    common::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    data: Box<[T]>,
    act_type: ActivationType,
    in_place: bool,
}

fn make_data<T: ArrayElement + Float>(n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // Use a range of values including negative to exercise both branches of activations
        data.push(T::from((i as f32 * 0.3) - 2.0).unwrap());
    }
    data
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create ActivationKernel");

    let n = input.data.len();

    if input.in_place {
        let mut output_allocation = alloc_allocation_with_data::<B, T>(&context, &input.data);
        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
        kernel.encode(None, &mut output_allocation, n as u32, input.act_type, &mut encoder);
        encoder.end_encoding().submit().wait_until_completed().unwrap();
        allocation_to_vec::<B, T>(&output_allocation)
    } else {
        let input_allocation = alloc_allocation_with_data::<B, T>(&context, &input.data);
        let mut output_allocation = alloc_allocation::<B, T>(&context, n);
        let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
        kernel.encode(Some(&input_allocation), &mut output_allocation, n as u32, input.act_type, &mut encoder);
        encoder.end_encoding().submit().wait_until_completed().unwrap();
        allocation_to_vec::<B, T>(&output_allocation)
    }
}

fn get_test_data<T: ArrayElement + Float>(
    act_type: ActivationType,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let n = 128;
    let data = make_data::<T>(n);

    let input = Input {
        data: data.into_boxed_slice(),
        act_type,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

/// Large input to exercise more threads on GPU.
fn get_test_data_large<T: ArrayElement + Float>(
    act_type: ActivationType,
    in_place: bool,
) -> (Input<T>, Vec<T>) {
    let n = 4096;
    let data = make_data::<T>(n);

    let input = Input {
        data: data.into_boxed_slice(),
        act_type,
        in_place,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
    test_name: &str,
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<T, B>(input);
        let msg = format!("Activation {} failed with backend={}", test_name, std::any::type_name::<B>(),);
        assert_eq_float::<T>(expected, &actual, eps, &msg);
    });
}

fn test_activation<T: ArrayElement + Float + Debug + Display>(
    act_type: ActivationType,
    in_place: bool,
) {
    let label = format!(
        "{:?}_{}",
        act_type,
        if in_place {
            "in_place"
        } else {
            "out_of_place"
        }
    );
    let (input, expected) = get_test_data::<T>(act_type, in_place);
    test_internal::<T>(&input, &expected, &label);
}

fn test_activation_large<T: ArrayElement + Float + Debug + Display>(
    act_type: ActivationType,
    in_place: bool,
) {
    let label = format!(
        "{:?}_{}_large",
        act_type,
        if in_place {
            "in_place"
        } else {
            "out_of_place"
        }
    );
    let (input, expected) = get_test_data_large::<T>(act_type, in_place);
    test_internal::<T>(&input, &expected, &label);
}

// SILU out-of-place tests
#[uzu_test]
fn test_silu_f32() {
    test_activation::<f32>(ActivationType::SILU, false);
}

#[uzu_test]
fn test_silu_f16() {
    test_activation::<f16>(ActivationType::SILU, false);
}

#[uzu_test]
fn test_silu_bf16() {
    test_activation::<bf16>(ActivationType::SILU, false);
}

// SILU in-place tests
#[uzu_test]
fn test_silu_in_place_f32() {
    test_activation::<f32>(ActivationType::SILU, true);
}

#[uzu_test]
fn test_silu_in_place_f16() {
    test_activation::<f16>(ActivationType::SILU, true);
}

#[uzu_test]
fn test_silu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::SILU, true);
}

// GELU out-of-place tests
#[uzu_test]
fn test_gelu_f32() {
    test_activation::<f32>(ActivationType::GELU, false);
}

#[uzu_test]
fn test_gelu_f16() {
    test_activation::<f16>(ActivationType::GELU, false);
}

#[uzu_test]
fn test_gelu_bf16() {
    test_activation::<bf16>(ActivationType::GELU, false);
}

// GELU in-place tests
#[uzu_test]
fn test_gelu_in_place_f32() {
    test_activation::<f32>(ActivationType::GELU, true);
}

#[uzu_test]
fn test_gelu_in_place_f16() {
    test_activation::<f16>(ActivationType::GELU, true);
}

#[uzu_test]
fn test_gelu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::GELU, true);
}

// Large SILU tests
#[uzu_test]
fn test_silu_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, false);
}

#[uzu_test]
fn test_silu_large_f16() {
    test_activation_large::<f16>(ActivationType::SILU, false);
}

#[uzu_test]
fn test_silu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::SILU, false);
}

// Large GELU tests
#[uzu_test]
fn test_gelu_large_f32() {
    test_activation_large::<f32>(ActivationType::GELU, false);
}

#[uzu_test]
fn test_gelu_large_f16() {
    test_activation_large::<f16>(ActivationType::GELU, false);
}

#[uzu_test]
fn test_gelu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::GELU, false);
}

// Large in-place tests
#[uzu_test]
fn test_silu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, true);
}

#[uzu_test]
fn test_gelu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::GELU, true);
}
