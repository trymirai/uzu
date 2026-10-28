use std::fmt::{Debug, Display};

use futures::executor::block_on;
use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::activation_type::ActivationType, kernel::ActivationKernel,
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

struct Input<T: DataTypeRepr + Float> {
    data: Box<[T]>,
    act_type: ActivationType,
    in_place: bool,
}

fn make_data<T: DataTypeRepr + Float>(n: usize) -> Vec<T> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        // Use a range of values including negative to exercise both branches of activations
        data.push(T::from((i as f32 * 0.3) - 2.0).unwrap());
    }
    data
}

fn get_output<T: DataTypeRepr + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = block_on(B::Context::new()).expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::ActivationKernel::new(&context, T::data_type(), input.in_place)
        .expect("Failed to create ActivationKernel");

    let n = input.data.len();

    if input.in_place {
        let mut output_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(None::<&B::GlobalGpuBuffer>, &mut output_buffer, n as u32, input.act_type, &mut command_buffer);
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        buffer_to_vec::<B, T>(&context, &output_buffer)
    } else {
        let input_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut output_buffer = create_buffer::<B, T>(&context, n);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(Some(&input_buffer), &mut output_buffer, n as u32, input.act_type, &mut command_buffer);
        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
        buffer_to_vec::<B, T>(&context, &output_buffer)
    }
}

fn get_test_data<T: DataTypeRepr + Float>(
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
fn get_test_data_large<T: DataTypeRepr + Float>(
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

fn test_internal<T: DataTypeRepr + Float + Debug + Display>(
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

fn test_activation<T: DataTypeRepr + Float + Debug + Display>(
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

fn test_activation_large<T: DataTypeRepr + Float + Debug + Display>(
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
    test_activation::<f32>(ActivationType::GELUApprox, false);
}

#[uzu_test]
fn test_gelu_f16() {
    test_activation::<f16>(ActivationType::GELUApprox, false);
}

#[uzu_test]
fn test_gelu_bf16() {
    test_activation::<bf16>(ActivationType::GELUApprox, false);
}

// Exact GELU out-of-place tests
#[uzu_test]
fn test_gelu_exact_f32() {
    test_activation::<f32>(ActivationType::GELUExact, false);
}

#[uzu_test]
fn test_gelu_exact_f16() {
    test_activation::<f16>(ActivationType::GELUExact, false);
}

#[uzu_test]
fn test_gelu_exact_bf16() {
    test_activation::<bf16>(ActivationType::GELUExact, false);
}

// GELU in-place tests
#[uzu_test]
fn test_gelu_in_place_f32() {
    test_activation::<f32>(ActivationType::GELUApprox, true);
}

#[uzu_test]
fn test_gelu_in_place_f16() {
    test_activation::<f16>(ActivationType::GELUApprox, true);
}

#[uzu_test]
fn test_gelu_in_place_bf16() {
    test_activation::<bf16>(ActivationType::GELUApprox, true);
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
    test_activation_large::<f32>(ActivationType::GELUApprox, false);
}

#[uzu_test]
fn test_gelu_large_f16() {
    test_activation_large::<f16>(ActivationType::GELUApprox, false);
}

#[uzu_test]
fn test_gelu_large_bf16() {
    test_activation_large::<bf16>(ActivationType::GELUApprox, false);
}

// Large in-place tests
#[uzu_test]
fn test_silu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::SILU, true);
}

#[uzu_test]
fn test_gelu_in_place_large_f32() {
    test_activation_large::<f32>(ActivationType::GELUApprox, true);
}
