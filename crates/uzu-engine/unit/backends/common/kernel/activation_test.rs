use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::ActivationType, kernel::ActivationKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
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
        let mut output_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(None::<&B::GlobalBuffer>, &mut output_buffer, n as u32, input.act_type, &mut command_buffer);
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<B, T>(&output_buffer)
    } else {
        let input_buffer = create_buffer_with_data::<B, T>(&context, &input.data);
        let mut output_buffer = create_buffer::<B, T>(&context, n);
        let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to get command buffer");
        kernel.encode(Some(&input_buffer), &mut output_buffer, n as u32, input.act_type, &mut command_buffer);
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<B, T>(&output_buffer)
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

/// Distance between two values in representable steps of their 16- or 32-bit storage type.
fn storage_steps<T: ArrayElement>(
    a: T,
    b: T,
) -> i64 {
    let ordinal = |value: T| {
        let (bits, sign) = match *bytemuck::bytes_of(&value) {
            [lo, hi] => (i64::from(u16::from_ne_bytes([lo, hi])), 1 << 15),
            [b0, b1, b2, b3] => (i64::from(u32::from_ne_bytes([b0, b1, b2, b3])), 1 << 31),
            _ => unreachable!("storage types are 16 or 32 bits"),
        };
        if bits & sign != 0 {
            -(bits & !sign)
        } else {
            bits
        }
    };
    (ordinal(a) - ordinal(b)).abs()
}

/// SiLU's negative tails through the CPU's FP32 exponential overflow below -88.72: the CPU and every other backend
/// within two storage steps of x / (1 + e^-x) in FP64 for the input as stored, or -0 where e^-x overflows FP32.
fn test_silu_negative_tail<T: ArrayElement + Float + Debug + Display>() {
    let tail = [-10.0f32, -16.0, -17.0, -20.0, -30.0, -50.0, -87.0, -88.0, -88.72, -88.73, -89.0];
    let input = Input::<T> {
        data: tail.iter().map(|&x| T::from(x).unwrap()).collect(),
        act_type: ActivationType::SILU,
        in_place: false,
    };
    let expected = input
        .data
        .iter()
        .map(|&x| {
            let x = x.to_f64().unwrap();
            let e = (-x).exp();
            T::from(if (e as f32).is_infinite() {
                -0.0
            } else {
                x / (1.0 + e)
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let check = |actual: &[T], backend: &str| {
        assert_eq!(actual.len(), input.data.len(), "SiLU {backend} {:?}: output length", T::data_type());
        for ((x, &expected), &actual) in input.data.iter().zip(&expected).zip(actual) {
            let same_zero =
                expected.is_zero() == actual.is_zero() && expected.is_sign_negative() == actual.is_sign_negative();
            assert!(
                storage_steps(expected, actual) <= 2 && same_zero,
                "SiLU {backend} {:?} x {x}: {actual}, expected {expected}",
                T::data_type()
            );
        }
    };
    check(&get_output::<T, Cpu>(&input), "CPU");
    for_each_non_cpu_backend!(|B| {
        check(&get_output::<T, B>(&input), std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_silu_negative_tail_all_types() {
    test_silu_negative_tail::<f32>();
    test_silu_negative_tail::<f16>();
    test_silu_negative_tail::<bf16>();
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
