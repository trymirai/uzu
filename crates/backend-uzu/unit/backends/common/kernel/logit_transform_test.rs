use std::fmt::Display;

use backend_uzu_macros::uzu_test;
use half::bf16;
use num_traits::Float;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::LogitTransformKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

fn get_output<T: ArrayElement + Float, B: Backend>(
    logits: &[T],
    scale: f32,
    soft_cap: Option<f32>,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::LogitTransformKernel::new(&context, T::data_type(), soft_cap.is_some())
            .expect("Failed to create LogitTransformKernel");

    let mut logits_allocation = alloc_allocation_with_data::<B, T>(&context, logits);
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&mut logits_allocation, logits.len() as u32, scale, soft_cap.unwrap_or(0.0), &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&logits_allocation)
}

fn test_logit_transform<T: ArrayElement + Float + Display>(
    scale: f32,
    soft_cap: Option<f32>,
) {
    let logits = [-120.0f32, -30.0, -3.5, 0.0, 2.0, 30.0, 120.0]
        .into_iter()
        .map(|value| T::from(value).unwrap())
        .collect::<Box<[_]>>();
    let expected = get_output::<T, Cpu>(&logits, scale, soft_cap);
    let epsilon = if matches!(T::data_type(), DataType::BF16) {
        0.05
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<T, B>(&logits, scale, soft_cap);
        let message = format!(
            "LogitTransform failed with backend={}, scale={}, soft_cap={:?}",
            std::any::type_name::<B>(),
            scale,
            soft_cap
        );
        assert_eq_float::<T>(&expected, &actual, epsilon, &message);
    });
}

#[uzu_test]
fn test_soft_cap_f32() {
    test_logit_transform::<f32>(1.0, Some(30.0));
}

#[uzu_test]
fn test_soft_cap_bf16() {
    test_logit_transform::<bf16>(1.0, Some(30.0));
}

#[uzu_test]
fn test_scale_f32() {
    test_logit_transform::<f32>(2.5, None);
}

#[uzu_test]
fn test_scale_bf16() {
    test_logit_transform::<bf16>(2.5, None);
}

#[uzu_test]
fn test_scale_and_soft_cap_f32() {
    test_logit_transform::<f32>(2.5, Some(30.0));
}

#[uzu_test]
fn test_scale_and_soft_cap_bf16() {
    test_logit_transform::<bf16>(2.5, Some(30.0));
}
