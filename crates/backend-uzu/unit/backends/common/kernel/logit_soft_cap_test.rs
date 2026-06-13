use std::fmt::Display;

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::LogitSoftCapKernel},
        cpu::Cpu,
    },
    common::helpers::{alloc_allocation_with_data, allocation_to_vec},
    data_type::DataType,
    tests::assert::assert_eq_float,
};

fn get_output<T: ArrayElement + Float, B: Backend>(
    logits: &[T],
    soft_cap: f32,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::LogitSoftCapKernel::new(&context, T::data_type())
        .expect("Failed to create LogitSoftCapKernel");

    let mut logits_allocation = alloc_allocation_with_data::<B, T>(&context, logits);
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&mut logits_allocation, logits.len() as u32, soft_cap, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&logits_allocation)
}

fn test_logit_soft_cap<T: ArrayElement + Float + Display>() {
    let logits = [-120.0f32, -30.0, -3.5, 0.0, 2.0, 30.0, 120.0]
        .into_iter()
        .map(|value| T::from(value).unwrap())
        .collect::<Box<[_]>>();
    let soft_cap = 30.0;
    let expected = get_output::<T, Cpu>(&logits, soft_cap);
    let epsilon = if matches!(T::data_type(), DataType::BF16) {
        0.05
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let actual = get_output::<T, B>(&logits, soft_cap);
        let message = format!("LogitSoftCap failed with backend={}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &actual, epsilon, &message);
    });
}

#[uzu_test]
fn test_logit_soft_cap_f32() {
    test_logit_soft_cap::<f32>();
}

#[uzu_test]
fn test_logit_soft_cap_bf16() {
    test_logit_soft_cap::<bf16>();
}
