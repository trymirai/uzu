use std::fmt::Debug;

use half::{bf16, f16};
use num_traits::Float;
use backend_uzu::{
    ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, kernel::TensorCopyKernel},
};

use crate::{
    common::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    src: Box<[T]>,
    length: u32,
}

fn get_test_data<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let length = 1025usize;

    let mut src: Vec<T> = vec![T::zero(); length];
    let mut expected: Vec<T> = vec![T::zero(); length];

    for i in 0..length {
        src[i] = T::from((i as f32).sin() * 30f32).unwrap();
        expected[i] = src[i];
    }

    let input = Input {
        src: src.into_boxed_slice(),
        length: length as u32,
    };

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TensorCopyKernel::new(&context, T::data_type())
        .expect("Failed to create TensorCopyKernel");

    let size = input.length as usize;
    let src_allocation = alloc_allocation_with_data::<B, T>(&context, &input.src[..size]);
    let mut dst_allocation = alloc_allocation::<B, T>(&context, size);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&src_allocation, &mut dst_allocation, input.length, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&dst_allocation)
}

fn test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = get_test_data::<T>();
    for_each_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq!(output, expected, "Results are not equal for backend {}", std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_f32() {
    test::<f32>();
}

#[uzu_test]
fn test_f16() {
    test::<f16>();
}

#[uzu_test]
fn test_bf16() {
    test::<bf16>();
}
