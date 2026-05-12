use std::fmt::Debug;

use half::{bf16, f16};
use num_traits::Float;
use backend_uzu::{
    ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, kernel::TensorAddSwapKernel},
};

use crate::{
    common::helpers::{alloc_allocation_with_data, allocation_to_vec},
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    skip_buffer: Box<[T]>,
    main_buffer: Box<[T]>,
    length: u32,
}

fn get_test_data<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let length = 1025usize;

    let mut skip_buffer: Vec<T> = vec![T::zero(); length];
    let mut main_buffer: Vec<T> = vec![T::zero(); length];
    let mut expected: Vec<T> = vec![T::zero(); length];

    for i in 0..length {
        skip_buffer[i] = T::from((i as f32).sin() * 30f32).unwrap();
        main_buffer[i] = T::from((i as f32).cos() * 30f32).unwrap();
        expected[i] = skip_buffer[i] + main_buffer[i];
    }

    let input = Input {
        skip_buffer: skip_buffer.into_boxed_slice(),
        main_buffer: main_buffer.into_boxed_slice(),
        length: length as u32,
    };

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> (Vec<T>, Vec<T>) {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TensorAddSwapKernel::new(&context, T::data_type())
        .expect("Failed to create TensorAddSwapKernel");

    let size = input.length as usize;
    let mut skip_allocation = alloc_allocation_with_data::<B, T>(&context, &input.skip_buffer[..size]);
    let mut main_allocation = alloc_allocation_with_data::<B, T>(&context, &input.main_buffer[..size]);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(&mut skip_allocation, &mut main_allocation, input.length, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (
        allocation_to_vec::<B, T>(&skip_allocation),
        allocation_to_vec::<B, T>(&main_allocation),
    )
}

fn test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = get_test_data::<T>();
    for_each_backend!(|B| {
        let (skip_output, main_output) = get_output::<T, B>(&input);
        assert_eq!(
            skip_output,
            expected,
            "skip_buffer results are not equal for backend {}",
            std::any::type_name::<B>()
        );
        assert_eq!(
            main_output,
            expected,
            "main_buffer results are not equal for backend {}",
            std::any::type_name::<B>()
        );
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
