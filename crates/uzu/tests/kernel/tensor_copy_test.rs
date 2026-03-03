use std::{fmt::Debug, ops::Deref};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::common::{Backend, CommandBuffer, Context, Kernels, kernel::TensorCopyKernel},
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
    let src_array = context.create_array_from(&[size], &input.src, "");
    let dst_array = context.create_array_uninitialized(&[size], T::data_type(), "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer");
    command_buffer.with_compute_encoder(|encoder| {
        kernel.encode(src_array.buffer().borrow().deref(), dst_array.buffer().borrow().deref(), input.length, encoder)
    });
    command_buffer.submit();
    command_buffer.wait_until_completed();

    dst_array.as_slice().to_vec()
}

fn test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = get_test_data::<T>();
    for_each_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq!(output, expected, "Results are not equal for backend {}", std::any::type_name::<B>());
    });
}

#[test]
fn test_f32() {
    test::<f32>();
}

#[test]
fn test_f16() {
    test::<f16>();
}

#[test]
fn test_bf16() {
    test::<bf16>();
}
