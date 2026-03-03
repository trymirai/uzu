use std::{fmt::Debug, ops::Deref};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::common::{Backend, CommandBuffer, Context, Kernels, kernel::TensorAddBiasKernel},
};

struct Input<T: ArrayElement + Float> {
    input: Box<[T]>,
    bias: Box<[T]>,
    num_cols: u32,
    length: u32,
}

fn get_test_data<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let length = 1025usize;
    let num_cols = 129usize;

    let mut input: Vec<T> = vec![T::zero(); length];
    let mut bias: Vec<T> = vec![T::zero(); num_cols];
    let mut expected: Vec<T> = vec![T::zero(); length];

    for i in 0..length {
        input[i] = T::from((i as f32).sin() * 30f32).unwrap();
        if i < num_cols {
            bias[i] = T::from((i as f32).cos() * 30f32).unwrap();
        }
        expected[i] = input[i] + bias[i % num_cols];
    }

    let input = Input {
        input: input.into_boxed_slice(),
        bias: bias.into_boxed_slice(),
        num_cols: num_cols as u32,
        length: length as u32,
    };

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(&context, T::data_type())
        .expect("Failed to create TensorAddBiasKernel");

    let size = input.length as usize;
    let input_array = context.create_array_from(&[size], &input.input, "");
    let bias_array = context.create_array_from(&[input.num_cols as usize], &input.bias, "");
    let output_array = context.create_array_uninitialized(&[size], T::data_type(), "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer");
    command_buffer.with_compute_encoder(|encoder| {
        kernel.encode(
            input_array.buffer().borrow().deref(),
            bias_array.buffer().borrow_mut().deref(),
            output_array.buffer().borrow().deref(),
            input.num_cols,
            input.length,
            encoder,
        )
    });
    command_buffer.submit();
    command_buffer.wait_until_completed();

    output_array.as_slice().to_vec()
}

fn test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = get_test_data::<T>();
    for_each_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq!(output, expected, "Results are not equals for backend {}", std::any::type_name::<B>());
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
