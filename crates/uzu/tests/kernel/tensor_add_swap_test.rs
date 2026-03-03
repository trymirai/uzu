use std::{fmt::Debug, ops::Deref};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::common::{Backend, CommandBuffer, Context, Kernels, kernel::TensorAddSwapKernel},
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
    let skip_array = context.create_array_from(&[size], &input.skip_buffer, "");
    let main_array = context.create_array_from(&[size], &input.main_buffer, "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to create command buffer");
    command_buffer.with_compute_encoder(|encoder| {
        kernel.encode(
            skip_array.buffer().borrow_mut().deref(),
            main_array.buffer().borrow_mut().deref(),
            input.length,
            encoder,
        )
    });
    command_buffer.submit();
    command_buffer.wait_until_completed();

    (skip_array.as_slice().to_vec(), main_array.as_slice().to_vec())
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
