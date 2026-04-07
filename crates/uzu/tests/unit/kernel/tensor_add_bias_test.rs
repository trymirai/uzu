use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, kernel::TensorAddBiasKernel},
};

use crate::uzu_test;

struct Input<T: ArrayElement + Float> {
    input: Box<[T]>,
    bias: Box<[T]>,
    output: Box<[T]>,
    num_cols: u32,
    length: u32,
}

fn get_test_data<T: ArrayElement + Float>(in_place: bool) -> (Input<T>, Vec<T>) {
    let length = 1025usize;
    let num_cols = 129usize;

    let mut input: Vec<T> = vec![T::zero(); length];
    let mut bias: Vec<T> = vec![T::zero(); num_cols];
    let mut output: Vec<T> = vec![T::zero(); length];
    let mut expected: Vec<T> = vec![T::zero(); length];

    for i in 0..length {
        input[i] = T::from((i as f32).sin() * 30f32).unwrap();
        if i < num_cols {
            bias[i] = T::from((i as f32).cos() * 30f32).unwrap();
        }
        output[i] = input[i];

        if in_place {
            expected[i] = output[i] + bias[i % num_cols];
        } else {
            expected[i] = input[i] + bias[i % num_cols];
        }
    }

    let input = Input {
        input: input.into_boxed_slice(),
        bias: bias.into_boxed_slice(),
        output: output.into_boxed_slice(),
        num_cols: num_cols as u32,
        length: length as u32,
    };

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(
    input: &Input<T>,
    in_place: bool,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::TensorAddBiasKernel::new(&context, T::data_type(), in_place)
        .expect("Failed to create TensorAddBiasKernel");

    let size = input.length as usize;
    let input_array = context.create_array_from(&[size], &input.input, "");
    let input_array_buffer_rc = input_array.buffer();
    let input_array_borrow = input_array_buffer_rc.borrow();
    let input_array_deref = input_array_borrow.deref();
    let input_buffer = (!in_place).then(|| input_array_deref);

    let bias_array = context.create_array_from(&[input.num_cols as usize], &input.bias, "");
    let output_array = match in_place {
        true => context.create_array_from(&[size], &input.output, ""),
        false => context.create_array_uninitialized(&[size], T::data_type(), ""),
    };

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        input_buffer,
        bias_array.buffer().borrow_mut().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.num_cols,
        input.length,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test<T: ArrayElement + Float + Debug>(in_place: bool) {
    let (input, expected) = get_test_data::<T>(in_place);
    for_each_backend!(|B| {
        let output = get_output::<T, B>(&input, in_place);
        assert_eq!(output, expected, "Results are not equals for backend {}", std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_f32() {
    test::<f32>(false);
}

#[uzu_test]
fn test_f32_in_place() {
    test::<f32>(true);
}

#[uzu_test]
fn test_f16() {
    test::<f16>(false);
}

#[uzu_test]
fn test_f16_in_place() {
    test::<f16>(true);
}

#[uzu_test]
fn test_bf16() {
    test::<bf16>(false);
}

#[uzu_test]
fn test_bf16_in_place() {
    test::<bf16>(true);
}
