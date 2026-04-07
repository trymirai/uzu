use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, kernel::ElementWiseMulStridedKernel},
};

// ---- Strided ElementWiseMulStrided tests ----

struct StridedInput<T: ArrayElement + Float> {
    input_a: Box<[T]>,
    input_b: Box<[T]>,
    ple_dim: u32,
    stride: u32,
    layer_offset: u32,
    rows: u32,
}

fn get_strided_test_data<T: ArrayElement + Float>() -> (StridedInput<T>, Vec<T>) {
    let rows = 4usize;
    let ple_dim = 64usize;
    let num_layers = 3usize;
    let stride = num_layers * ple_dim;
    let target_layer = 1usize;
    let layer_offset = target_layer * ple_dim;

    // input_a is contiguous [rows, ple_dim]
    let mut input_a: Vec<T> = vec![T::zero(); rows * ple_dim];
    // input_b is strided [rows, num_layers * ple_dim]
    let mut input_b: Vec<T> = vec![T::zero(); rows * stride];
    let mut expected: Vec<T> = vec![T::zero(); rows * ple_dim];

    for row in 0..rows {
        for col in 0..ple_dim {
            let a_val = T::from(((row * ple_dim + col) as f32).sin() * 10f32).unwrap();
            input_a[row * ple_dim + col] = a_val;

            // Fill all layers in the strided buffer
            for layer in 0..num_layers {
                let b_val = T::from(((row * stride + layer * ple_dim + col) as f32).cos() * 5f32).unwrap();
                input_b[row * stride + layer * ple_dim + col] = b_val;
            }

            // Expected uses the target_layer slice
            let b_val = input_b[row * stride + layer_offset + col];
            let a_f = a_val.to_f32().unwrap();
            let b_f = b_val.to_f32().unwrap();
            expected[row * ple_dim + col] = T::from(a_f * b_f).unwrap();
        }
    }

    let input = StridedInput {
        input_a: input_a.into_boxed_slice(),
        input_b: input_b.into_boxed_slice(),
        ple_dim: ple_dim as u32,
        stride: stride as u32,
        layer_offset: layer_offset as u32,
        rows: rows as u32,
    };

    (input, expected)
}

fn get_strided_output<T: ArrayElement + Float, B: Backend>(input: &StridedInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::ElementWiseMulStridedKernel::new(&context, T::data_type())
        .expect("Failed to create ElementWiseMulStridedKernel");

    let a_size = (input.rows * input.ple_dim) as usize;
    let b_size = (input.rows * input.stride) as usize;
    let a_array = context.create_array_from(&[a_size], &input.input_a, "");
    let b_array = context.create_array_from(&[b_size], &input.input_b, "");
    let out_array = context.create_array_uninitialized(&[a_size], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        a_array.buffer().borrow().deref(),
        b_array.buffer().borrow().deref(),
        out_array.buffer().borrow_mut().deref_mut(),
        input.ple_dim,
        input.stride,
        input.layer_offset,
        input.rows,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    out_array.as_slice().to_vec()
}

fn test_strided<T: ArrayElement + Float + Debug>() {
    let (input, expected) = get_strided_test_data::<T>();
    for_each_backend!(|B| {
        let output = get_strided_output::<T, B>(&input);
        assert_eq!(
            output,
            expected,
            "ElementWiseMulStrided results are not equal for backend {}",
            std::any::type_name::<B>()
        );
    });
}

#[test]
fn test_strided_f32() {
    test_strided::<f32>();
}

#[test]
fn test_strided_f16() {
    test_strided::<f16>();
}

#[test]
fn test_strided_bf16() {
    test_strided::<bf16>();
}
