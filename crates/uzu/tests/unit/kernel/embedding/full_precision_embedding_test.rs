use std::{
    fmt::Debug,
};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayContextExt, ArrayElement,
    backends::common::{Backend, Context, Encoder, Kernels, kernel::FullPrecisionEmbeddingLookupKernel},
};

use crate::uzu_test;

struct Input<T: ArrayElement + Float> {
    token_ids: Box<[u64]>,
    weights: Box<[T]>,
    batch_size: usize,
    vocab_size: usize,
    model_dim: usize,
    input_scale: f32,
}

fn get_test_data<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let vocab_size = 11;
    let model_dim = 513;
    let batch_size = 7;
    let input_scale = 2.0_f32;

    let token_ids: Box<[u64]> = Box::new([3, 7, 1, 10, 0, 5, 8]);
    let weights: Vec<T> =
        (0..vocab_size * model_dim).map(|i| T::from((i as f32).sin() * 30.0f32).unwrap()).collect::<Vec<_>>();
    let mut expected: Vec<T> = Vec::with_capacity(batch_size * model_dim);
    for &tid in token_ids.iter() {
        let row_start = tid as usize * model_dim;
        for j in 0..model_dim {
            expected.push(weights[row_start + j] * T::from(input_scale).unwrap());
        }
    }

    let input = Input {
        token_ids,
        weights: weights.iter().map(|&v| T::from(v).unwrap()).collect(),
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
    };

    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel =
        <<B as Backend>::Kernels as Kernels>::FullPrecisionEmbeddingLookupKernel::new(&context, T::data_type())
            .expect("Failed to create FullPrecisionEmbeddingLookupKernel");

    let token_ids_array = context.create_array_from(&[input.batch_size], &input.token_ids, "");
    let weights_array = context.create_array_from(&[input.vocab_size, input.model_dim], &input.weights, "");
    let mut output = context
        .create_array_uninitialized(&[input.batch_size, input.model_dim], T::data_type(), "")
        .into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to get encoder");
    kernel.encode(
        token_ids_array.allocation(),
        weights_array.allocation(),
        &mut output,
        input.batch_size as u32,
        input.vocab_size as u32,
        input.model_dim as u32,
        input.input_scale,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    crate::common::helpers::allocation_to_vec(&output)
}

fn test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = get_test_data::<T>();
    for_each_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq!(output, expected, "Results are not equals for backend {}", std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_f32() {
    test::<f32>()
}

#[uzu_test]
fn test_f16() {
    test::<f16>()
}

#[uzu_test]
fn test_bf16() {
    test::<bf16>()
}
