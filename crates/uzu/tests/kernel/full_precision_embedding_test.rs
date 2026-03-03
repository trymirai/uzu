use std::{fmt::Debug, ops::Deref};

use half::{bf16, f16};
use num_traits::Float;
use uzu::{
    ArrayElement,
    array::ArrayContextExt,
    backends::common::{Backend, CommandBuffer, Context, Kernels, kernel::FullPrecisionEmbeddingLookupKernel},
};

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
    let output_array = context.create_array_uninitialized(&[input.batch_size, input.model_dim], T::data_type(), "");

    let mut command_buffer = context.create_command_buffer().expect("Failed to get command buffer");
    command_buffer.with_compute_encoder(|encoder| {
        kernel.encode(
            token_ids_array.buffer().borrow().deref(),
            weights_array.buffer().borrow().deref(),
            output_array.buffer().borrow().deref(),
            input.batch_size as u32,
            input.vocab_size as u32,
            input.model_dim as u32,
            input.input_scale,
            encoder,
        );
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
    test::<f32>()
}

#[test]
fn test_f16() {
    test::<f16>()
}

#[test]
fn test_bf16() {
    test::<bf16>()
}
