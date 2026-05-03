use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{
            Backend, Context, Encoder, Kernels, gpu_types::QuantizationMode, kernel::QuantizedEmbeddingLookupKernel,
        },
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    token_ids: Box<[u64]>,
    weights: Box<[u8]>,
    scales: Box<[T]>,
    biases: Box<[T]>,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
    group_size: u32,
    quant_mode: QuantizationMode,
}

fn get_test_data<T: ArrayElement + Float>(quant_mode: QuantizationMode) -> (Input<T>, Vec<T>) {
    let batch_size = 3u32;
    let vocab_size = 8u32;
    let model_dim = 64u32;
    let group_size = 32u32;
    let input_scale = 1.5f32;

    let token_ids: Box<[u64]> = Box::new([2, 5, 0]);

    let packing_divisor = quant_mode.packing_divisor() as u32;
    let weights_stride = model_dim / packing_divisor;
    let weights: Vec<u8> = (0..vocab_size as usize * weights_stride as usize)
        .map(|i| ((i % 16) as u8) | ((((i + 3) % 16) as u8) * 16))
        .collect();

    let num_groups = (model_dim + group_size - 1) / group_size;
    let scales: Vec<T> = (0..vocab_size as usize * num_groups as usize)
        .map(|i| T::from(0.5 + (i as f32 * 0.1).sin() * 0.3).unwrap())
        .collect();
    let biases: Vec<T> = (0..vocab_size as usize * num_groups as usize)
        .map(|i| T::from((i as f32 * 0.2).cos() * 0.1).unwrap())
        .collect();

    let input = Input {
        token_ids,
        weights: weights.into_boxed_slice(),
        scales: scales.into_boxed_slice(),
        biases: biases.into_boxed_slice(),
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
        group_size,
        quant_mode,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_oob<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let batch_size = 2u32;
    let vocab_size = 4u32;
    let model_dim = 32u32;
    let group_size = 32u32;
    let input_scale = 1.0f32;

    // token_id 1 is valid, token_id 10 is out of bounds
    let token_ids: Box<[u64]> = Box::new([1, 10]);

    let weights: Vec<u8> = (0..vocab_size as usize * model_dim as usize).map(|i| (i % 200) as u8).collect();

    let num_groups = (model_dim + group_size - 1) / group_size;
    let scales: Vec<T> =
        (0..vocab_size as usize * num_groups as usize).map(|i| T::from(0.5 + i as f32 * 0.1).unwrap()).collect();
    let biases: Vec<T> =
        (0..vocab_size as usize * num_groups as usize).map(|i| T::from(i as f32 * 0.01).unwrap()).collect();

    let input = Input {
        token_ids,
        weights: weights.into_boxed_slice(),
        scales: scales.into_boxed_slice(),
        biases: biases.into_boxed_slice(),
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
        group_size,
        quant_mode: QuantizationMode::U8,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
        &context,
        T::data_type(),
        input.group_size,
        input.quant_mode.to_u32(),
    )
    .expect("Failed to create QuantizedEmbeddingLookupKernel");

    let packing_divisor = input.quant_mode.packing_divisor();
    let weights_stride = input.model_dim as usize / packing_divisor;
    let num_groups = (input.model_dim as usize + input.group_size as usize - 1) / input.group_size as usize;

    let token_ids_array = context.create_array_from(&[input.batch_size as usize], &input.token_ids, "");
    let weights_array = context.create_array_from(&[input.vocab_size as usize, weights_stride], &input.weights, "");
    let scales_array = context.create_array_from(&[input.vocab_size as usize, num_groups], &input.scales, "");
    let biases_array = context.create_array_from(&[input.vocab_size as usize, num_groups], &input.biases, "");
    let output_array =
        context.create_array_uninitialized(&[input.batch_size as usize, input.model_dim as usize], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        token_ids_array.buffer().borrow().deref(),
        weights_array.buffer().borrow().deref(),
        scales_array.buffer().borrow().deref(),
        biases_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.batch_size,
        input.vocab_size,
        input.model_dim,
        input.input_scale,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test_quant_mode<T: ArrayElement + Float + Debug + Display>(quant_mode: QuantizationMode) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.5f32
    } else {
        1e-4
    };
    let (input, expected) = get_test_data::<T>(quant_mode);
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("QuantizedEmbeddingLookup {quant_mode:?} test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_oob<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.5f32
    } else {
        1e-4
    };
    let (input, expected) = get_test_data_oob::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("QuantizedEmbeddingLookup OOB test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

// UINT4 tests
#[uzu_test]
fn test_uint4_f32() {
    test_quant_mode::<f32>(QuantizationMode::U4);
}

#[uzu_test]
fn test_uint4_f16() {
    test_quant_mode::<f16>(QuantizationMode::U4);
}

#[uzu_test]
fn test_uint4_bf16() {
    test_quant_mode::<bf16>(QuantizationMode::U4);
}

// INT8 tests
#[uzu_test]
fn test_int8_f32() {
    test_quant_mode::<f32>(QuantizationMode::I8);
}

#[uzu_test]
fn test_int8_f16() {
    test_quant_mode::<f16>(QuantizationMode::I8);
}

#[uzu_test]
fn test_int8_bf16() {
    test_quant_mode::<bf16>(QuantizationMode::I8);
}

// UINT8 tests
#[uzu_test]
fn test_uint8_f32() {
    test_quant_mode::<f32>(QuantizationMode::U8);
}

#[uzu_test]
fn test_uint8_f16() {
    test_quant_mode::<f16>(QuantizationMode::U8);
}

#[uzu_test]
fn test_uint8_bf16() {
    test_quant_mode::<bf16>(QuantizationMode::U8);
}

// OOB tests
#[uzu_test]
fn test_oob_f32() {
    test_oob::<f32>();
}

#[uzu_test]
fn test_oob_f16() {
    test_oob::<f16>();
}

#[uzu_test]
fn test_oob_bf16() {
    test_oob::<bf16>();
}
