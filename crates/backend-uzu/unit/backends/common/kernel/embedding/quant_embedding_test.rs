use std::fmt::{Debug, Display};

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            gpu_types::{QuantizationMethod, QuantizationMode},
            kernel::QuantizedEmbeddingLookupKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::assert::assert_eq_float,
};

struct Input<T: ArrayElement + Float> {
    token_ids: Box<[u64]>,
    weights: Box<[u8]>,
    scales: Box<[T]>,
    zero_points: Option<Box<[u8]>>,
    biases: Option<Box<[T]>>,
    batch_size: u32,
    vocab_size: u32,
    model_dim: u32,
    input_scale: f32,
    group_size: u32,
    quant_mode: QuantizationMode,
    quant_method: QuantizationMethod,
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

    let num_groups = model_dim.div_ceil(group_size);
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
        zero_points: None,
        biases: Some(biases.into_boxed_slice()),
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
        group_size,
        quant_mode,
        quant_method: QuantizationMethod::ScaleBias,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_zero_point_group16<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let batch_size = 3u32;
    let vocab_size = 8u32;
    let model_dim = 64u32;
    let group_size = 16u32;
    let input_scale = 1.5f32;
    let quant_mode = QuantizationMode::U4;

    let token_ids: Box<[u64]> = Box::new([2, 5, 0]);

    let packing_divisor = quant_mode.packing_divisor() as u32;
    let weights_stride = model_dim / packing_divisor;
    let weights: Vec<u8> = (0..vocab_size as usize * weights_stride as usize)
        .map(|i| ((i % 16) as u8) | ((((i + 3) % 16) as u8) * 16))
        .collect();

    let num_groups = model_dim.div_ceil(group_size);
    let scales: Vec<T> = (0..vocab_size as usize * num_groups as usize)
        .map(|i| T::from(0.5 + (i as f32 * 0.1).sin() * 0.3).unwrap())
        .collect();
    let zero_point_stride = num_groups.div_ceil(packing_divisor);
    let zero_points: Vec<u8> = (0..vocab_size as usize * zero_point_stride as usize)
        .map(|i| ((i % 16) as u8) | ((((i + 5) % 16) as u8) * 16))
        .collect();

    let input = Input {
        token_ids,
        weights: weights.into_boxed_slice(),
        scales: scales.into_boxed_slice(),
        zero_points: Some(zero_points.into_boxed_slice()),
        biases: None,
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
        group_size,
        quant_mode,
        quant_method: QuantizationMethod::ScaleZeroPoint,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_symmetric_u8<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let batch_size = 3u32;
    let vocab_size = 8u32;
    let model_dim = 64u32;
    let group_size = 32u32;
    let input_scale = 1.5f32;
    let quant_mode = QuantizationMode::U8;

    let token_ids: Box<[u64]> = Box::new([2, 5, 0]);

    let weights: Vec<u8> =
        (0..vocab_size as usize * model_dim as usize).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();

    let num_groups = model_dim.div_ceil(group_size);
    let scales: Vec<T> = (0..vocab_size as usize * num_groups as usize)
        .map(|i| T::from(0.5 + (i as f32 * 0.1).sin() * 0.3).unwrap())
        .collect();

    let input = Input {
        token_ids,
        weights: weights.into_boxed_slice(),
        scales: scales.into_boxed_slice(),
        zero_points: None,
        biases: None,
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
        group_size,
        quant_mode,
        quant_method: QuantizationMethod::ScaleSymmetric,
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

    let num_groups = model_dim.div_ceil(group_size);
    let scales: Vec<T> =
        (0..vocab_size as usize * num_groups as usize).map(|i| T::from(0.5 + i as f32 * 0.1).unwrap()).collect();
    let biases: Vec<T> =
        (0..vocab_size as usize * num_groups as usize).map(|i| T::from(i as f32 * 0.01).unwrap()).collect();

    let input = Input {
        token_ids,
        weights: weights.into_boxed_slice(),
        scales: scales.into_boxed_slice(),
        zero_points: None,
        biases: Some(biases.into_boxed_slice()),
        batch_size,
        vocab_size,
        model_dim,
        input_scale,
        group_size,
        quant_mode: QuantizationMode::U8,
        quant_method: QuantizationMethod::ScaleBias,
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
        input.quant_mode,
        input.quant_method,
        false,
    )
    .expect("Failed to create QuantizedEmbeddingLookupKernel");

    let packing_divisor = input.quant_mode.packing_divisor();
    let weights_stride = input.model_dim as usize / packing_divisor;
    let num_groups = (input.model_dim as usize).div_ceil(input.group_size as usize);

    let token_ids_array = context.create_array_from(&[input.batch_size as usize], &input.token_ids);
    let weights_array = context.create_array_from(&[input.vocab_size as usize, weights_stride], &input.weights);
    let scales_array = context.create_array_from(&[input.vocab_size as usize, num_groups], &input.scales);
    let zero_point_stride = match input.quant_mode {
        QuantizationMode::U4 => num_groups.div_ceil(2),
        QuantizationMode::I8 | QuantizationMode::U8 => num_groups,
    };
    let zero_points_array = input
        .zero_points
        .as_ref()
        .map(|zero_points| context.create_array_from(&[input.vocab_size as usize, zero_point_stride], zero_points));
    let biases_array =
        input.biases.as_ref().map(|biases| context.create_array_from(&[input.vocab_size as usize, num_groups], biases));
    let mut output = context
        .create_array_uninitialized(&[input.batch_size as usize, input.model_dim as usize], T::data_type())
        .into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        token_ids_array.allocation(),
        weights_array.allocation(),
        scales_array.allocation(),
        zero_points_array.as_ref().map(|array| array.allocation()),
        biases_array.as_ref().map(|array| array.allocation()),
        &mut output,
        None::<&Allocation<B>>,
        input.batch_size,
        input.vocab_size,
        input.model_dim,
        input.input_scale,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    crate::tests::helpers::allocation_to_vec(&output)
}

fn test_zero_point_group16<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.5f32
    } else {
        1e-4
    };
    let (input, expected) = get_test_data_zero_point_group16::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!(
                "QuantizedEmbeddingLookup ScaleZeroPoint group16 test failed for backend {}",
                std::any::type_name::<B>()
            ),
        );
    });
}

fn test_symmetric_u8<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.5f32
    } else {
        1e-4
    };
    let (input, expected) = get_test_data_symmetric_u8::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("QuantizedEmbeddingLookup ScaleSymmetric test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

#[cfg(metal_backend)]
fn test_zero_point_group16_hadamard_constructor<T: ArrayElement + Float + Debug + Display>() {
    use crate::backends::metal::Metal;

    let context = <Metal as Backend>::Context::new().expect("Metal context");
    <<Metal as Backend>::Kernels as Kernels>::QuantizedEmbeddingLookupKernel::new(
        &context,
        T::data_type(),
        16,
        QuantizationMode::U4,
        QuantizationMethod::ScaleZeroPoint,
        true,
    )
    .expect("QuantizedEmbeddingLookupKernel group16 zero-point hadamard");
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

#[uzu_test]
fn test_uint4_zero_point_group16_bf16() {
    test_zero_point_group16::<bf16>();
}

#[cfg(metal_backend)]
#[uzu_test]
fn test_uint4_zero_point_group16_hadamard_bf16_constructor() {
    test_zero_point_group16_hadamard_constructor::<bf16>();
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

#[uzu_test]
fn test_uint8_symmetric_bf16() {
    test_symmetric_u8::<bf16>();
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
