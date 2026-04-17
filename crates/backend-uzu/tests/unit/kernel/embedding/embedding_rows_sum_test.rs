use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::EmbeddingRowsSumKernel},
        cpu::Cpu,
    },
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{common::assert::assert_eq_float, uzu_test};

struct Input<T: ArrayElement + Float> {
    token_indices: Box<[u32]>,
    weights: Box<[T]>,
    num_rows: u32,
    total_rows: u32,
    model_dim: u32,
    codebook_stride: u32,
}

fn get_test_data_basic<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let num_rows = 4u32;
    let codebook_size = 8u32;
    let codebook_stride = codebook_size;
    let total_rows = num_rows * codebook_size;
    let model_dim = 513u32;

    let token_indices: Box<[u32]> = Box::new([3, 7, 1, 5]);
    let weights: Vec<T> = (0..total_rows as usize * model_dim as usize)
        .map(|i| T::from((i as f32 * 0.1).sin() * 10.0).unwrap())
        .collect();

    let input = Input {
        token_indices,
        weights: weights.into_boxed_slice(),
        num_rows,
        total_rows,
        model_dim,
        codebook_stride,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_edge<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    // Single row, small dim
    let num_rows = 1u32;
    let codebook_stride = 4u32;
    let total_rows = 4u32;
    let model_dim = 4u32;

    let token_indices: Box<[u32]> = Box::new([2]);
    let weights: Vec<T> =
        (0..total_rows as usize * model_dim as usize).map(|i| T::from(i as f32 + 1.0).unwrap()).collect();

    let input = Input {
        token_indices,
        weights: weights.into_boxed_slice(),
        num_rows,
        total_rows,
        model_dim,
        codebook_stride,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_test_data_oob<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    // Token index causes row to exceed total_rows, should be skipped
    let num_rows = 2u32;
    let codebook_stride = 4u32;
    let total_rows = 4u32;
    let model_dim = 4u32;

    // row_idx=0: row = 0*4+1 = 1 (valid)
    // row_idx=1: row = 1*4+3 = 7 (out of bounds, total_rows=4)
    let token_indices: Box<[u32]> = Box::new([1, 3]);
    let weights: Vec<T> =
        (0..total_rows as usize * model_dim as usize).map(|i| T::from(i as f32 + 1.0).unwrap()).collect();

    let input = Input {
        token_indices,
        weights: weights.into_boxed_slice(),
        num_rows,
        total_rows,
        model_dim,
        codebook_stride,
    };

    let expected = get_output::<T, Cpu>(&input);
    (input, expected)
}

fn get_output<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::EmbeddingRowsSumKernel::new(&context, T::data_type())
        .expect("Failed to create EmbeddingRowsSumKernel");

    let token_indices_array = context.create_array_from(&[input.num_rows as usize], &input.token_indices, "");
    let weights_array =
        context.create_array_from(&[input.total_rows as usize, input.model_dim as usize], &input.weights, "");
    let output_array = context.create_array_uninitialized(&[input.model_dim as usize], T::data_type(), "");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        token_indices_array.buffer().borrow().deref(),
        weights_array.buffer().borrow().deref(),
        output_array.buffer().borrow_mut().deref_mut(),
        input.num_rows,
        input.total_rows,
        input.model_dim,
        input.codebook_stride,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    output_array.as_slice().to_vec()
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.1f32
    } else {
        1e-4
    };
    let (input, expected) = get_test_data_basic::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("EmbeddingRowsSum basic test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_edge<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.05f32
    } else {
        1e-5
    };
    let (input, expected) = get_test_data_edge::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("EmbeddingRowsSum edge test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

fn test_oob<T: ArrayElement + Float + Debug + Display>() {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.05f32
    } else {
        1e-5
    };
    let (input, expected) = get_test_data_oob::<T>();
    for_each_non_cpu_backend!(|B| {
        let output = get_output::<T, B>(&input);
        assert_eq_float::<T>(
            &expected,
            &output,
            eps,
            &format!("EmbeddingRowsSum oob test failed for backend {}", std::any::type_name::<B>()),
        );
    });
}

// basic tests
#[uzu_test]
fn test_basic_f32() {
    test_basic::<f32>();
}

#[uzu_test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[uzu_test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

// edge tests
#[uzu_test]
fn test_edge_f32() {
    test_edge::<f32>();
}

#[uzu_test]
fn test_edge_f16() {
    test_edge::<f16>();
}

#[uzu_test]
fn test_edge_bf16() {
    test_edge::<bf16>();
}

// out-of-bounds tests
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
