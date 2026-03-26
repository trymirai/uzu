use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
};

use half::{bf16, f16};
use num_traits::Float;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::MoeExpertsDecodeDownFused2DKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    hidden: Box<[f32]>,
    row_expert_map: Box<[u32]>,
    w2_all: Box<[T]>,
    down_biases: Box<[T]>,
    total_rows: u32,
    d_model: u32,
    d_ff: u32,
    e: u32,
}

fn get_test_data<T: ArrayElement + Float>(
    total_rows: usize,
    d_model: usize,
    d_ff: usize,
    e: usize,
    seed: u64,
) -> (Input<T>, Vec<T>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let hidden: Vec<f32> = (0..total_rows * d_ff).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let row_expert_map: Vec<u32> = (0..total_rows).map(|i| (i % e) as u32).collect();
    let w2_all: Vec<T> = (0..e * d_model * d_ff).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let down_biases: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-0.01f32..0.01)).unwrap()).collect();

    let input = Input {
        hidden: hidden.into_boxed_slice(),
        row_expert_map: row_expert_map.into_boxed_slice(),
        w2_all: w2_all.into_boxed_slice(),
        down_biases: down_biases.into_boxed_slice(),
        total_rows: total_rows as u32,
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        e: e as u32,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_test_data_basic<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(4, 128, 256, 8, 0xD001)
}

fn get_test_data_single_row<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(1, 128, 256, 8, 0xD002)
}

fn get_test_data_many_rows<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(16, 128, 256, 8, 0xD003)
}

fn get_test_data_larger<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(4, 256, 512, 8, 0xD004)
}

fn get_test_data_zero_hidden<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let mut rng = StdRng::seed_from_u64(0xD005);
    let (total_rows, d_model, d_ff, e) = (2, 64, 128, 4);

    let hidden = vec![0.0f32; total_rows * d_ff];
    let row_expert_map = vec![0u32, 1];
    let w2_all: Vec<T> = (0..e * d_model * d_ff).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let down_biases: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-0.1f32..0.1)).unwrap()).collect();

    let input = Input {
        hidden: hidden.into_boxed_slice(),
        row_expert_map: row_expert_map.into_boxed_slice(),
        w2_all: w2_all.into_boxed_slice(),
        down_biases: down_biases.into_boxed_slice(),
        total_rows: total_rows as u32,
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        e: e as u32,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_test_data_same_expert<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let mut rng = StdRng::seed_from_u64(0xD006);
    let (total_rows, d_model, d_ff, e) = (4, 128, 256, 8);

    let hidden: Vec<f32> = (0..total_rows * d_ff).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let row_expert_map = vec![3u32; total_rows];
    let w2_all: Vec<T> = (0..e * d_model * d_ff).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let down_biases: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-0.01f32..0.01)).unwrap()).collect();

    let input = Input {
        hidden: hidden.into_boxed_slice(),
        row_expert_map: row_expert_map.into_boxed_slice(),
        w2_all: w2_all.into_boxed_slice(),
        down_biases: down_biases.into_boxed_slice(),
        total_rows: total_rows as u32,
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        e: e as u32,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::MoeExpertsDecodeDownFused2DKernel::new(
        &context,
        T::data_type(),
        DataType::F32,
    )
    .expect("Failed to create MoeExpertsDecodeDownFused2DKernel");

    let total = input.total_rows as usize;
    let dm = input.d_model as usize;
    let df = input.d_ff as usize;

    let hidden_array = context.create_array_from(&[total * df], &input.hidden, "hidden");
    let map_array = context.create_array_from(&[total], &input.row_expert_map, "row_expert_map");
    let w2_array = context.create_array_from(&[input.w2_all.len()], &input.w2_all, "w2");
    let bias_array = context.create_array_from(&[input.down_biases.len()], &input.down_biases, "biases");
    let y_array = context.create_array_uninitialized(&[total * dm], T::data_type(), "y_out");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        hidden_array.buffer().borrow().deref(),
        map_array.buffer().borrow().deref(),
        w2_array.buffer().borrow().deref(),
        bias_array.buffer().borrow().deref(),
        y_array.buffer().borrow_mut().deref_mut(),
        input.total_rows,
        input.d_model,
        input.d_ff,
        input.e,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    y_array.as_slice::<T>().to_vec()
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[T],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        1e-3
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let msg = format!(
            "DownFused2D failed with backend={}, T={}",
            std::any::type_name::<B>(),
            std::any::type_name::<T>(),
        );
        assert_eq_float::<T>(expected, &output, eps, &msg);
    });
}

fn test_basic<T: ArrayElement + Float + Debug + Display>() {
    let (input, expected) = get_test_data_basic::<T>();
    test_internal(&input, &expected);
}

// Basic tests
#[test]
fn test_basic_bf16() {
    test_basic::<bf16>();
}

#[test]
fn test_basic_f16() {
    test_basic::<f16>();
}

#[test]
fn test_basic_f32() {
    test_basic::<f32>();
}

// Single row
#[test]
fn test_single_row_bf16() {
    let (input, expected) = get_test_data_single_row::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_single_row_f32() {
    let (input, expected) = get_test_data_single_row::<f32>();
    test_internal(&input, &expected);
}

// Many rows
#[test]
fn test_many_rows_bf16() {
    let (input, expected) = get_test_data_many_rows::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_many_rows_f32() {
    let (input, expected) = get_test_data_many_rows::<f32>();
    test_internal(&input, &expected);
}

// Larger dimensions
#[test]
fn test_larger_bf16() {
    let (input, expected) = get_test_data_larger::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_larger_f32() {
    let (input, expected) = get_test_data_larger::<f32>();
    test_internal(&input, &expected);
}

// Zero hidden
#[test]
fn test_zero_hidden_bf16() {
    let (input, expected) = get_test_data_zero_hidden::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_zero_hidden_f32() {
    let (input, expected) = get_test_data_zero_hidden::<f32>();
    test_internal(&input, &expected);
}

// Same expert for all rows
#[test]
fn test_same_expert_bf16() {
    let (input, expected) = get_test_data_same_expert::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_same_expert_f32() {
    let (input, expected) = get_test_data_same_expert::<f32>();
    test_internal(&input, &expected);
}
