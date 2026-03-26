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
        common::{Backend, Context, Encoder, Kernels, kernel::MoeExpertsDecodeSinglePassBKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    hidden: Box<[f32]>,
    topk_ids: Box<[i32]>,
    topk_probs: Box<[T]>,
    w2_all: Box<[T]>,
    down_biases: Box<[T]>,
    d_model: u32,
    d_ff: u32,
    k: u32,
}

fn get_test_data<T: ArrayElement + Float>(
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    seed: u64,
) -> (Input<T>, Vec<T>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let hidden: Vec<f32> = (0..k * d_ff).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let topk_ids: Vec<i32> = (0..k).map(|i| (i % e) as i32).collect();
    let topk_probs: Vec<T> = {
        let raw: Vec<f32> = (0..k).map(|_| rng.random_range(0.1f32..1.0)).collect();
        let sum: f32 = raw.iter().sum();
        raw.iter().map(|p| T::from(p / sum).unwrap()).collect()
    };
    let w2_all: Vec<T> = (0..e * d_ff * d_model).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let down_biases: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-0.01f32..0.01)).unwrap()).collect();

    let input = Input {
        hidden: hidden.into_boxed_slice(),
        topk_ids: topk_ids.into_boxed_slice(),
        topk_probs: topk_probs.into_boxed_slice(),
        w2_all: w2_all.into_boxed_slice(),
        down_biases: down_biases.into_boxed_slice(),
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        k: k as u32,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_test_data_basic<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(128, 256, 8, 2, 0xB001)
}

fn get_test_data_k1<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(128, 256, 8, 1, 0xB002)
}

fn get_test_data_k4<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(128, 256, 8, 4, 0xB003)
}

fn get_test_data_larger<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    get_test_data(256, 512, 8, 2, 0xB004)
}

fn get_test_data_zero_hidden<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let mut rng = StdRng::seed_from_u64(0xB005);
    let (d_model, d_ff, e, k) = (64, 128, 4, 2);

    let hidden = vec![0.0f32; k * d_ff];
    let topk_ids = vec![0i32, 1];
    let topk_probs = vec![T::from(0.5f32).unwrap(); k];
    let w2_all: Vec<T> = (0..e * d_ff * d_model).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let down_biases: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-0.1f32..0.1)).unwrap()).collect();

    let input = Input {
        hidden: hidden.into_boxed_slice(),
        topk_ids: topk_ids.into_boxed_slice(),
        topk_probs: topk_probs.into_boxed_slice(),
        w2_all: w2_all.into_boxed_slice(),
        down_biases: down_biases.into_boxed_slice(),
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        k: k as u32,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_test_data_dominant_expert<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let mut rng = StdRng::seed_from_u64(0xB006);
    let (d_model, d_ff, e, k) = (128, 256, 8, 2);

    let hidden: Vec<f32> = (0..k * d_ff).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let topk_ids = vec![3i32, 7];
    let topk_probs = vec![T::from(0.99f32).unwrap(), T::from(0.01f32).unwrap()];
    let w2_all: Vec<T> = (0..e * d_ff * d_model).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let down_biases: Vec<T> = (0..e * d_model).map(|_| T::from(rng.random_range(-0.01f32..0.01)).unwrap()).collect();

    let input = Input {
        hidden: hidden.into_boxed_slice(),
        topk_ids: topk_ids.into_boxed_slice(),
        topk_probs: topk_probs.into_boxed_slice(),
        w2_all: w2_all.into_boxed_slice(),
        down_biases: down_biases.into_boxed_slice(),
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        k: k as u32,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::MoeExpertsDecodeSinglePassBKernel::new(&context, T::data_type())
        .expect("Failed to create MoeExpertsDecodeSinglePassBKernel");

    let dm = input.d_model as usize;
    let df = input.d_ff as usize;
    let k = input.k as usize;

    let hidden_array = context.create_array_from(&[k * df], &input.hidden, "hidden");
    let ids_array = context.create_array_from(&[k], &input.topk_ids, "topk_ids");
    let probs_array = context.create_array_from(&[k], &input.topk_probs, "topk_probs");
    let w2_array = context.create_array_from(&[input.w2_all.len()], &input.w2_all, "w2");
    let bias_array = context.create_array_from(&[input.down_biases.len()], &input.down_biases, "biases");
    let y_array = context.create_array_uninitialized(&[dm], T::data_type(), "y");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        hidden_array.buffer().borrow().deref(),
        ids_array.buffer().borrow().deref(),
        probs_array.buffer().borrow().deref(),
        w2_array.buffer().borrow().deref(),
        bias_array.buffer().borrow().deref(),
        y_array.buffer().borrow_mut().deref_mut(),
        input.d_model,
        input.d_ff,
        input.k,
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
        0.02f32
    } else {
        1e-4
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let msg =
            format!("Pass B failed with backend={}, T={}", std::any::type_name::<B>(), std::any::type_name::<T>(),);
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

// K=1
#[test]
fn test_k1_bf16() {
    let (input, expected) = get_test_data_k1::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_k1_f32() {
    let (input, expected) = get_test_data_k1::<f32>();
    test_internal(&input, &expected);
}

// K=4
#[test]
fn test_k4_bf16() {
    let (input, expected) = get_test_data_k4::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_k4_f32() {
    let (input, expected) = get_test_data_k4::<f32>();
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

// Dominant expert
#[test]
fn test_dominant_expert_bf16() {
    let (input, expected) = get_test_data_dominant_expert::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_dominant_expert_f32() {
    let (input, expected) = get_test_data_dominant_expert::<f32>();
    test_internal(&input, &expected);
}
