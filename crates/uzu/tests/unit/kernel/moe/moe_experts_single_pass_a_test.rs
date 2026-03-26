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
        common::{Backend, Context, Encoder, Kernels, kernel::MoeExpertsDecodeSinglePassAKernel},
        cpu::Cpu,
    },
};

use crate::common::assert::assert_eq_float;

struct Input<T: ArrayElement + Float> {
    x: Box<[T]>,
    topk_ids: Box<[i32]>,
    w13_all: Box<[T]>,
    up_biases: Box<[T]>,
    d_model: u32,
    d_ff: u32,
    k: u32,
    gating_code: u32,
    silu_alpha: f32,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
}

fn get_test_data<T: ArrayElement + Float>(
    d_model: usize,
    d_ff: usize,
    e: usize,
    k: usize,
    gating_code: u32,
    gate_clip_min: f32,
    gate_clip_max: f32,
    up_clip_min: f32,
    up_clip_max: f32,
    seed: u64,
) -> (Input<T>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let x: Vec<T> = (0..d_model).map(|_| T::from(rng.random_range(-1.0f32..1.0)).unwrap()).collect();
    let topk_ids: Vec<i32> = (0..k).map(|i| (i % e) as i32).collect();
    let w13_all: Vec<T> =
        (0..e * 2 * d_ff * d_model).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let up_biases: Vec<T> = (0..e * 2 * d_ff).map(|_| T::from(rng.random_range(-0.01f32..0.01)).unwrap()).collect();

    let input = Input {
        x: x.into_boxed_slice(),
        topk_ids: topk_ids.into_boxed_slice(),
        w13_all: w13_all.into_boxed_slice(),
        up_biases: up_biases.into_boxed_slice(),
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        k: k as u32,
        gating_code,
        silu_alpha: 1.0,
        gate_clip_min,
        gate_clip_max,
        up_clip_min,
        up_clip_max,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_test_data_basic<T: ArrayElement + Float>(gating_code: u32) -> (Input<T>, Vec<f32>) {
    get_test_data(
        128,
        256,
        8,
        2,
        gating_code,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        0xA001,
    )
}

fn get_test_data_clipped<T: ArrayElement + Float>(gating_code: u32) -> (Input<T>, Vec<f32>) {
    get_test_data(128, 256, 8, 2, gating_code, -5.0, 5.0, -10.0, 10.0, 0xA002)
}

fn get_test_data_k4<T: ArrayElement + Float>(gating_code: u32) -> (Input<T>, Vec<f32>) {
    get_test_data(
        128,
        256,
        8,
        4,
        gating_code,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        0xA003,
    )
}

fn get_test_data_negative_id<T: ArrayElement + Float>() -> (Input<T>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(0xA004);
    let (d_model, d_ff, e, k) = (64, 128, 4, 2);

    let x: Vec<T> = (0..d_model).map(|_| T::from(rng.random_range(-1.0f32..1.0)).unwrap()).collect();
    let topk_ids = vec![1i32, -1];
    let w13_all: Vec<T> =
        (0..e * 2 * d_ff * d_model).map(|_| T::from(rng.random_range(-0.05f32..0.05)).unwrap()).collect();
    let up_biases: Vec<T> = (0..e * 2 * d_ff).map(|_| T::from(rng.random_range(-0.01f32..0.01)).unwrap()).collect();

    let input = Input {
        x: x.into_boxed_slice(),
        topk_ids: topk_ids.into_boxed_slice(),
        w13_all: w13_all.into_boxed_slice(),
        up_biases: up_biases.into_boxed_slice(),
        d_model: d_model as u32,
        d_ff: d_ff as u32,
        k: k as u32,
        gating_code: 2,
        silu_alpha: 1.0,
        gate_clip_min: f32::NEG_INFINITY,
        gate_clip_max: f32::INFINITY,
        up_clip_min: f32::NEG_INFINITY,
        up_clip_max: f32::INFINITY,
    };

    let output = get_output::<Cpu, T>(&input);
    (input, output)
}

fn get_output<B: Backend, T: ArrayElement + Float>(input: &Input<T>) -> Vec<f32> {
    let context = B::Context::new().expect("Failed to create Context");

    let kernel = <<B as Backend>::Kernels as Kernels>::MoeExpertsDecodeSinglePassAKernel::new(
        &context,
        T::data_type(),
        input.gating_code,
    )
    .expect("Failed to create MoeExpertsDecodeSinglePassAKernel");

    let dm = input.d_model as usize;
    let df = input.d_ff as usize;
    let k = input.k as usize;

    let x_array = context.create_array_from(&[dm], &input.x, "x");
    let ids_array = context.create_array_from(&[k], &input.topk_ids, "topk_ids");
    let w13_array = context.create_array_from(&[input.w13_all.len()], &input.w13_all, "w13");
    let bias_array = context.create_array_from(&[input.up_biases.len()], &input.up_biases, "biases");
    let hidden_array = context.create_array_uninitialized(&[k * df], DataType::F32, "hidden");

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        x_array.buffer().borrow().deref(),
        ids_array.buffer().borrow().deref(),
        w13_array.buffer().borrow().deref(),
        bias_array.buffer().borrow().deref(),
        hidden_array.buffer().borrow_mut().deref_mut(),
        input.d_model,
        input.d_ff,
        input.k,
        input.silu_alpha,
        input.gate_clip_min,
        input.gate_clip_max,
        input.up_clip_min,
        input.up_clip_max,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    hidden_array.as_slice::<f32>().to_vec()
}

fn test_internal<T: ArrayElement + Float + Debug + Display>(
    input: &Input<T>,
    expected: &[f32],
) {
    let eps = if matches!(T::data_type(), DataType::F16 | DataType::BF16) {
        0.02f32
    } else {
        1e-4
    };

    for_each_non_cpu_backend!(|B| {
        let output = get_output::<B, T>(input);
        let msg = format!(
            "Pass A failed with backend={}, gating={}, T={}",
            std::any::type_name::<B>(),
            input.gating_code,
            std::any::type_name::<T>(),
        );
        assert_eq_float::<f32>(expected, &output, eps, &msg);
    });
}

fn test_gating<T: ArrayElement + Float + Debug + Display>(gating_code: u32) {
    let (input, expected) = get_test_data_basic::<T>(gating_code);
    test_internal(&input, &expected);
}

// SwiGLU (gating=2)
#[test]
fn test_swiglu_bf16() {
    test_gating::<bf16>(2);
}

#[test]
fn test_swiglu_f16() {
    test_gating::<f16>(2);
}

#[test]
fn test_swiglu_f32() {
    test_gating::<f32>(2);
}

// GELU (gating=0)
#[test]
fn test_gelu_bf16() {
    test_gating::<bf16>(0);
}

#[test]
fn test_gelu_f16() {
    test_gating::<f16>(0);
}

#[test]
fn test_gelu_f32() {
    test_gating::<f32>(0);
}

// SiLU (gating=1)
#[test]
fn test_silu_bf16() {
    test_gating::<bf16>(1);
}

#[test]
fn test_silu_f16() {
    test_gating::<f16>(1);
}

#[test]
fn test_silu_f32() {
    test_gating::<f32>(1);
}

// GEGLU (gating=3)
#[test]
fn test_geglu_bf16() {
    test_gating::<bf16>(3);
}

#[test]
fn test_geglu_f16() {
    test_gating::<f16>(3);
}

#[test]
fn test_geglu_f32() {
    test_gating::<f32>(3);
}

// Clipping
#[test]
fn test_clipping_bf16() {
    let (input, expected) = get_test_data_clipped::<bf16>(2);
    test_internal(&input, &expected);
}

#[test]
fn test_clipping_f32() {
    let (input, expected) = get_test_data_clipped::<f32>(2);
    test_internal(&input, &expected);
}

// K=4
#[test]
fn test_k4_bf16() {
    let (input, expected) = get_test_data_k4::<bf16>(2);
    test_internal(&input, &expected);
}

#[test]
fn test_k4_f32() {
    let (input, expected) = get_test_data_k4::<f32>(2);
    test_internal(&input, &expected);
}

// Negative expert ID
#[test]
fn test_negative_id_bf16() {
    let (input, expected) = get_test_data_negative_id::<bf16>();
    test_internal(&input, &expected);
}

#[test]
fn test_negative_id_f32() {
    let (input, expected) = get_test_data_negative_id::<f32>();
    test_internal(&input, &expected);
}
