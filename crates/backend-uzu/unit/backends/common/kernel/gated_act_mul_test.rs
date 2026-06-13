use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::GatedActMulKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        for_each_backend, for_each_non_cpu_backend,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

struct InterleavedInput<T: ArrayElement + Float> {
    fused_up: Box<[T]>,
    gated_dim: u32,
    batch_dim: u32,
    act_type: ActivationType,
}

fn interleaved_input<T: ArrayElement + Float>(act_type: ActivationType) -> InterleavedInput<T> {
    let gated_dim = 64u32;
    let batch_dim = 4u32;
    let fused_length = (batch_dim * 2 * gated_dim) as usize;
    let mut fused_up: Vec<T> = vec![T::zero(); fused_length];
    for index in 0..fused_length {
        fused_up[index] = T::from((index as f32 * 0.1).sin() * 2.0f32).unwrap();
    }
    InterleavedInput {
        fused_up: fused_up.into_boxed_slice(),
        gated_dim,
        batch_dim,
        act_type,
    }
}

fn run_interleaved<T: ArrayElement + Float, B: Backend>(input: &InterleavedInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(&context, T::data_type(), true, false)
        .expect("create GatedActMulKernel");

    let fused_length = (input.batch_dim * 2 * input.gated_dim) as usize;
    let output_length = (input.batch_dim * input.gated_dim) as usize;
    let fused_up = alloc_allocation_with_data::<B, T>(&context, &input.fused_up[..fused_length]);
    let mut output = alloc_allocation::<B, T>(&context, output_length);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &fused_up,
        None::<&Allocation<B>>,
        &mut output,
        None::<&Allocation<B>>,
        input.gated_dim,
        input.batch_dim,
        0,
        0,
        input.act_type,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&output)
}

fn interleaved_test<T: ArrayElement + Float + Debug + Display>(act_type: ActivationType) {
    let eps = if matches!(T::data_type(), DataType::BF16) {
        0.02f32
    } else {
        1e-5
    };
    let input = interleaved_input::<T>(act_type);
    let expected = run_interleaved::<T, Cpu>(&input);
    for_each_non_cpu_backend!(|B| {
        let output = run_interleaved::<T, B>(&input);
        let message = format!("interleaved mismatch for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &message);
    });
}

#[uzu_test]
fn test_gated_act_mul_interleaved_silu_f32() {
    interleaved_test::<f32>(ActivationType::SILU);
}

#[uzu_test]
fn test_gated_act_mul_interleaved_silu_bf16() {
    interleaved_test::<bf16>(ActivationType::SILU);
}

#[uzu_test]
fn test_gated_act_mul_interleaved_gelu_f32() {
    interleaved_test::<f32>(ActivationType::GELUApprox);
}

#[uzu_test]
fn test_gated_act_mul_interleaved_gelu_bf16() {
    interleaved_test::<bf16>(ActivationType::GELUApprox);
}

#[uzu_test]
fn test_gated_act_mul_interleaved_gelu_exact_f32() {
    interleaved_test::<f32>(ActivationType::GELUExact);
}

struct SeparateInput<T: ArrayElement + Float> {
    gate_out: Box<[T]>,
    per_layer_input: Box<[T]>,
    gated_dim: u32,
    batch_dim: u32,
    value_offset: u32,
    value_row_stride: u32,
    act_type: ActivationType,
}

fn separate_input<T: ArrayElement + Float>() -> (SeparateInput<T>, Vec<T>) {
    let gate_out = [1.0_f32, 2.0, 3.0, 4.0].into_iter().map(|value| T::from(value).unwrap()).collect::<Vec<_>>();
    let per_layer_input = [0.0_f32, 0.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 50.0, 60.0, 70.0, 80.0]
        .into_iter()
        .map(|value| T::from(value).unwrap())
        .collect::<Vec<_>>();
    let expected = [10.0_f32, 40.0, 150.0, 240.0].into_iter().map(|value| T::from(value).unwrap()).collect::<Vec<_>>();

    // ple_dim=2, batch=2, num_layers=3, layer_index=1 -> value_offset=2, value_row_stride=6
    (
        SeparateInput {
            gate_out: gate_out.into_boxed_slice(),
            per_layer_input: per_layer_input.into_boxed_slice(),
            gated_dim: 2,
            batch_dim: 2,
            value_offset: 2,
            value_row_stride: 6,
            act_type: ActivationType::IDENTITY,
        },
        expected,
    )
}

fn run_separate<T: ArrayElement + Float, B: Backend>(input: &SeparateInput<T>) -> Vec<T> {
    let context = B::Context::new().expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(&context, T::data_type(), false, false)
        .expect("create GatedActMulKernel");

    let gate_out = alloc_allocation_with_data::<B, T>(&context, &input.gate_out);
    let per_layer_input = alloc_allocation_with_data::<B, T>(&context, &input.per_layer_input);
    let mut output = alloc_allocation::<B, T>(&context, (input.batch_dim * input.gated_dim) as usize);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &gate_out,
        Some(&per_layer_input),
        &mut output,
        None::<&Allocation<B>>,
        input.gated_dim,
        input.batch_dim,
        input.value_offset,
        input.value_row_stride,
        input.act_type,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&output)
}

fn separate_test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = separate_input::<T>();
    for_each_backend!(|B| {
        let output = run_separate::<T, B>(&input);
        assert_eq!(expected, output, "separate mismatch for backend {}", std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_gated_act_mul_separate_f32() {
    separate_test::<f32>();
}

#[uzu_test]
fn test_gated_act_mul_separate_bf16() {
    separate_test::<bf16>();
}
