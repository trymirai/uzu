use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use proc_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder, Kernels,
            gpu_types::{ActivationType, GatedActMulOp},
            kernel::{ActivationTransform, GatedActMulKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{
            alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_backend, for_each_non_cpu_backend,
        },
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
    let kernel = <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(
        &context,
        T::data_type(),
        GatedActMulOp::FullPrecision,
        true,
        false,
        32,
        32,
    )
    .expect("create GatedActMulKernel");

    let fused_length = (input.batch_dim * 2 * input.gated_dim) as usize;
    let output_length = (input.batch_dim * input.gated_dim) as usize;
    let fused_up = alloc_allocation_with_data::<B, T>(&context, &input.fused_up[..fused_length]);
    let mut output = alloc_allocation::<B, T>(&context, output_length);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &fused_up,
        None::<&Allocation<B>>,
        Some(&mut output),
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
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

struct QuantizedInterleavedInput {
    fused_up: Box<[f32]>,
    factors: Box<[i32]>,
    gated_dim: u32,
    batch_dim: u32,
}

fn quantized_interleaved_input() -> QuantizedInterleavedInput {
    let gated_dim = 128u32;
    let batch_dim = 3u32;
    let fused_length = (batch_dim * 2 * gated_dim) as usize;
    let fused_up =
        (0..fused_length).map(|index| ((index as f32) * 0.017).sin() * 2.0).collect::<Vec<_>>().into_boxed_slice();
    let factors = (0..gated_dim as usize)
        .map(|index| {
            if index % 3 == 0 {
                -1
            } else {
                1
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    QuantizedInterleavedInput {
        fused_up,
        factors,
        gated_dim,
        batch_dim,
    }
}

fn run_unfused_quantized(input: &QuantizedInterleavedInput) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let context = <Cpu as Backend>::Context::new().expect("create context");
    let fused_up = alloc_allocation_with_data::<Cpu, f32>(&context, &input.fused_up);
    let factors = alloc_allocation_with_data::<Cpu, i32>(&context, &input.factors);
    let mut hidden = alloc_allocation::<Cpu, f32>(&context, (input.batch_dim * input.gated_dim) as usize);
    let mut values = alloc_allocation::<Cpu, i8>(&context, (input.batch_dim * input.gated_dim) as usize);
    let mut scales = alloc_allocation::<Cpu, f32>(&context, (input.batch_dim * input.gated_dim / 128) as usize);
    let mut group_sums = alloc_allocation::<Cpu, i32>(&context, (input.batch_dim * input.gated_dim / 32) as usize);

    let gate = <<Cpu as Backend>::Kernels as Kernels>::GatedActMulKernel::new(
        &context,
        f32::data_type(),
        GatedActMulOp::FullPrecision,
        true,
        false,
        32,
        32,
    )
    .expect("create GatedActMulKernel");
    let quantize = ActivationTransform::quantize(context.as_ref(), f32::data_type(), 128, Some(32)).expect("quantize");
    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    gate.encode(
        &fused_up,
        None::<&Allocation<Cpu>>,
        Some(&mut hidden),
        None::<&mut Allocation<Cpu>>,
        None::<&mut Allocation<Cpu>>,
        None::<&mut Allocation<Cpu>>,
        None::<&Allocation<Cpu>>,
        input.gated_dim,
        input.batch_dim,
        0,
        0,
        ActivationType::SILU,
        &mut encoder,
    );
    quantize.encode_quantize(
        &hidden,
        &mut values,
        &mut scales,
        Some(&mut group_sums),
        &factors,
        input.batch_dim,
        input.gated_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&values), allocation_to_vec(&scales), allocation_to_vec(&group_sums))
}

fn run_fused_quantized<B: Backend>(input: &QuantizedInterleavedInput) -> (Vec<i8>, Vec<f32>, Vec<i32>) {
    let context = B::Context::new().expect("create context");
    let fused_up = alloc_allocation_with_data::<B, f32>(&context, &input.fused_up);
    let factors = alloc_allocation_with_data::<B, i32>(&context, &input.factors);
    let mut values = alloc_allocation::<B, i8>(&context, (input.batch_dim * input.gated_dim) as usize);
    let mut scales = alloc_allocation::<B, f32>(&context, (input.batch_dim * input.gated_dim / 128) as usize);
    let mut group_sums = alloc_allocation::<B, i32>(&context, (input.batch_dim * input.gated_dim / 32) as usize);

    let gate = <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(
        &context,
        f32::data_type(),
        GatedActMulOp::QuantizeWithGroupSums,
        true,
        true,
        128,
        32,
    )
    .expect("create GatedActMulKernel");
    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    gate.encode(
        &fused_up,
        None::<&Allocation<B>>,
        None::<&mut Allocation<B>>,
        Some(&mut values),
        Some(&mut scales),
        Some(&mut group_sums),
        Some(&factors),
        input.gated_dim,
        input.batch_dim,
        0,
        0,
        ActivationType::SILU,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&values), allocation_to_vec(&scales), allocation_to_vec(&group_sums))
}

fn run_interleaved_hadamard<B: Backend>(input: &QuantizedInterleavedInput) -> Vec<f32> {
    let context = B::Context::new().expect("create context");
    let fused_up = alloc_allocation_with_data::<B, f32>(&context, &input.fused_up);
    let factors = alloc_allocation_with_data::<B, i32>(&context, &input.factors);
    let mut output = alloc_allocation::<B, f32>(&context, (input.batch_dim * input.gated_dim) as usize);
    let kernel = <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(
        &context,
        f32::data_type(),
        GatedActMulOp::FullPrecision,
        true,
        true,
        32,
        32,
    )
    .expect("create GatedActMulKernel");
    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &fused_up,
        None::<&Allocation<B>>,
        Some(&mut output),
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
        Some(&factors),
        input.gated_dim,
        input.batch_dim,
        0,
        0,
        ActivationType::SILU,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&output)
}

#[uzu_test]
fn test_gated_act_mul_interleaved_hadamard_f32() {
    let input = quantized_interleaved_input();
    let expected = run_interleaved_hadamard::<Cpu>(&input);
    for_each_backend!(|B| {
        let actual = run_interleaved_hadamard::<B>(&input);
        assert_eq_float::<f32>(&expected, &actual, 1e-5, "Hadamard gated activation mismatch");
    });
}

#[uzu_test]
fn test_gated_act_mul_quantized_matches_unfused() {
    let input = quantized_interleaved_input();
    let expected = run_unfused_quantized(&input);
    for_each_backend!(|B| {
        let actual = run_fused_quantized::<B>(&input);
        for (index, (&actual, &expected)) in actual.0.iter().zip(&expected.0).enumerate() {
            assert!((i32::from(actual) - i32::from(expected)).abs() <= 1, "code {index}: {actual} != {expected}");
        }
        for (index, (&actual, &expected)) in actual.1.iter().zip(&expected.1).enumerate() {
            let relative_error = (actual - expected).abs() / expected.abs().max(1e-6);
            assert!(relative_error < 1e-3, "scale {index}: {actual} != {expected}");
        }
        assert_eq!(actual.2, expected.2, "group sums mismatch for {}", std::any::type_name::<B>());
    });
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
    let kernel = <<B as Backend>::Kernels as Kernels>::GatedActMulKernel::new(
        &context,
        T::data_type(),
        GatedActMulOp::FullPrecision,
        false,
        false,
        32,
        32,
    )
    .expect("create GatedActMulKernel");

    let gate_out = alloc_allocation_with_data::<B, T>(&context, &input.gate_out);
    let per_layer_input = alloc_allocation_with_data::<B, T>(&context, &input.per_layer_input);
    let mut output = alloc_allocation::<B, T>(&context, (input.batch_dim * input.gated_dim) as usize);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &gate_out,
        Some(&per_layer_input),
        Some(&mut output),
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
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
