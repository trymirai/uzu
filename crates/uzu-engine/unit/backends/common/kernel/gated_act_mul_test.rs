use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    ClippingBounds,
    array::ArrayElement,
    backends::{
        common::{
            Allocation, Backend, Context, Encoder,
            gpu_types::ActivationType,
            kernel::{GatedActMul, GatedActMulSettings},
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
    hadamard_factors: Box<[i32]>,
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
        hadamard_factors: vec![1; gated_dim as usize].into_boxed_slice(),
        gated_dim,
        batch_dim,
        act_type,
    }
}

fn run_interleaved<T: ArrayElement + Float, B: Backend>(
    input: &InterleavedInput<T>,
    use_hadamard: bool,
) -> Vec<T> {
    let context = B::Context::new().expect("create context");
    let kernel =
        GatedActMul::<B>::full_precision(&context, T::data_type(), true, use_hadamard, GatedActMulSettings::default())
            .expect("create GatedActMul");

    let fused_length = (input.batch_dim * 2 * input.gated_dim) as usize;
    let output_length = (input.batch_dim * input.gated_dim) as usize;
    let fused_up = alloc_allocation_with_data::<B, T>(&context, &input.fused_up[..fused_length]);
    let hadamard_factors = alloc_allocation_with_data::<B, i32>(&context, &input.hadamard_factors);
    let mut output = alloc_allocation::<B, T>(&context, output_length);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode_fp(
        &fused_up,
        None::<&Allocation<B>>,
        &mut output,
        use_hadamard.then_some(&hadamard_factors),
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
    let expected = run_interleaved::<T, Cpu>(&input, false);
    for_each_non_cpu_backend!(|B| {
        let output = run_interleaved::<T, B>(&input, false);
        let message = format!("interleaved mismatch for backend {}", std::any::type_name::<B>());
        assert_eq_float::<T>(&expected, &output, eps, &message);
    });
}

#[uzu_test]
fn test_gated_act_mul_interleaved_hadamard_bf16() {
    let input = interleaved_input::<bf16>(ActivationType::SILU);
    let expected = run_interleaved::<bf16, Cpu>(&input, true);
    for_each_non_cpu_backend!(|B| {
        let actual = run_interleaved::<bf16, B>(&input, true);
        // Cross-backend RHT results can differ by one bf16 ULP.
        assert_eq_float::<bf16>(&expected, &actual, 0.04, "Hadamard gated activation mismatch");
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
    let kernel =
        GatedActMul::<B>::full_precision(&context, T::data_type(), false, false, GatedActMulSettings::default())
            .expect("create GatedActMul");

    let gate_out = alloc_allocation_with_data::<B, T>(&context, &input.gate_out);
    let per_layer_input = alloc_allocation_with_data::<B, T>(&context, &input.per_layer_input);
    let mut output = alloc_allocation::<B, T>(&context, (input.batch_dim * input.gated_dim) as usize);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode_fp(
        &gate_out,
        Some(&per_layer_input),
        &mut output,
        None,
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

fn transformed_interleaved_test<T: ArrayElement + Float + Debug + Display>(
    gate_clipping: ClippingBounds,
    value_clipping: ClippingBounds,
) {
    const GATED_DIM: u32 = 4;
    const ACTIVATION_ALPHA: f32 = 0.5;

    let values = [-4.0f32, -1.0, 2.0, 5.0];
    let gates = [-3.0f32, -0.5, 1.0, 4.0];
    let fused_up = values.into_iter().chain(gates).map(|value| T::from(value).unwrap()).collect::<Vec<_>>();
    let expected = values
        .into_iter()
        .zip(gates)
        .map(|(value, gate)| {
            let value = T::from(value_clipping.apply(value)).unwrap();
            let gate = gate_clipping.apply(gate);
            let activated = T::from(gate / (1.0 + (-ACTIVATION_ALPHA * gate).exp())).unwrap();
            value * activated
        })
        .collect::<Vec<_>>();
    let tolerance = if T::data_type() == DataType::BF16 {
        0.02
    } else {
        1e-5
    };
    let settings = GatedActMulSettings {
        activation_alpha: Some(ACTIVATION_ALPHA),
        gate_clipping,
        value_clipping,
    };

    for_each_backend!(|B| {
        let context = <B as Backend>::Context::new().expect("create context");
        let kernel = GatedActMul::<B>::full_precision(&context, T::data_type(), true, false, settings)
            .expect("create transformed GatedActMul");
        let fused_up = alloc_allocation_with_data::<B, T>(&context, &fused_up);
        let mut output = alloc_allocation::<B, T>(&context, GATED_DIM as usize);
        let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
        kernel.encode_fp(
            &fused_up,
            None::<&Allocation<B>>,
            &mut output,
            None::<&Allocation<B>>,
            GATED_DIM,
            1,
            0,
            0,
            ActivationType::SILU,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();

        let output = allocation_to_vec::<B, T>(&output);
        assert_eq_float(
            &expected,
            &output,
            tolerance,
            &format!("transformed mismatch for backend {}", <B as Backend>::NAME),
        );
    });
}

#[uzu_test]
fn test_gated_act_mul_transforms_f32() {
    transformed_interleaved_test::<f32>(ClippingBounds::bounded(-1.0, 2.0), ClippingBounds::bounded(-2.0, 3.0));
}

#[uzu_test]
fn test_gated_act_mul_transforms_bf16() {
    transformed_interleaved_test::<bf16>(ClippingBounds::bounded(-1.0, 2.0), ClippingBounds::bounded(-2.0, 3.0));
}

#[uzu_test]
fn test_gated_act_mul_one_sided_clipping_f32() {
    transformed_interleaved_test::<f32>(ClippingBounds::lower_bounded(-1.0), ClippingBounds::upper_bounded(3.0));
}

#[uzu_test]
fn test_gated_act_mul_one_sided_clipping_bf16() {
    transformed_interleaved_test::<bf16>(ClippingBounds::lower_bounded(-1.0), ClippingBounds::upper_bounded(3.0));
}

#[uzu_test]
fn test_gated_act_mul_clipping_bounds_wire_format() {
    let cases = [
        ("null", ClippingBounds::default()),
        ("[-1.0,2.0]", ClippingBounds::bounded(-1.0, 2.0)),
        ("[-1.0,null]", ClippingBounds::lower_bounded(-1.0)),
        ("[null,3.0]", ClippingBounds::upper_bounded(3.0)),
    ];

    for (json, expected) in cases {
        let bounds = serde_json::from_str::<ClippingBounds>(json).expect("deserialize clipping bounds");
        assert_eq!(bounds, expected);
        assert_eq!(serde_json::to_string(&bounds).expect("serialize clipping bounds"), json);
    }

    let empty_pair = serde_json::from_str::<ClippingBounds>("[null,null]").expect("deserialize empty bounds");
    assert_eq!(empty_pair, ClippingBounds::default());
    assert_eq!(serde_json::to_string(&empty_pair).expect("serialize empty bounds"), "null");
}
