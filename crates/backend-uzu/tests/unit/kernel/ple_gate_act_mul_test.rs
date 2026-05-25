use std::fmt::Debug;

use backend_uzu::{
    ArrayContextExt, ArrayElement, DataType,
    backends::common::{Backend, Context, Encoder, Kernels, gpu_types::ActivationType, kernel::PleGateActMulKernel},
};
use half::{bf16, f16};
use num_traits::Float;

use crate::{
    common::helpers::{alloc_allocation, allocation_to_vec},
    uzu_test,
};

struct Input<T: ArrayElement + Float> {
    gate_out: Box<[T]>,
    per_layer_input: Box<[T]>,
    ple_dim: i32,
    batch_dim: i32,
    num_layers: i32,
    layer_offset: i32,
    activation: ActivationType,
}

fn test_input<T: ArrayElement + Float>() -> (Input<T>, Vec<T>) {
    let gate_out = [1.0_f32, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|value| T::from(value).unwrap())
        .collect::<Vec<_>>();
    let per_layer_input = [
        0.0_f32, 0.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0, 50.0, 60.0, 70.0, 80.0,
    ]
    .into_iter()
    .map(|value| T::from(value).unwrap())
    .collect::<Vec<_>>();
    let expected = [10.0_f32, 40.0, 150.0, 240.0]
        .into_iter()
        .map(|value| T::from(value).unwrap())
        .collect::<Vec<_>>();

    (
        Input {
            gate_out: gate_out.into_boxed_slice(),
            per_layer_input: per_layer_input.into_boxed_slice(),
            ple_dim: 2,
            batch_dim: 2,
            num_layers: 3,
            layer_offset: 2,
            activation: ActivationType::IDENTITY,
        },
        expected,
    )
}

fn run_kernel<T: ArrayElement + Float, B: Backend>(input: &Input<T>) -> Vec<T> {
    let context = B::Context::new().expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::PleGateActMulKernel::new(&context, T::data_type())
        .expect("create PleGateActMulKernel");
    let gate_out = context.create_array_from(&[input.batch_dim as usize, input.ple_dim as usize], &input.gate_out);
    let per_layer_input = context.create_array_from(
        &[input.batch_dim as usize, input.num_layers as usize, input.ple_dim as usize],
        &input.per_layer_input,
    );
    let mut output = alloc_allocation::<B, T>(&context, input.batch_dim as usize * input.ple_dim as usize);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        gate_out.allocation(),
        per_layer_input.allocation(),
        &mut output,
        input.ple_dim,
        input.batch_dim,
        input.num_layers,
        input.layer_offset,
        input.activation,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, T>(&output)
}

fn test<T: ArrayElement + Float + Debug>() {
    let (input, expected) = test_input::<T>();
    for_each_backend!(|B| {
        let output = run_kernel::<T, B>(&input);
        assert_eq!(expected, output, "{}", std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_ple_gate_act_mul_f32() {
    test::<f32>();
}

#[uzu_test]
fn test_ple_gate_act_mul_f16() {
    test::<f16>();
}

#[uzu_test]
fn test_ple_gate_act_mul_bf16() {
    test::<bf16>();
}
