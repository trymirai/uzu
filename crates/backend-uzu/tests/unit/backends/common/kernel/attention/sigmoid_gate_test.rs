use std::fmt::Debug;

use half::{bf16, f16};
use num_traits::Float;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::SigmoidGateKernel},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec},
};

struct Config {
    num_heads: u32,
    head_dim: u32,
    suffix_length: u32,
}

fn get_output<T: ArrayElement + Float, B: Backend>(
    gate_data: &[T],
    output_data: &[T],
    config: &Config,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::SigmoidGateKernel::new(&context, T::data_type())
        .expect("Failed to create SigmoidGateKernel");

    let gate_allocation = alloc_allocation_with_data::<B, T>(&context, gate_data);
    let mut output = alloc_allocation_with_data::<B, T>(&context, output_data);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let total_elements = config.suffix_length * config.num_heads * config.head_dim;
    kernel.encode(&gate_allocation, &mut output, total_elements, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&output)
}

fn run_test<T: ArrayElement + Float + Debug>(config: &Config) {
    let size = (config.suffix_length * config.num_heads * config.head_dim) as usize;

    let gate_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let output_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 + 0.5).collect();

    let gate_data: Vec<T> = gate_f32.iter().map(|&v| T::from(v).unwrap()).collect();
    let output_data: Vec<T> = output_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    let expected = get_output::<T, Cpu>(&gate_data, &output_data, config);

    let rtol = if std::mem::size_of::<T>() <= 2 {
        0.01
    } else {
        1e-5
    };

    for_each_non_cpu_backend!(|B| {
        let result = get_output::<T, B>(&gate_data, &output_data, config);

        for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            let got_f32 = got.to_f32().unwrap();
            let exp_f32 = exp.to_f32().unwrap();
            let diff = (got_f32 - exp_f32).abs();
            let tol = rtol * exp_f32.abs().max(1.0);
            assert!(
                diff < tol,
                "Backend {}: mismatch at index {}: got {} expected {} (diff {}, tol {})",
                std::any::type_name::<B>(),
                i,
                got_f32,
                exp_f32,
                diff,
                tol,
            );
        }
    });
}

#[uzu_test]
fn test_sigmoid_gate_f32() {
    run_test::<f32>(&Config {
        num_heads: 8,
        head_dim: 64,
        suffix_length: 4,
    });
    run_test::<f32>(&Config {
        num_heads: 16,
        head_dim: 256,
        suffix_length: 1,
    });
    run_test::<f32>(&Config {
        num_heads: 2,
        head_dim: 64,
        suffix_length: 8,
    });
}

#[uzu_test]
fn test_sigmoid_gate_f16() {
    run_test::<f16>(&Config {
        num_heads: 8,
        head_dim: 64,
        suffix_length: 4,
    });
    run_test::<f16>(&Config {
        num_heads: 16,
        head_dim: 256,
        suffix_length: 1,
    });
}

#[uzu_test]
fn test_sigmoid_gate_bf16() {
    run_test::<bf16>(&Config {
        num_heads: 8,
        head_dim: 64,
        suffix_length: 4,
    });
    run_test::<bf16>(&Config {
        num_heads: 16,
        head_dim: 256,
        suffix_length: 1,
    });
}
