use std::fmt::Debug;

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::SigmoidGateKernel},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
};

struct Config {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    suffix_length: u32,
}

fn get_output<T: ArrayElement + Float, B: Backend>(
    qkvg_data: &[T],
    output_data: &[T],
    config: &Config,
) -> Vec<T> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::SigmoidGateKernel::new(&context, T::data_type())
        .expect("Failed to create SigmoidGateKernel");

    let qkvg = alloc_allocation_with_data::<B, T>(&context, qkvg_data);
    let mut output = alloc_allocation_with_data::<B, T>(&context, output_data);

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    let gate_dim = config.num_heads * config.head_dim;
    let gate_offset = (config.num_heads + 2 * config.num_kv_heads) * config.head_dim;
    let qkvg_dim = gate_offset + gate_dim;
    kernel.encode(
        (&qkvg, gate_offset as usize * size_of::<T>()),
        &mut output,
        gate_dim,
        config.suffix_length,
        qkvg_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&output)
}

fn run_test<T: ArrayElement + Float + Debug>(config: &Config) {
    let size = (config.suffix_length * config.num_heads * config.head_dim) as usize;
    let gate_dim = (config.num_heads * config.head_dim) as usize;
    let gate_offset = ((config.num_heads + 2 * config.num_kv_heads) * config.head_dim) as usize;
    let qkvg_dim = gate_offset + gate_dim;

    let qkvg_f32 = (0..config.suffix_length as usize)
        .flat_map(|row| {
            (0..qkvg_dim).map(move |column| {
                if column < gate_offset {
                    return -100.0;
                }
                let gate_index = row * gate_dim + column - gate_offset;
                (gate_index as f32) * 0.1 - 2.0
            })
        })
        .collect::<Vec<_>>();
    let output_f32: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05 + 0.5).collect();

    let qkvg_data: Vec<T> = qkvg_f32.iter().map(|&v| T::from(v).unwrap()).collect();
    let output_data: Vec<T> = output_f32.iter().map(|&v| T::from(v).unwrap()).collect();
    let expected = (0..size)
        .map(|output_index| {
            let batch_index = output_index / gate_dim;
            let gate_index = output_index % gate_dim;
            let gate = qkvg_data[batch_index * qkvg_dim + gate_offset + gate_index].to_f32().unwrap();
            let sigmoid = 1.0 / (1.0 + (-gate).exp());
            T::from(output_data[output_index].to_f32().unwrap() * sigmoid).unwrap()
        })
        .collect::<Vec<_>>();

    let rtol = if std::mem::size_of::<T>() <= 2 {
        0.01
    } else {
        1e-5
    };

    let assert_output = |result: &[T], backend: &str| {
        for (i, (got, exp)) in result.iter().zip(expected.iter()).enumerate() {
            let got_f32 = got.to_f32().unwrap();
            let exp_f32 = exp.to_f32().unwrap();
            let diff = (got_f32 - exp_f32).abs();
            let tol = rtol * exp_f32.abs().max(1.0);
            assert!(
                diff < tol,
                "Backend {}: mismatch at index {}: got {} expected {} (diff {}, tol {})",
                backend,
                i,
                got_f32,
                exp_f32,
                diff,
                tol,
            );
        }
    };

    let result = get_output::<T, Cpu>(&qkvg_data, &output_data, config);
    assert_output(&result, std::any::type_name::<Cpu>());
    for_each_non_cpu_backend!(|B| {
        let result = get_output::<T, B>(&qkvg_data, &output_data, config);
        assert_output(&result, std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_sigmoid_gate_f32() {
    run_test::<f32>(&Config {
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 64,
        suffix_length: 4,
    });
    run_test::<f32>(&Config {
        num_heads: 16,
        num_kv_heads: 4,
        head_dim: 256,
        suffix_length: 1,
    });
    run_test::<f32>(&Config {
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 64,
        suffix_length: 8,
    });
}

#[uzu_test]
fn test_sigmoid_gate_bf16() {
    run_test::<bf16>(&Config {
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 64,
        suffix_length: 4,
    });
    run_test::<bf16>(&Config {
        num_heads: 16,
        num_kv_heads: 4,
        head_dim: 256,
        suffix_length: 1,
    });
}
