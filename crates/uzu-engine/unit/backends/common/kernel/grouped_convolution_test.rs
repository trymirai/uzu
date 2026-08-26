use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::GroupedConvolutionKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{
            alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend, submit_encoder,
        },
    },
};

fn run<B: Backend>(
    input: &[bf16],
    coefficients: &[bf16],
    base_kernel: &[bf16],
    stage: u32,
) -> Vec<f32> {
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::GroupedConvolutionKernel::new(&context, bf16::data_type()).unwrap();
    let input = alloc_allocation_with_data::<B, bf16>(&context, input);
    let coefficients = alloc_allocation_with_data::<B, bf16>(&context, coefficients);
    let base_kernel = alloc_allocation_with_data::<B, bf16>(&context, base_kernel);
    let mut output = alloc_allocation::<B, bf16>(&context, 4 * 4);
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(&input, &coefficients, &base_kernel, &mut output, 4, 4, 2, 2, 2, stage, &mut encoder);
    submit_encoder(encoder);
    allocation_to_vec::<B, bf16>(&output).into_iter().map(|value| value.to_f32()).collect()
}

fn test() {
    const EXPECTED: [[f32; 16]; 2] = [
        [1.1, 0.75, 0.425, 0.1875, 0.2, -0.075, -0.25, -0.3, -0.3, -0.175, 0.15, 0.5, 0.8, 1.325, 2.15, 2.9],
        [0.1, 0.0, -0.075, -0.0625, -0.8, -0.575, -0.25, 0.2, 0.7, 1.325, 2.15, 3.0, 3.8, 4.825, 6.15, 7.4],
    ];
    let input = (0..16).map(|index| bf16::from_f32(index as f32 * 0.25 - 1.0)).collect::<Vec<_>>();
    let coefficients = (0..32).map(|index| bf16::from_f32(index as f32 * 0.05 - 0.4)).collect::<Vec<_>>();
    let base_kernel = (0..16).map(|index| bf16::from_f32(index as f32 * 0.1 - 0.7)).collect::<Vec<_>>();
    for stage in [0u32, 1] {
        let expected = run::<Cpu>(&input, &coefficients, &base_kernel, stage);
        assert_eq_float(&EXPECTED[stage as usize], &expected, 0.05, "grouped convolution CPU reference");
        for_each_non_cpu_backend!(|B| {
            let actual = run::<B>(&input, &coefficients, &base_kernel, stage);
            assert_eq_float(&expected, &actual, 0.05, "grouped convolution backend parity");
        });
    }
}

#[uzu_test]
fn test_bf16() {
    test();
}
