use std::mem::size_of;

use half::bf16;
use uzu_engine_macros::uzu_test;

use super::{ConvolutionShape, INPUT_STAGE, OUTPUT_STAGE};
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

fn run_kernel<B: Backend>(
    shape: ConvolutionShape,
    input: &[bf16],
    coefficients: &[bf16],
    base_kernel: &[bf16],
    stage: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), shape.input_len());
    assert_eq!(coefficients.len(), shape.coefficients_len());
    assert_eq!(base_kernel.len(), shape.base_kernel_len());
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::GroupedConvolutionKernel::new(
        &context,
        bf16::data_type(),
        shape.model_dim as u32,
        shape.group_size as u32,
        shape.kernel_size as u32,
    )
    .unwrap();
    let input = alloc_allocation_with_data::<B, bf16>(&context, input);
    let coefficients = alloc_allocation_with_data::<B, bf16>(&context, coefficients);
    let base_kernel = alloc_allocation_with_data::<B, bf16>(&context, base_kernel);
    let mut output = alloc_allocation::<B, bf16>(&context, shape.input_len());
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    let (coefficient_offset, base_kernel_offset) = shape.stage_offsets(stage, size_of::<bf16>());
    kernel.encode(
        &input,
        (&coefficients, coefficient_offset),
        (&base_kernel, base_kernel_offset),
        &mut output,
        shape.sequence_length as u32,
        &mut encoder,
    );
    submit_encoder(encoder);
    allocation_to_vec::<B, bf16>(&output).into_iter().map(|value| value.to_f32()).collect()
}

#[uzu_test]
fn test_bf16_group_sizes() {
    for (model_dim, group_size) in [(4, 2), (16, 16)] {
        let shape = ConvolutionShape {
            sequence_length: 4,
            model_dim,
            group_size,
            kernel_size: 2,
        };
        let input =
            (0..shape.input_len()).map(|index| bf16::from_f32((index % 17) as f32 * 0.125 - 1.0)).collect::<Vec<_>>();
        let coefficients = (0..shape.coefficients_len())
            .map(|index| bf16::from_f32((index % 11) as f32 * 0.02 - 0.1))
            .collect::<Vec<_>>();
        let base_kernel = (0..shape.base_kernel_len())
            .map(|index| bf16::from_f32((index % 13) as f32 * 0.03 - 0.18))
            .collect::<Vec<_>>();

        for stage in [INPUT_STAGE, OUTPUT_STAGE] {
            let expected = run_kernel::<Cpu>(shape, &input, &coefficients, &base_kernel, stage);
            for_each_non_cpu_backend!(|B| {
                let actual = run_kernel::<B>(shape, &input, &coefficients, &base_kernel, stage);
                assert_eq_float(&expected, &actual, 0.05, "grouped convolution backend parity");
            });
        }
    }
}
