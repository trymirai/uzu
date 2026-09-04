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

#[derive(Clone, Copy)]
struct ConvolutionShape {
    sequence_length: usize,
    model_dim: usize,
    group_size: usize,
    kernel_size: usize,
}

impl ConvolutionShape {
    fn groups(self) -> usize {
        self.model_dim / self.group_size
    }

    fn input_len(self) -> usize {
        self.sequence_length * self.model_dim
    }

    fn coefficients_len(self) -> usize {
        self.sequence_length * 2 * self.kernel_size * self.groups()
    }

    fn base_kernel_len(self) -> usize {
        2 * self.kernel_size * self.model_dim
    }

    fn stage_offsets(
        self,
        stage: u32,
    ) -> (usize, usize) {
        let stage = stage as usize;
        let element_size = bf16::data_type().size_in_bytes();
        (
            stage * self.kernel_size * self.groups() * element_size,
            stage * self.kernel_size * self.model_dim * element_size,
        )
    }
}

const INPUT_STAGE: u32 = 0;
const OUTPUT_STAGE: u32 = 1;

fn run_kernel<B: Backend>(
    shape: ConvolutionShape,
    input: &[bf16],
    coefficients: &[bf16],
    base_kernel: &[bf16],
    stage: u32,
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
    let (coefficient_offset, base_kernel_offset) = shape.stage_offsets(stage);
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
fn test_bf16_scalar_golden() {
    let shape = ConvolutionShape {
        sequence_length: 4,
        model_dim: 4,
        group_size: 2,
        kernel_size: 2,
    };
    const EXPECTED: [[f32; 16]; 2] = [
        [1.1, 0.75, 0.425, 0.1875, 0.2, -0.075, -0.25, -0.3, -0.3, -0.175, 0.15, 0.5, 0.8, 1.325, 2.15, 2.9],
        [0.1, 0.0, -0.075, -0.0625, -0.8, -0.575, -0.25, 0.2, 0.7, 1.325, 2.15, 3.0, 3.8, 4.825, 6.15, 7.4],
    ];
    let input = (0..16).map(|index| bf16::from_f32(index as f32 * 0.25 - 1.0)).collect::<Vec<_>>();
    let coefficients = (0..32).map(|index| bf16::from_f32(index as f32 * 0.05 - 0.4)).collect::<Vec<_>>();
    let base_kernel = (0..16).map(|index| bf16::from_f32(index as f32 * 0.1 - 0.7)).collect::<Vec<_>>();
    for stage in [INPUT_STAGE, OUTPUT_STAGE] {
        let expected = run_kernel::<Cpu>(shape, &input, &coefficients, &base_kernel, stage);
        assert_eq_float(&EXPECTED[stage as usize], &expected, 0.05, "grouped convolution CPU reference");
        for_each_non_cpu_backend!(|B| {
            let actual = run_kernel::<B>(shape, &input, &coefficients, &base_kernel, stage);
            assert_eq_float(&expected, &actual, 0.05, "grouped convolution backend parity");
        });
    }
}

#[uzu_test]
fn test_bf16_vectorized_backend_parity() {
    let shape = ConvolutionShape {
        sequence_length: 4,
        model_dim: 16,
        group_size: 16,
        kernel_size: 2,
    };
    let input = (0..shape.input_len()).map(|index| bf16::from_f32(index as f32 * 0.125 - 1.0)).collect::<Vec<_>>();
    let coefficients =
        (0..shape.coefficients_len()).map(|index| bf16::from_f32(index as f32 * 0.02 - 0.2)).collect::<Vec<_>>();
    let base_kernel =
        (0..shape.base_kernel_len()).map(|index| bf16::from_f32(index as f32 * 0.03 - 0.4)).collect::<Vec<_>>();

    for stage in [INPUT_STAGE, OUTPUT_STAGE] {
        let expected = run_kernel::<Cpu>(shape, &input, &coefficients, &base_kernel, stage);
        for_each_non_cpu_backend!(|B| {
            let actual = run_kernel::<B>(shape, &input, &coefficients, &base_kernel, stage);
            assert_eq_float(&expected, &actual, 0.05, "grouped convolution vectorized parity");
        });
    }
}
