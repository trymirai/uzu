use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::SeparableCausalConvKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
    },
};

fn run_kernel<B: Backend>() -> Vec<bf16> {
    const SEQUENCE_LENGTH: u32 = 2;
    const MODEL_DIM: u32 = 4;
    const KERNEL_SIZE: u32 = 2;
    const GROUP_SIZE: u32 = 2;
    const COEFFICIENT_ROW_STRIDE: u32 = 8;

    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(bf16::from_f32);
    let weights = [10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0].map(bf16::from_f32);
    let coefficient_deltas =
        [1.0, 2.0, 3.0, 4.0, 99.0, 99.0, 99.0, 99.0, 5.0, 6.0, 7.0, 8.0, 99.0, 99.0, 99.0, 99.0].map(bf16::from_f32);

    let context = B::Context::new().expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::SeparableCausalConvKernel::new(
        &context,
        bf16::data_type(),
        MODEL_DIM,
        KERNEL_SIZE,
        GROUP_SIZE,
        false,
    )
    .expect("create separable causal convolution kernel");

    let input = alloc_allocation_with_data::<B, bf16>(&context, &input);
    let coefficient_deltas = alloc_allocation_with_data::<B, bf16>(&context, &coefficient_deltas);
    let weights = alloc_allocation_with_data::<B, bf16>(&context, &weights);
    let mut output = alloc_allocation::<B, bf16>(&context, (SEQUENCE_LENGTH * MODEL_DIM) as usize);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &input,
        &coefficient_deltas,
        &weights,
        None::<&Allocation<B>>,
        &mut output,
        SEQUENCE_LENGTH,
        COEFFICIENT_ROW_STRIDE,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec::<B, bf16>(&output)
}

#[uzu_test]
fn test_separable_causal_convolution() {
    let expected = run_kernel::<Cpu>();

    for_each_non_cpu_backend!(|B| {
        let actual = run_kernel::<B>();
        assert_eq_float(&expected, &actual, 0.05, "separable causal convolution backend parity");
    });
}
