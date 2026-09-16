use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::SeparableCausalConvKernel},
        cpu::Cpu,
    },
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
};

fn run_kernel<B: Backend>() -> Vec<bf16> {
    const SEQUENCE_LENGTH: u32 = 2;
    const MODEL_DIM: u32 = 16;
    const KERNEL_SIZE: u32 = 2;
    const GROUP_SIZE: u32 = 16;
    const COEFFICIENT_ROW_STRIDE: u32 = 2 * KERNEL_SIZE * (MODEL_DIM / GROUP_SIZE);

    let input = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
    ]
    .map(bf16::from_f32);
    let weights = [
        10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0, 50.0, 5.0, 60.0, 6.0, 70.0, 7.0, 80.0, 8.0, 90.0, 9.0, 100.0, 10.0,
        110.0, 11.0, 120.0, 12.0, 130.0, 13.0, 140.0, 14.0, 150.0, 15.0, 160.0, 16.0,
    ]
    .map(bf16::from_f32);
    let coefficient_deltas = [1.0, 2.0, 99.0, 99.0, 5.0, 6.0, 99.0, 99.0].map(bf16::from_f32);

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
        assert_eq!(expected, actual);
    });
}
