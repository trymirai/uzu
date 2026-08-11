use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::AttentionPrepareKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, for_each_non_cpu_backend},
};

struct Output {
    queries: Vec<bf16>,
    keys: Vec<bf16>,
    values: Vec<bf16>,
}

fn run<B: Backend>() -> Output {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        true,
        false,
    )
    .expect("Failed to create AttentionPrepareKernel");
    let qkvg = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 101.0, 102.0, 103.0, 104.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 111.0, 112.0, 113.0, 114.0,
    ]
    .map(bf16::from_f32);
    let qkvg = alloc_allocation_with_data::<B, bf16>(&context, &qkvg);
    let mut queries = alloc_allocation::<B, bf16>(&context, 8);
    let mut keys = alloc_allocation::<B, bf16>(&context, 4);
    let mut values = alloc_allocation::<B, bf16>(&context, 4);
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    kernel.encode(
        &qkvg,
        &mut queries,
        Some(&mut keys),
        Some(&mut values),
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        2,
        Some(1),
        2,
        None,
        Some(0),
        12,
        2,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        queries: allocation_to_vec(&queries),
        keys: allocation_to_vec(&keys),
        values: allocation_to_vec(&values),
    }
}

fn assert_output(
    output: Output,
    backend: &str,
) {
    let expected_queries = [1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0].map(bf16::from_f32);
    let expected_keys = [5.0, 6.0, 15.0, 16.0].map(bf16::from_f32);
    let expected_values = [7.0, 8.0, 17.0, 18.0].map(bf16::from_f32);

    assert_eq!(output.queries, expected_queries, "query mismatch on {backend}");
    assert_eq!(output.keys, expected_keys, "key mismatch on {backend}");
    assert_eq!(output.values, expected_values, "value mismatch on {backend}");
}

fn run_query_only<B: Backend>() -> Vec<bf16> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::AttentionPrepareKernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        false,
        false,
    )
    .expect("Failed to create AttentionPrepareKernel");
    let qg = [1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0, 11.0, 12.0, 13.0, 14.0, 111.0, 112.0, 113.0, 114.0]
        .map(bf16::from_f32);
    let qg = alloc_allocation_with_data::<B, bf16>(&context, &qg);
    let mut queries = alloc_allocation::<B, bf16>(&context, 8);
    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");

    kernel.encode(
        &qg,
        &mut queries,
        None::<&mut Allocation<B>>,
        None::<&mut Allocation<B>>,
        None::<&Allocation<B>>,
        None::<&Allocation<B>>,
        2,
        None,
        2,
        None,
        None,
        8,
        2,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    allocation_to_vec(&queries)
}

#[uzu_test]
fn test_attention_prepare_uses_qkvg_row_stride() {
    assert_output(run::<Cpu>(), "CPU");
    for_each_non_cpu_backend!(|B| {
        assert_output(run::<B>(), std::any::type_name::<B>());
    });
}

#[uzu_test]
fn test_attention_prepare_uses_query_gate_row_stride() {
    let expected = [1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 13.0, 14.0].map(bf16::from_f32);
    assert_eq!(run_query_only::<Cpu>(), expected, "query mismatch on CPU");
    for_each_non_cpu_backend!(|B| {
        assert_eq!(run_query_only::<B>(), expected, "query mismatch on {}", std::any::type_name::<B>());
    });
}
