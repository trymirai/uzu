use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::{AttentionPrepareInt8Kernel, AttentionPrepareKernel},
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
};

struct Output {
    queries: Vec<bf16>,
    keys: Vec<bf16>,
    values: Vec<bf16>,
}

fn run<B: Backend>() -> Output {
    let context = block_on(B::Context::new()).expect("Failed to create Context");
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
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &qkvg);
    let mut queries = create_buffer::<B, bf16>(&context, 8);
    let mut keys = create_buffer::<B, bf16>(&context, 4);
    let mut values = create_buffer::<B, bf16>(&context, 4);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");

    kernel.encode(
        &qkvg,
        &mut queries,
        Some(&mut keys),
        Some(&mut values),
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        2,
        Some(1),
        2,
        None,
        Some(0),
        12,
        2,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");

    Output {
        queries: buffer_to_vec(&*context, &queries),
        keys: buffer_to_vec(&*context, &keys),
        values: buffer_to_vec(&*context, &values),
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
    let context = block_on(B::Context::new()).expect("Failed to create Context");
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
    let qg = create_buffer_with_data::<B, bf16>(&context, &qg);
    let mut queries = create_buffer::<B, bf16>(&context, 8);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");

    kernel.encode(
        &qg,
        &mut queries,
        None::<&mut B::GlobalGpuBuffer>,
        None::<&mut B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        2,
        None,
        2,
        None,
        None,
        8,
        2,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("Failed to wait command buffer");

    buffer_to_vec(&*context, &queries)
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

#[derive(Debug, PartialEq)]
struct Int8Output {
    codes: [Vec<u8>; 3],
    scales: [Vec<bf16>; 3],
}

fn run_int8<B: Backend>(num_q_heads: u32) -> Int8Output {
    const DIM: usize = 256;
    const CACHE_TOKENS: usize = 64;
    let code_bytes = CACHE_TOKENS * DIM;
    let rows: Vec<bf16> = (0..3 * DIM).map(|i| bf16::from_f32((i as f32 * 0.37).sin())).collect();
    let context = block_on(B::Context::new()).unwrap();
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &rows[(1 - num_q_heads) as usize * DIM..]);
    let mut q_codes = create_buffer_with_data::<B, i8>(&context, &[0; DIM]);
    let mut q_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO]);
    let mut k_codes = create_buffer_with_data::<B, i8>(&context, &vec![0; code_bytes]);
    let mut v_codes = create_buffer_with_data::<B, u8>(&context, &vec![0; code_bytes]);
    let mut k_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; CACHE_TOKENS]);
    let mut v_scales = create_buffer_with_data::<B, bf16>(&context, &[bf16::ZERO; CACHE_TOKENS]);
    let kernel = <B::Kernels as Kernels>::AttentionPrepareInt8Kernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        DIM as u32,
        false,
        true,
        1,
        true,
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &qkvg,
        None::<&mut B::GlobalGpuBuffer>,
        Some(&mut q_codes),
        Some(&mut q_scales),
        Some(&mut k_codes),
        Some(&mut k_scales),
        Some(&mut v_codes),
        Some(&mut v_scales),
        None::<&B::GlobalGpuBuffer>,
        None::<&B::GlobalGpuBuffer>,
        num_q_heads,
        None,
        33,
        (num_q_heads + 2) * DIM as u32,
        1,
        num_q_heads,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    Int8Output {
        codes: [&q_codes, &k_codes, &v_codes].map(|buffer| buffer_to_vec::<B, u8>(&context, buffer)),
        scales: [&q_scales, &k_scales, &v_scales].map(|buffer| buffer_to_vec::<B, bf16>(&context, buffer)),
    }
}

#[uzu_test]
fn test_attention_prepare_int8() {
    for num_q_heads in [0, 1] {
        let cpu = run_int8::<Cpu>(num_q_heads);
        for_each_non_cpu_backend!(|B| {
            assert_eq!(cpu, run_int8::<B>(num_q_heads), "INT8 prepare mismatch on {}", std::any::type_name::<B>());
        });
    }
}
