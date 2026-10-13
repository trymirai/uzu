use half::bf16;
use uzu_engine_macros::uzu_test;

#[cfg(backend = "metal")]
use super::kv_layout::Int8KVLayout;
#[cfg(backend = "metal")]
use crate::backends::common::kernel::AttentionPrepareInt8Kernel;
#[cfg(backend = "metal")]
use crate::backends::metal::Metal;
use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::AttentionPrepareKernel,
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
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        2,
        Some(1),
        2,
        None,
        Some(0),
        12,
        2,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    Output {
        queries: buffer_to_vec(&queries),
        keys: buffer_to_vec(&keys),
        values: buffer_to_vec(&values),
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
    let qg = create_buffer_with_data::<B, bf16>(&context, &qg);
    let mut queries = create_buffer::<B, bf16>(&context, 8);
    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");

    kernel.encode(
        &qg,
        &mut queries,
        None::<&mut B::GlobalBuffer>,
        None::<&mut B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        2,
        None,
        2,
        None,
        None,
        8,
        2,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().expect("Failed to wait command buffer");

    buffer_to_vec(&queries)
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

#[cfg(backend = "metal")]
#[derive(Debug, PartialEq)]
struct Int8Output {
    codes: [Vec<i16>; 3],
    scales: [bf16; 3],
}

#[cfg(backend = "metal")]
fn run_int8<B: Backend>() -> Int8Output {
    const DIM: usize = 256;
    let layout = Int8KVLayout {
        head_dim: DIM as u32,
        num_kv_heads: 1,
        padded_capacity: 64,
    };
    let rows: Vec<bf16> = (0..3 * DIM).map(|i| bf16::from_f32((i as f32 * 0.37).sin())).collect();
    let context = B::Context::new().unwrap();
    let qkvg = create_buffer_with_data::<B, bf16>(&context, &rows);
    let mut q_codes = create_buffer::<B, i8>(&context, DIM);
    let mut q_scales = create_buffer::<B, bf16>(&context, 1);
    let mut k_codes = create_buffer::<B, i8>(&context, layout.code_bytes());
    let mut v_codes = create_buffer::<B, u8>(&context, layout.code_bytes());
    let mut k_scales = create_buffer::<B, bf16>(&context, layout.scale_count());
    let mut v_scales = create_buffer::<B, bf16>(&context, layout.scale_count());
    let kernel = <B::Kernels as Kernels>::AttentionPrepareInt8Kernel::new(
        &context,
        DataType::BF16,
        DataType::F32,
        DIM as u32,
        false,
        true,
        1,
    )
    .unwrap();
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &qkvg,
        None::<&mut B::GlobalBuffer>,
        Some(&mut q_codes),
        Some(&mut q_scales),
        &mut k_codes,
        &mut k_scales,
        &mut v_codes,
        &mut v_scales,
        None::<&B::GlobalBuffer>,
        None::<&B::GlobalBuffer>,
        1,
        None,
        33,
        layout.padded_capacity,
        (3 * DIM) as u32,
        1,
        1,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();

    let q_codes = buffer_to_vec::<B, i8>(&q_codes);
    let k_codes = buffer_to_vec::<B, i8>(&k_codes);
    let v_codes = buffer_to_vec::<B, u8>(&v_codes);
    Int8Output {
        codes: [
            q_codes.into_iter().map(i16::from).collect(),
            (0..DIM).map(|c| i16::from(k_codes[layout.k_code_index(33, 0, c as u32)])).collect(),
            (0..DIM).map(|c| i16::from(v_codes[layout.v_code_index(33, 0, c as u32)]) - 128).collect(),
        ],
        scales: [(&q_scales, 0), (&k_scales, 33), (&v_scales, 33)]
            .map(|(buffer, slot)| buffer_to_vec::<B, bf16>(buffer)[slot]),
    }
}

#[cfg(backend = "metal")]
#[uzu_test]
fn test_attention_prepare_int8() {
    const SIGN_BITS: [u32; 8] =
        [0x9e3779b9, 0x85ebca6b, 0xc2b2ae35, 0x27d4eb2f, 0x165667b1, 0xbf58476d, 0x94d049bb, 0xd6e8feb8];
    const DIM: usize = 256;
    let cpu = run_int8::<Cpu>();
    let metal = run_int8::<Metal>();
    assert_eq!(cpu, metal);
    let rows: Vec<bf16> = (0..3 * DIM).map(|i| bf16::from_f32((i as f32 * 0.37).sin())).collect();
    for (head, row) in rows.chunks(DIM).enumerate() {
        let mut x: Vec<f64> =
            (0..DIM).map(|c| row[c].to_f64() * (1.0 - 2.0 * f64::from(SIGN_BITS[c / 32] >> (c % 32) & 1))).collect();
        for h in (0..8).map(|bit| 1 << bit) {
            for i in (0..DIM).filter(|i| i & h == 0) {
                (x[i], x[i + h]) = ((x[i] + x[i + h]) / 2f64.sqrt(), (x[i] - x[i + h]) / 2f64.sqrt());
            }
        }
        let scale = cpu.scales[head].to_f64();
        assert!((scale * 127.0 / x.iter().fold(0.0, |m: f64, v| m.max(v.abs())) - 1.0).abs() < 1e-2);
        for (channel, &code) in cpu.codes[head].iter().enumerate() {
            assert!(
                ((code as f64) - x[channel] / scale).abs() <= 0.51,
                "head {head} channel {channel}: {code} vs {}",
                x[channel] / scale
            );
        }
    }
}
