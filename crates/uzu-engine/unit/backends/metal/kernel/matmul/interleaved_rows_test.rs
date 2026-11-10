//! `GemmBLayout::InterleavedRows`: every kernel that reads interleaved-row W4 strips matches the row-major layout.
use QuantizationMethod::{ScaleBias, ScaleZeroPoint};
use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::quantization_method::QuantizationMethod,
            kernel::{
                ActivationTransform,
                matmul::{ActivationFormat, MatmulA, MatmulKernel, MatmulShape, packed_half_bytes},
            },
        },
        metal::{Metal, context::MetalContext, gpu_types::gemm::GemmBLayout, kernel::matmul::MatmulMetalKernel},
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data},
        matmul::{QuantBuffers, QuantInput, quant_arguments},
    },
};

fn run(
    context: &MetalContext,
    input: &QuantInput<bf16>,
    interleaved_rows: bool,
    gather_indices: Option<&<Metal as Backend>::GlobalGpuBuffer>,
    n: u32,
) -> Vec<bf16> {
    let mut matmul = MatmulMetalKernel::new(context, DataType::BF16, DataType::BF16, DataType::BF16).expect("matmul");
    let mut buffers = QuantBuffers::<Metal, bf16>::allocate(context, input);
    if interleaved_rows {
        let shape = MatmulShape::from_arguments(&quant_arguments(&mut buffers, input));
        buffers.weights_layout = matmul.optimize_weights_layout(&shape, &mut buffers.w, context).expect("relayout").0;
        assert_eq!(buffers.weights_layout, GemmBLayout::InterleavedRows);
    }
    let format = ActivationFormat::PackedHalf {
        block_k: input.group_size,
        bits: 4,
    };
    let packed = gather_indices.is_none() && (4..=8).contains(&input.m);
    let mut packed_activations =
        create_buffer::<Metal, u32>(context, packed_half_bytes(input.k, input.group_size, 4) / size_of::<u32>());
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    if packed {
        let factors = create_buffer_with_data::<Metal, i32>(context, &vec![1; input.k as usize]);
        ActivationTransform::<Metal>::quantize(context, DataType::BF16, format).expect("pack").encode_pack_half(
            &buffers.x,
            &mut packed_activations,
            &factors,
            input.m,
            input.k,
            &mut command_buffer,
        );
    }
    let mut arguments = quant_arguments(&mut buffers, input);
    if packed {
        arguments.a = MatmulA::PackedHalf {
            values: &packed_activations,
            block_k: input.group_size,
            bits: 4,
        };
    }
    arguments.gather_indices = gather_indices;
    arguments.n = n;
    matmul.encode(arguments, &mut command_buffer).expect("matmul encode");
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
    buffer_to_vec::<Metal, bf16>(context, &buffers.y)[..(input.m * n) as usize].to_vec()
}

fn assert_close(
    label: &str,
    expected: &[bf16],
    actual: &[bf16],
) {
    let rms = (expected.iter().map(|v| v.to_f32().powi(2)).sum::<f32>() / expected.len() as f32).sqrt();
    assert_eq_float(expected, actual, 2e-2 * rms, label);
}

#[uzu_test]
fn interleaved_rows_matches_row_major() {
    let context = block_on(MetalContext::new()).expect("Metal context");
    if context.supports_mxu {
        return;
    }
    for (k, group_size, method) in [(1024, 32, ScaleZeroPoint), (16384, 64, ScaleBias)] {
        for m in [1, 3, 8, 16, 64] {
            let input = QuantInput::<bf16>::new(m, k, 256, group_size, 4, method, 7).with_group_output();
            let label = format!("m={m} k={k} G{group_size} {method:?}");
            assert_close(&label, &run(&context, &input, false, None, 256), &run(&context, &input, true, None, 256));
        }
        let input = QuantInput::<bf16>::new(3, k, 256, group_size, 4, method, 11).with_group_output();
        let ids = create_buffer_with_data::<Metal, u32>(&context, &(0..120).map(|i| i * 7 % 256).collect::<Vec<u32>>());
        let label = format!("gathered k={k} G{group_size} {method:?}");
        assert_close(
            &label,
            &run(&context, &input, false, Some(&ids), 40),
            &run(&context, &input, true, Some(&ids), 40),
        );
    }
}
