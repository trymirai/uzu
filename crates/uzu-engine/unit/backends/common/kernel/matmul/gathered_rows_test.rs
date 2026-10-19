#![cfg(backend = "metal")]

use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context,
            gpu_types::QuantizationMethod,
            kernel::{
                Kernels,
                activation_transform::ACTIVATION_SCALE_GROUP_SIZE,
                matmul::{MatmulB, MatmulKernel},
            },
        },
        metal::Metal,
    },
    data_type::DataType,
    tests::{
        helpers::{buffer_to_vec, create_buffer_with_data},
        matmul::{QuantBuffers, QuantInput, quant_arguments},
        util::shared_metal_context,
    },
};

// Gathered weight rows (the draft vocabulary readout) give exactly the full matmul's columns for those rows.
#[uzu_test]
fn gathered_rows_match_the_full_matmul_columns() {
    let context = shared_metal_context();
    if !context.supports_mxu {
        return;
    }
    let (m, k, n) = (15u32, 1024u32, 8192u32);
    let input = QuantInput::<bf16>::new(m, k, n, 64, 4, QuantizationMethod::ScaleSymmetric, 7)
        .with_signed_weight_codes()
        .with_group_output()
        .with_prepared_a(ACTIVATION_SCALE_GROUP_SIZE, None);
    let rows: Vec<u32> = (0..n).rev().step_by(3).take(2048).collect();

    let run = |gathered: bool| {
        let mut buffers = QuantBuffers::<Metal, bf16>::allocate(&context, &input);
        let row_buffer = create_buffer_with_data::<Metal, u32>(&context, &rows);
        let mut matmul = <<Metal as Backend>::Kernels as Kernels>::MatmulKernel::new(
            &context,
            DataType::BF16,
            DataType::BF16,
            DataType::BF16,
        )
        .unwrap();
        let mut arguments = quant_arguments(&mut buffers, &input);
        if gathered {
            arguments.n = rows.len() as u32;
            let MatmulB::Quantized(quantized) = &mut arguments.b else {
                unreachable!()
            };
            quantized.rows = Some(&row_buffer);
        }
        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
        matmul.encode(arguments, &mut command_buffer).unwrap();
        command_buffer.end_encoding().submit().wait_until_completed().unwrap();
        buffer_to_vec::<Metal, bf16>(&buffers.y)
    };
    let full = run(false);
    let gathered = run(true);
    for token in 0..m as usize {
        for (column, &row) in rows.iter().enumerate() {
            assert_eq!(
                gathered[token * rows.len() + column].to_bits(),
                full[token * n as usize + row as usize].to_bits(),
                "token {token} column {column} (row {row})"
            );
        }
    }
}
