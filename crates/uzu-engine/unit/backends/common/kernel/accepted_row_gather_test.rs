use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::AcceptedRowGatherKernel,
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
};

fn run_accepted_row_gather<B: Backend>(
    src: &[u8],
    accepted_indices: &[u32],
    num_tokens: u32,
    dst_stride: u32,
    row_bytes: u32,
) -> Vec<u8> {
    let context = block_on(B::Context::new()).expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::AcceptedRowGatherKernel::new(&context)
        .expect("create accepted row gather kernel");

    let src = create_buffer_with_data::<B, u8>(&context, src);
    let accepted_indices = create_buffer_with_data::<B, u32>(&context, accepted_indices);
    let mut dst = create_buffer::<B, u8>(&context, num_tokens as usize * dst_stride as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(&mut dst, &src, &accepted_indices, num_tokens, dst_stride, row_bytes, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    buffer_to_vec::<B, u8>(&context, &dst)
}

#[uzu_test]
fn accepted_row_gather_copies_accepted_rows_in_rank_order() {
    // 3 rows of 2 bytes each
    let src: Vec<u8> = vec![1, 1, 2, 2, 3, 3];
    // [len][indices...], deepest accepted row at word 1: accepted ranks [0, 2] shallow to deep
    let accepted_indices: Vec<u32> = vec![2, 2, 0];
    let num_tokens = 2;
    let dst_stride = 4;
    let row_bytes = 2;

    let mut expected = vec![0u8; num_tokens as usize * dst_stride as usize];
    expected[0..2].copy_from_slice(&[1, 1]);
    expected[4..6].copy_from_slice(&[3, 3]);

    let actual = run_accepted_row_gather::<Cpu>(&src, &accepted_indices, num_tokens, dst_stride, row_bytes);
    assert_eq!(actual, expected);

    for_each_non_cpu_backend!(|B| {
        let actual = run_accepted_row_gather::<B>(&src, &accepted_indices, num_tokens, dst_stride, row_bytes);
        assert_eq!(actual, expected);
    });
}
