use futures::executor::block_on;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::RowGatherKernel,
        },
        cpu::Cpu,
    },
    tests::helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
};

fn run_row_gather<B: Backend>(
    src: &[u8],
    accepted_indices: &[u32],
    row_bytes: u32,
) -> Vec<u8> {
    let context = block_on(B::Context::new()).expect("create context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::RowGatherKernel::new(&context).expect("create row gather kernel");

    let src = create_buffer_with_data::<B, u8>(&context, src);
    let accepted_indices = create_buffer_with_data::<B, u32>(&context, accepted_indices);
    let mut dst = create_buffer::<B, u8>(&context, row_bytes as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(&mut dst, &src, &accepted_indices, row_bytes, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();

    buffer_to_vec::<B, u8>(&context, &dst)
}

#[uzu_test]
fn row_gather_copies_the_row_named_by_word_one() {
    let src: Vec<u8> = vec![1, 1, 2, 2, 3, 3];
    let accepted_indices: Vec<u32> = vec![2, 2, 0];

    let expected = vec![3, 3];
    let actual = run_row_gather::<Cpu>(&src, &accepted_indices, 2);
    assert_eq!(actual, expected);

    for_each_non_cpu_backend!(|B| {
        let actual = run_row_gather::<B>(&src, &accepted_indices, 2);
        assert_eq!(actual, expected);
    });
}
