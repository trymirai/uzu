use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            kernel::DraftLogitScatterKernel,
        },
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_non_cpu_backend},
};

const ROWS: usize = 3;
const VOCAB: usize = 600;

// Every third token is in the draft vocabulary, in reverse order.
fn slots() -> Vec<i32> {
    let draft_size = VOCAB.div_ceil(3) as i32;
    (0..VOCAB)
        .map(|token| {
            if token % 3 == 0 {
                draft_size - 1 - (token / 3) as i32
            } else {
                -1
            }
        })
        .collect()
}

fn scatter<B: Backend>(draft_logits: &[bf16]) -> Vec<bf16> {
    let context = B::Context::new().unwrap();
    let kernel = <B::Kernels as Kernels>::DraftLogitScatterKernel::new(&context, DataType::BF16).unwrap();
    let draft_size = draft_logits.len() / ROWS;
    let draft_logits = create_buffer_with_data::<B, bf16>(&context, draft_logits);
    let slots = create_buffer_with_data::<B, i32>(&context, &slots());
    let mut logits = create_buffer_with_data::<B, bf16>(&context, &vec![bf16::ZERO; ROWS * VOCAB]);
    let mut command_buffer = context.create_command_buffer(None, None).unwrap();
    kernel.encode(
        &draft_logits,
        &slots,
        &mut logits,
        ROWS as u32,
        draft_size as u32,
        VOCAB as u32,
        &mut command_buffer,
    );
    command_buffer.end_encoding().submit().wait_until_completed().unwrap();
    buffer_to_vec::<B, bf16>(&logits)
}

#[uzu_test]
fn draft_logit_scatter_places_slots_and_masks_the_rest() {
    let draft_size = VOCAB.div_ceil(3);
    let draft_logits: Vec<bf16> =
        (0..ROWS * draft_size).map(|index| bf16::from_f32(index as f32 * 0.5 - 40.0)).collect();
    let expected = scatter::<Cpu>(&draft_logits);
    for_each_non_cpu_backend!(|B| {
        assert_eq!(scatter::<B>(&draft_logits), expected);
    });
}
