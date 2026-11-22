use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::{log_scaling::LogScalingParams, trie::TrieNode},
            kernel::AttentionRelativeLogitsKernel,
        },
        cpu::Cpu,
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_parity,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

const HEADS: u32 = 5;
const DIM: u32 = 16;
const EXTENT: u32 = 300;
const TOKENS: u32 = 3;
const ROW_STRIDE: u32 = HEADS * DIM + 11;

fn run<B: Backend>(
    has_log_scaling: bool,
    has_trie: bool,
) -> Vec<bf16> {
    let context = block_on(B::Context::new()).expect("create context");
    let kernel = <B::Kernels as Kernels>::AttentionRelativeLogitsKernel::new(
        &context,
        bf16::data_type(),
        HEADS,
        DIM,
        EXTENT,
        has_log_scaling,
        has_trie,
    )
    .expect("create relative logits kernel");
    // Some heads exceed the clip threshold and some stay below it.
    let states = (0..(TOKENS * ROW_STRIDE) as usize)
        .map(|i| bf16::from_f32((i as f32 * 0.61).sin() * (1.0 + (i / DIM as usize) as f32 * 0.5)))
        .collect::<Vec<_>>();
    let clip_scales = (0..(HEADS * DIM) as usize).map(|i| 0.75 + 0.5 * (i as f32 * 0.17).cos()).collect::<Vec<f32>>();
    let profiles = (0..(DIM * EXTENT) as usize).map(|i| (i as f32 * 0.013).sin()).collect::<Vec<f32>>();
    let nodes = [(0, -1), (1, 0), (1, 0)].map(|(height, parent)| TrieNode {
        trie_start: 0,
        trie_end: 0,
        height,
        parent,
    });
    let states = create_buffer_with_data::<B, bf16>(&context, &states);
    let clip_scales = create_buffer_with_data::<B, f32>(&context, &clip_scales);
    let profiles = create_buffer_with_data::<B, f32>(&context, &profiles);
    let trie = create_buffer_with_data::<B, u32>(&context, bytemuck::cast_slice(&nodes));
    let mut logits = create_buffer::<B, bf16>(&context, (HEADS * TOKENS * EXTENT) as usize);
    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &states,
        &clip_scales,
        &profiles,
        &mut logits,
        has_trie.then_some(&trie),
        TOKENS,
        ROW_STRIDE,
        2.5,
        has_log_scaling.then_some(LogScalingParams {
            n_floor: 8.0,
            alpha: 0.1,
            position_offset: 40,
        }),
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run relative logits");
    buffer_to_vec::<B, bf16>(&context, &logits)
}

#[uzu_test]
fn test_relative_logits_match_cpu() {
    for has_log_scaling in [false, true] {
        for has_trie in [false, true] {
            let expected = run::<Cpu>(has_log_scaling, has_trie);
            for_each_non_cpu_backend!(|B| {
                let actual = run::<B>(has_log_scaling, has_trie);
                assert_parity(
                    &format!("relative logits log_scaling={has_log_scaling} trie={has_trie}"),
                    &expected,
                    &actual,
                    1e-2,
                    1e-2,
                );
            });
        }
    }
}
