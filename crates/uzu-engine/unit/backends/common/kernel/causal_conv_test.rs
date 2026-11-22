use futures::executor::block_on;
use half::bf16;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
            gpu_types::trie::TrieNode,
            kernel::{CausalConvKernel, CausalConvShiftKernel, CausalConvTokensKernel},
        },
        cpu::Cpu,
    },
    data_type::DataTypeRepr,
    tests::{
        assert::assert_parity,
        helpers::{buffer_to_vec, create_buffer, create_buffer_with_data, for_each_non_cpu_backend},
    },
};

const CHANNELS: u32 = 300;
const KERNEL_SIZE: u32 = 4;
const GROUP_SIZE: u32 = 100;
const TOKENS: u32 = 5;
const INPUT_STRIDE: u32 = CHANNELS + 7;
const TAPS_STRIDE: u32 = (CHANNELS / GROUP_SIZE) * KERNEL_SIZE + 3;

struct ConvOutput {
    output: Vec<bf16>,
    state: Vec<f32>,
    node_states: Vec<f32>,
}

fn value(index: usize, salt: f32) -> f32 {
    ((index as f32 * 0.731 + salt).sin() * 1.7).clamp(-1.5, 1.5)
}

fn run<B: Backend>(
    has_taps: bool,
    has_identity: bool,
    trie: Option<&[TrieNode]>,
) -> ConvOutput {
    let context = block_on(B::Context::new()).expect("create context");
    let kernel = <B::Kernels as Kernels>::CausalConvKernel::new(
        &context,
        bf16::data_type(),
        CHANNELS,
        KERNEL_SIZE,
        GROUP_SIZE,
        has_identity,
        has_taps,
        trie.is_some(),
    )
    .expect("create causal convolution kernel");

    let input = (0..(TOKENS * INPUT_STRIDE) as usize).map(|i| bf16::from_f32(value(i, 0.3))).collect::<Vec<_>>();
    let taps = (0..(TOKENS * TAPS_STRIDE) as usize).map(|i| bf16::from_f32(0.25 * value(i, 1.1))).collect::<Vec<_>>();
    let weights = (0..(CHANNELS * KERNEL_SIZE) as usize).map(|i| 0.5 * value(i, 2.9)).collect::<Vec<f32>>();
    let state = (0..(CHANNELS * (KERNEL_SIZE - 1)) as usize).map(|i| value(i, 4.2)).collect::<Vec<f32>>();

    let input = create_buffer_with_data::<B, bf16>(&context, &input);
    let taps = create_buffer_with_data::<B, bf16>(&context, &taps);
    let weights = create_buffer_with_data::<B, f32>(&context, &weights);
    let mut state = create_buffer_with_data::<B, f32>(&context, &state);
    let nodes = trie.map(|nodes| create_buffer_with_data::<B, u32>(&context, bytemuck::cast_slice(nodes)));
    let mut node_states =
        create_buffer::<B, f32>(&context, (TOKENS * CHANNELS * (KERNEL_SIZE - 1)) as usize);
    let mut output = create_buffer::<B, bf16>(&context, (TOKENS * CHANNELS) as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    kernel.encode(
        &input,
        has_taps.then_some(&taps),
        &weights,
        &mut state,
        nodes.as_ref(),
        trie.is_some().then_some(&mut node_states),
        &mut output,
        TOKENS,
        INPUT_STRIDE,
        TAPS_STRIDE,
        0.4,
        0,
        &mut command_buffer,
    );
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run causal convolution");

    ConvOutput {
        output: buffer_to_vec::<B, bf16>(&context, &output),
        state: buffer_to_vec::<B, f32>(&context, &state),
        node_states: if trie.is_some() {
            buffer_to_vec::<B, f32>(&context, &node_states)
        } else {
            Vec::new()
        },
    }
}

fn node(
    height: u32,
    parent: i32,
) -> TrieNode {
    TrieNode {
        trie_start: 0,
        trie_end: 0,
        height,
        parent,
    }
}

#[uzu_test]
fn test_causal_conv_matches_cpu() {
    let trie = [node(0, -1), node(1, 0), node(1, 0), node(2, 1), node(3, 3)];
    for trie in [None, Some(trie.as_slice())] {
        for has_taps in [false, true] {
            for has_identity in [false, true] {
                let expected = run::<Cpu>(has_taps, has_identity, trie);
                for_each_non_cpu_backend!(|B| {
                    let actual = run::<B>(has_taps, has_identity, trie);
                    let label = format!("trie={} taps={has_taps} identity={has_identity}", trie.is_some());
                    assert_parity(&format!("{label} output"), &expected.output, &actual.output, 1e-2, 1e-2);
                    assert_parity(&format!("{label} state"), &expected.state, &actual.state, 1e-6, 1e-6);
                    assert_parity(
                        &format!("{label} node states"),
                        &expected.node_states,
                        &actual.node_states,
                        1e-6,
                        1e-6,
                    );
                });
            }
        }
    }
}

/// Convolving a batch in two chunks through the state matches convolving it at once.
#[uzu_test]
fn test_causal_conv_state_carries_over() {
    let whole = run::<Cpu>(true, true, None);
    let trie = [node(0, -1), node(1, 0), node(2, 1), node(3, 2), node(4, 3)];
    let chain = run::<Cpu>(true, true, Some(&trie));
    assert_parity("chain trie output", &whole.output, &chain.output, 0.0, 0.0);
    let history = (KERNEL_SIZE - 1) as usize;
    let last_node = &chain.node_states[(TOKENS - 1) as usize * CHANNELS as usize * history..];
    assert_parity("chain trie last node", &whole.state, last_node, 0.0, 0.0);
}

/// The token-parallel convolution and the state shift that follows it.
fn run_tokens<B: Backend>(
    has_taps: bool,
    has_identity: bool,
) -> ConvOutput {
    let context = block_on(B::Context::new()).expect("create context");
    let tokens_kernel = <B::Kernels as Kernels>::CausalConvTokensKernel::new(
        &context,
        bf16::data_type(),
        CHANNELS,
        KERNEL_SIZE,
        GROUP_SIZE,
        has_identity,
        has_taps,
    )
    .expect("create token-parallel causal convolution kernel");
    let shift_kernel =
        <B::Kernels as Kernels>::CausalConvShiftKernel::new(&context, bf16::data_type(), CHANNELS, KERNEL_SIZE)
            .expect("create causal convolution shift kernel");

    let input = (0..(TOKENS * INPUT_STRIDE) as usize).map(|i| bf16::from_f32(value(i, 0.3))).collect::<Vec<_>>();
    let taps = (0..(TOKENS * TAPS_STRIDE) as usize).map(|i| bf16::from_f32(0.25 * value(i, 1.1))).collect::<Vec<_>>();
    let weights = (0..(CHANNELS * KERNEL_SIZE) as usize).map(|i| 0.5 * value(i, 2.9)).collect::<Vec<f32>>();
    let state = (0..(CHANNELS * (KERNEL_SIZE - 1)) as usize).map(|i| value(i, 4.2)).collect::<Vec<f32>>();
    let input = create_buffer_with_data::<B, bf16>(&context, &input);
    let taps = create_buffer_with_data::<B, bf16>(&context, &taps);
    let weights = create_buffer_with_data::<B, f32>(&context, &weights);
    let mut state = create_buffer_with_data::<B, f32>(&context, &state);
    let mut output = create_buffer::<B, bf16>(&context, (TOKENS * CHANNELS) as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("command buffer");
    tokens_kernel.encode(
        &input,
        has_taps.then_some(&taps),
        &weights,
        &state,
        &mut output,
        TOKENS,
        INPUT_STRIDE,
        TAPS_STRIDE,
        0.4,
        0,
        &mut command_buffer,
    );
    shift_kernel.encode(&input, &mut state, TOKENS, INPUT_STRIDE, &mut command_buffer);
    block_on(command_buffer.end_encoding().submit().wait_until_completed()).expect("run causal convolution");

    ConvOutput {
        output: buffer_to_vec::<B, bf16>(&context, &output),
        state: buffer_to_vec::<B, f32>(&context, &state),
        node_states: Vec::new(),
    }
}

#[uzu_test]
fn test_causal_conv_tokens_match_sequential() {
    for has_taps in [false, true] {
        for has_identity in [false, true] {
            let expected = run::<Cpu>(has_taps, has_identity, None);
            let label = format!("taps={has_taps} identity={has_identity}");
            let cpu = run_tokens::<Cpu>(has_taps, has_identity);
            assert_parity(&format!("{label} cpu output"), &expected.output, &cpu.output, 0.0, 0.0);
            assert_parity(&format!("{label} cpu state"), &expected.state, &cpu.state, 0.0, 0.0);
            for_each_non_cpu_backend!(|B| {
                let actual = run_tokens::<B>(has_taps, has_identity);
                assert_parity(&format!("{label} output"), &expected.output, &actual.output, 1e-2, 1e-2);
                assert_parity(&format!("{label} state"), &expected.state, &actual.state, 1e-6, 1e-6);
            });
        }
    }
}
