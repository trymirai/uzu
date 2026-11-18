use std::fmt::{Debug, Display};

use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Backend, Context, Kernels,
            gpu_types::trie::TrieNode,
            kernel::{NGramConvInjectKernel, NGramConvInjectTrieKernel, NGramGateKernel},
        },
        cpu::Cpu,
    },
    data_type::{DataType, DataTypeRepr},
    tests::{
        assert::assert_eq_float,
        helpers::{
            buffer_to_vec, create_buffer, create_buffer_with_data, create_context, for_each_non_cpu_backend,
            submit_command_buffer,
        },
    },
};

fn values<T: DataTypeRepr + Float>(
    count: usize,
    seed: f32,
) -> Vec<T> {
    (0..count).map(|i| T::from(((i as f32 * 0.37 + seed).sin()) * 1.5).unwrap()).collect()
}

fn run_gate<T: DataTypeRepr + Float, B: Backend>(
    num_tokens: u32,
    stream_count: u32,
    model_dim: u32,
) -> (Vec<T>, Vec<T>) {
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::NGramGateKernel::new(&context, T::data_type())
        .expect("Failed to create NGramGateKernel");
    let rows = (num_tokens * stream_count) as usize;
    let dim = model_dim as usize;

    let keys = create_buffer_with_data::<B, T>(&context, &values::<T>(rows * dim, 0.1));
    let stream = create_buffer_with_data::<B, T>(&context, &values::<T>(rows * dim, 1.3));
    let value_rows = create_buffer_with_data::<B, T>(&context, &values::<T>(num_tokens as usize * dim, 2.7));
    let scales = |seed| create_buffer_with_data::<B, f32>(&context, &values::<f32>(stream_count as usize * dim, seed));
    let (key_scales, query_scales, conv_scales) = (scales(0.5), scales(0.9), scales(1.7));
    let mut gated = create_buffer::<B, T>(&context, rows * dim);
    let mut conv_input = create_buffer::<B, T>(&context, rows * dim);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &keys,
        &stream,
        &value_rows,
        &key_scales,
        &query_scales,
        &conv_scales,
        &mut gated,
        &mut conv_input,
        num_tokens,
        stream_count,
        model_dim,
        1e-6,
        0.0,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    (buffer_to_vec::<B, T>(&context, &gated), buffer_to_vec::<B, T>(&context, &conv_input))
}

struct ConvInput<T> {
    conv_input: Vec<T>,
    state: Vec<T>,
    weights: Vec<f32>,
    gated: Vec<T>,
    stream: Vec<T>,
    num_tokens: u32,
    channels: u32,
    kernel_size: u32,
}

impl<T: DataTypeRepr + Float> ConvInput<T> {
    fn new(
        num_tokens: u32,
        channels: u32,
        kernel_size: u32,
    ) -> Self {
        let tokens = num_tokens as usize;
        let channels_count = channels as usize;
        let state_rows = kernel_size as usize - 1;
        Self {
            conv_input: values::<T>(tokens * channels_count, 0.2),
            state: values::<T>(state_rows * channels_count, 3.1),
            weights: values::<f32>(channels_count * kernel_size as usize, 0.8),
            gated: values::<T>(tokens * channels_count, 4.4),
            stream: values::<T>(tokens * channels_count, 5.5),
            num_tokens,
            channels,
            kernel_size,
        }
    }

    /// The input restricted to the rows of `tokens`, in that order, as one flat sequence.
    fn rows(
        &self,
        tokens: &[usize],
    ) -> Self {
        let channels = self.channels as usize;
        let gather = |data: &[T]| {
            tokens.iter().flat_map(|&token| data[token * channels..(token + 1) * channels].iter().copied()).collect()
        };
        Self {
            conv_input: gather(&self.conv_input),
            state: self.state.clone(),
            weights: self.weights.clone(),
            gated: gather(&self.gated),
            stream: gather(&self.stream),
            num_tokens: tokens.len() as u32,
            channels: self.channels,
            kernel_size: self.kernel_size,
        }
    }

    fn state_rows(&self) -> usize {
        self.kernel_size as usize - 1
    }
}

fn run_conv_on<T: DataTypeRepr + Float, B: Backend>(input: &ConvInput<T>) -> (Vec<T>, Vec<T>) {
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::NGramConvInjectKernel::new(&context, T::data_type())
        .expect("Failed to create NGramConvInjectKernel");

    let conv_input = create_buffer_with_data::<B, T>(&context, &input.conv_input);
    let state = create_buffer_with_data::<B, T>(&context, &input.state);
    let weights = create_buffer_with_data::<B, f32>(&context, &input.weights);
    let gated = create_buffer_with_data::<B, T>(&context, &input.gated);
    let mut stream = create_buffer_with_data::<B, T>(&context, &input.stream);
    let mut next_state = create_buffer::<B, T>(&context, input.state_rows() * input.channels as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &conv_input,
        &state,
        &weights,
        &gated,
        &mut stream,
        &mut next_state,
        input.num_tokens,
        input.channels,
        input.kernel_size,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    (buffer_to_vec::<B, T>(&context, &stream), buffer_to_vec::<B, T>(&context, &next_state))
}

fn run_conv<T: DataTypeRepr + Float, B: Backend>(
    num_tokens: u32,
    channels: u32,
    kernel_size: u32,
) -> (Vec<T>, Vec<T>) {
    run_conv_on::<T, B>(&ConvInput::new(num_tokens, channels, kernel_size))
}

/// Returns the stream and the per-node states, `[num_tokens, kernel_size - 1, channels]`.
fn run_conv_trie_on<T: DataTypeRepr + Float, B: Backend>(
    input: &ConvInput<T>,
    parents: &[i32],
) -> (Vec<T>, Vec<T>) {
    assert_eq!(parents.len(), input.num_tokens as usize);
    let context = create_context::<B>();
    let kernel = <B::Kernels as Kernels>::NGramConvInjectTrieKernel::new(&context, T::data_type())
        .expect("Failed to create NGramConvInjectTrieKernel");

    let trie = parents
        .iter()
        .enumerate()
        .map(|(node, &parent)| TrieNode {
            trie_start: node as u32,
            trie_end: node as u32,
            height: 0,
            parent,
        })
        .collect::<Vec<_>>();
    let trie = create_buffer_with_data::<B, u32>(&context, bytemuck::cast_slice(&trie));

    let conv_input = create_buffer_with_data::<B, T>(&context, &input.conv_input);
    let state = create_buffer_with_data::<B, T>(&context, &input.state);
    let weights = create_buffer_with_data::<B, f32>(&context, &input.weights);
    let gated = create_buffer_with_data::<B, T>(&context, &input.gated);
    let mut stream = create_buffer_with_data::<B, T>(&context, &input.stream);
    let mut node_states =
        create_buffer::<B, T>(&context, input.num_tokens as usize * input.state_rows() * input.channels as usize);

    let mut command_buffer = context.create_command_buffer(None, None).expect("Failed to create command buffer");
    kernel.encode(
        &conv_input,
        &state,
        &trie,
        &weights,
        &gated,
        &mut stream,
        &mut node_states,
        input.num_tokens,
        input.channels,
        input.kernel_size,
        &mut command_buffer,
    );
    submit_command_buffer(command_buffer);
    (buffer_to_vec::<B, T>(&context, &stream), buffer_to_vec::<B, T>(&context, &node_states))
}

/// A trie over `num_tokens` nodes where node `i > 0` hangs off node `(i - 1) / 2`, so most nodes have two children.
fn binary_parents(num_tokens: u32) -> Box<[i32]> {
    (0..num_tokens as i32)
        .map(|node| {
            if node == 0 {
                -1
            } else {
                (node - 1) / 2
            }
        })
        .collect()
}

fn tolerance<T: DataTypeRepr>() -> f32 {
    match T::data_type() {
        DataType::F32 => 1e-4,
        _ => 0.05,
    }
}

fn test_ngram_gate<T: DataTypeRepr + Float + Debug + Display>() {
    for (num_tokens, stream_count, model_dim) in [(1, 4, 64), (5, 2, 1000)] {
        let (expected_gated, expected_conv_input) = run_gate::<T, Cpu>(num_tokens, stream_count, model_dim);
        for_each_non_cpu_backend!(|B| {
            let (gated, conv_input) = run_gate::<T, B>(num_tokens, stream_count, model_dim);
            let message = format!("NGramGate failed with backend={}", std::any::type_name::<B>());
            assert_eq_float::<T>(&expected_gated, &gated, tolerance::<T>(), &format!("{message} (gated)"));
            assert_eq_float::<T>(
                &expected_conv_input,
                &conv_input,
                tolerance::<T>(),
                &format!("{message} (conv input)"),
            );
        });
    }
}

fn test_ngram_conv_inject<T: DataTypeRepr + Float + Debug + Display>() {
    for (num_tokens, channels, kernel_size) in [(1, 256, 10), (3, 300, 10), (17, 64, 4)] {
        let (expected_stream, expected_state) = run_conv::<T, Cpu>(num_tokens, channels, kernel_size);
        for_each_non_cpu_backend!(|B| {
            let (stream, next_state) = run_conv::<T, B>(num_tokens, channels, kernel_size);
            let message = format!("NGramConvInject failed with backend={}", std::any::type_name::<B>());
            assert_eq_float::<T>(&expected_stream, &stream, tolerance::<T>(), &format!("{message} (stream)"));
            assert_eq_float::<T>(&expected_state, &next_state, tolerance::<T>(), &format!("{message} (state)"));
        });
    }
}

#[uzu_test]
fn test_ngram_gate_f32() {
    test_ngram_gate::<f32>();
}

#[uzu_test]
fn test_ngram_gate_bf16() {
    test_ngram_gate::<bf16>();
}

fn test_ngram_conv_inject_trie<T: DataTypeRepr + Float + Debug + Display>() {
    for (num_tokens, channels, kernel_size) in [(1, 256, 10), (6, 300, 10), (17, 64, 4)] {
        let input = ConvInput::<T>::new(num_tokens, channels, kernel_size);
        let parents = binary_parents(num_tokens);
        let (expected_stream, expected_states) = run_conv_trie_on::<T, Cpu>(&input, &parents);
        for_each_non_cpu_backend!(|B| {
            let (stream, node_states) = run_conv_trie_on::<T, B>(&input, &parents);
            let message = format!("NGramConvInjectTrie failed with backend={}", std::any::type_name::<B>());
            assert_eq_float::<T>(&expected_stream, &stream, tolerance::<T>(), &format!("{message} (stream)"));
            assert_eq_float::<T>(&expected_states, &node_states, tolerance::<T>(), &format!("{message} (states)"));
        });
    }
}

#[uzu_test]
fn test_ngram_conv_inject_f32() {
    test_ngram_conv_inject::<f32>();
}

#[uzu_test]
fn test_ngram_conv_inject_bf16() {
    test_ngram_conv_inject::<bf16>();
}

#[uzu_test]
fn test_ngram_conv_inject_trie_f32() {
    test_ngram_conv_inject_trie::<f32>();
}

#[uzu_test]
fn test_ngram_conv_inject_trie_bf16() {
    test_ngram_conv_inject_trie::<bf16>();
}

/// Every node of a trie must see exactly what the flat kernel sees on that node's root-to-node path: the same
/// injected row, and a state slot equal to the flat kernel's next state after the path.
#[uzu_test]
fn test_ngram_conv_inject_trie_matches_flat_paths() {
    let parents = [-1, 0, 1, 0, 3, 3, 2];
    let (channels, kernel_size) = (16, 4);
    let input = ConvInput::<f32>::new(parents.len() as u32, channels, kernel_size);
    let (stream, node_states) = run_conv_trie_on::<f32, Cpu>(&input, &parents);

    let channels = channels as usize;
    let slot = input.state_rows() * channels;
    for node in 0..parents.len() {
        let mut path = vec![node];
        while let Ok(parent) = usize::try_from(parents[*path.last().unwrap()]) {
            path.push(parent);
        }
        path.reverse();

        let (flat_stream, flat_state) = run_conv_on::<f32, Cpu>(&input.rows(&path));
        let message = format!("node {node} with path {path:?}");
        assert_eq_float::<f32>(
            &flat_stream[(path.len() - 1) * channels..],
            &stream[node * channels..(node + 1) * channels],
            tolerance::<f32>(),
            &format!("{message} (stream)"),
        );
        assert_eq_float::<f32>(
            &flat_state,
            &node_states[node * slot..(node + 1) * slot],
            tolerance::<f32>(),
            &format!("{message} (state)"),
        );
    }
}
