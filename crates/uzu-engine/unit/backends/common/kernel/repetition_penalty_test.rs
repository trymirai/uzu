use std::{collections::HashSet, fmt::Debug, ops::Range};

use futures::executor::block_on;
use half::bf16;
use num_traits::Float;
use uzu_engine_macros::uzu_test;

use crate::{
    backends::common::{
        Backend, CommandBufferEncoding, CommandBufferExecutable, CommandBufferPending, Context, Kernels,
        gpu_types::trie::TrieNode, kernel::RepetitionPenaltyKernel,
    },
    data_type::DataTypeRepr,
    encodable_block::{
        batch_topology::BatchTopology,
        sampling::{Sampling, SamplingMethod},
    },
    tests::helpers::{buffer_to_vec, create_buffer_with_data, for_each_backend},
};

const VOCAB_SIZE: usize = 16;
const TOKENS: [u32; 9] = [1, 2, 3, 2, 4, 5, 1, 6, 7];
const PATHS: [&[u32]; 9] =
    [&[1], &[1, 2], &[1, 2, 3], &[1, 2, 3, 2], &[1, 2, 4], &[1, 5], &[1, 5, 1], &[1, 5, 1, 6], &[1, 5, 7]];

fn nodes(flat: bool) -> Vec<TrieNode> {
    let ends = [8, 4, 3, 3, 4, 8, 7, 7, 8];
    let mut stack: Vec<usize> = Vec::new();
    (0..TOKENS.len())
        .map(|index| {
            let height = if flat {
                index as u32
            } else {
                PATHS[index].len() as u32 - 1
            };
            stack.truncate(height as usize);
            let parent = stack.last().map(|i| *i as i32).unwrap_or(-1);
            stack.push(index);
            TrieNode {
                trie_start: index as u32,
                trie_end: if flat {
                    8
                } else {
                    ends[index]
                },
                height,
                parent,
            }
        })
        .collect()
}

fn ring(
    history: &[u32],
    window: usize,
    offset: usize,
) -> Vec<u32> {
    let history = &history[history.len().saturating_sub(window)..];
    let mut ring = vec![0; 2 + window];
    ring[0] = offset as u32;
    ring[1] = history.len() as u32;
    for (index, &token) in history.iter().enumerate() {
        ring[2 + (offset + index) % window] = token;
    }
    ring
}

fn expected<T: DataTypeRepr + Float>(
    logits: &[T],
    history: &[u32],
    window: usize,
    range: Range<usize>,
    flat: bool,
) -> Vec<T> {
    let mut output = logits.to_vec();
    for (row, index) in range.enumerate() {
        let path = if flat {
            &TOKENS[..=index]
        } else {
            PATHS[index]
        };
        let history: Vec<_> = history.iter().chain(path).copied().collect();
        let tokens: HashSet<_> = history[history.len().saturating_sub(window)..].iter().copied().collect();
        for token in tokens {
            let offset = row * VOCAB_SIZE + token as usize;
            let logit = logits[offset].to_f32().unwrap();
            output[offset] = T::from(if logit > 0.0 {
                logit / 2.0
            } else {
                logit * 2.0
            })
            .unwrap();
        }
    }
    output
}

fn check_kernel<B: Backend, T: DataTypeRepr + Float + Debug>() {
    let context = block_on(B::Context::new()).unwrap();
    for flat in [false, true] {
        let nodes = nodes(flat);
        let kernel = <B::Kernels as Kernels>::RepetitionPenaltyKernel::new(&context, T::data_type(), !flat).unwrap();
        for window in [1, 3, 4, 16] {
            for history in [&[][..], &[8, 2][..], &[9, 1, 8, 4, 3, 2, 7, 6, 5, 2, 8, 9][..]] {
                for offset in [0, window - 1] {
                    for range in [0..9, 3..5, 7..9] {
                        let logits: Vec<T> = (0..range.len() * VOCAB_SIZE)
                            .map(|index| T::from((index as i32 % 17 - 8) as f32 * 1.25).unwrap())
                            .collect();
                        let expected = expected(&logits, history, window, range.clone(), flat);
                        let mut output = create_buffer_with_data::<B, T>(&context, &logits);
                        let logits = create_buffer_with_data::<B, T>(&context, &logits);
                        let ring = create_buffer_with_data::<B, u32>(&context, &ring(history, window, offset));
                        let tokens = create_buffer_with_data::<B, u32>(&context, &TOKENS);
                        let mut command_buffer = context.create_command_buffer(None, None).unwrap();
                        let trie = (!flat).then(|| command_buffer.allocate_constant_from_slice(&nodes).unwrap());
                        kernel.encode(
                            &logits,
                            &mut output,
                            &ring,
                            &tokens,
                            trie.as_ref(),
                            2.0,
                            window as u32,
                            VOCAB_SIZE as u32,
                            range.start as u32,
                            range.len() as u32,
                            &mut command_buffer,
                        );
                        drop(trie);
                        block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
                        assert_eq!(
                            buffer_to_vec::<B, T>(&context, &output),
                            expected,
                            "flat={flat}, window={window}, history={history:?}, offset={offset}, range={range:?}",
                        );
                    }
                }
            }
        }
    }
}

fn check_sampling<B: Backend, T: DataTypeRepr + Float + Debug>() {
    let context = block_on(B::Context::new()).unwrap();
    let sampling = Sampling::<B>::new(VOCAB_SIZE as u32, T::data_type());
    let method = SamplingMethod::Stochastic {
        temperature: None,
        top_k: Some(1),
        top_p: None,
        min_p: None,
        repetition_penalty: Some(2.0),
        suffix_repetition_length: Some(3),
    };
    for flat in [false, true] {
        let nodes = nodes(flat);
        let topology = BatchTopology::new(nodes.into_boxed_slice(), flat);
        for range in [0..9, 3..5, 7..9] {
            let mut logits = vec![T::from(-16.0).unwrap(); range.len() * VOCAB_SIZE];
            for row in logits.as_chunks_mut::<VOCAB_SIZE>().0 {
                row[2] = T::from(10.0).unwrap();
                row[5] = T::from(9.0).unwrap();
                row[7] = T::from(8.0).unwrap();
            }
            let expected: Vec<_> = expected(&logits, &[2], 3, range.clone(), flat)
                .as_chunks::<VOCAB_SIZE>()
                .0
                .iter()
                .map(|row| row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32)
                .collect();
            let logits = create_buffer_with_data::<B, T>(&context, &logits);
            let ring = create_buffer_with_data::<B, u32>(&context, &ring(&[2], 3, 2));
            let tokens = create_buffer_with_data::<B, u32>(&context, &TOKENS);
            let seeds = create_buffer_with_data::<B, u64>(&context, &vec![0; range.len()]);
            let mut command_buffer = context.create_command_buffer(None, None).unwrap();
            let output = sampling
                .encode(
                    &logits,
                    Some(&seeds),
                    None::<&B::GlobalGpuBuffer>,
                    Some(&ring),
                    Some(&tokens),
                    &topology,
                    (range.start as u32..range.end as u32).into(),
                    &method,
                    &mut command_buffer,
                )
                .unwrap();
            block_on(command_buffer.end_encoding().submit().wait_until_completed()).unwrap();
            assert_eq!(buffer_to_vec::<B, u32>(&context, &output), expected, "flat={flat}, range={range:?}");
        }
    }
}

#[uzu_test]
fn repetition_penalty_f32() {
    for_each_backend!(|B| check_kernel::<B, f32>());
}

#[uzu_test]
fn repetition_penalty_bf16() {
    for_each_backend!(|B| check_kernel::<B, bf16>());
}

#[uzu_test]
fn repetition_penalty_sampling_trie() {
    for_each_backend!(|B| {
        check_sampling::<B, f32>();
        check_sampling::<B, bf16>();
    });
}
