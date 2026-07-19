#![cfg(metal_backend)]

use std::{sync::Arc, time::Duration};

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Allocation, Backend, Encoder, Kernels, kernel::AttentionLastQueryKernel},
        cpu::Cpu,
        metal::Metal,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec, create_context},
    },
};

const HEAD_DIM: usize = 128;
const NUM_HEADS: u32 = 16;

struct Runner<B: Backend> {
    context: Arc<B::Context>,
    kernel: <B::Kernels as Kernels>::AttentionLastQueryKernel,
    prefix_qkv: Allocation<B>,
    node_qkv: Allocation<B>,
    current_qkv: Allocation<B>,
    ancestor_indices: Allocation<B>,
    ancestor_counts: Allocation<B>,
    node_indices: Allocation<B>,
    output: Allocation<B>,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
}

impl<B: Backend> Runner<B> {
    fn new(
        rows: usize,
        prefix_length: usize,
        ancestor_stride: usize,
        nodes: usize,
    ) -> Self {
        assert!(rows < nodes && ancestor_stride > 0);
        let packed_width = 3 * NUM_HEADS as usize * HEAD_DIM;
        let values = |length, offset| {
            (0..length)
                .map(|index| bf16::from_f32((((index + offset) * 17 % 251) as f32 - 125.0) / 128.0))
                .collect::<Vec<_>>()
        };
        let first_output_node = nodes - rows;
        let ancestor_counts = (0..rows).map(|row| (row % (ancestor_stride + 1)) as u32).collect::<Vec<_>>();
        let mut ancestor_indices = vec![0; rows * ancestor_stride];
        for row in 0..rows {
            for offset in 0..ancestor_counts[row] as usize {
                ancestor_indices[row * ancestor_stride + offset] =
                    ((row * ancestor_stride + offset) % first_output_node) as u32;
            }
        }

        let context = create_context::<B>();
        let kernel =
            <B::Kernels as Kernels>::AttentionLastQueryKernel::new(context.as_ref(), HEAD_DIM as u32, NUM_HEADS)
                .unwrap();
        Self {
            prefix_qkv: alloc_allocation_with_data::<B, bf16>(&context, &values(prefix_length * packed_width, 0)),
            node_qkv: alloc_allocation_with_data::<B, bf16>(&context, &values(nodes * packed_width, 11)),
            current_qkv: alloc_allocation_with_data::<B, bf16>(&context, &values(rows * packed_width, 29)),
            ancestor_indices: alloc_allocation_with_data::<B, u32>(&context, &ancestor_indices),
            ancestor_counts: alloc_allocation_with_data::<B, u32>(&context, &ancestor_counts),
            node_indices: alloc_allocation_with_data::<B, u32>(
                &context,
                &(first_output_node as u32..nodes as u32).collect::<Vec<_>>(),
            ),
            output: alloc_allocation::<B, bf16>(&context, rows * NUM_HEADS as usize * HEAD_DIM),
            context,
            kernel,
            rows: rows as u32,
            prefix_length: prefix_length as u32,
            ancestor_stride: ancestor_stride as u32,
        }
    }

    fn encode(
        &mut self,
        repetitions: u32,
    ) -> Duration {
        let mut encoder = Encoder::new(self.context.as_ref()).unwrap();
        for _ in 0..repetitions {
            self.kernel.encode(
                &self.prefix_qkv,
                &mut self.node_qkv,
                &self.current_qkv,
                &self.ancestor_indices,
                &self.ancestor_counts,
                &self.node_indices,
                &mut self.output,
                self.rows,
                self.prefix_length,
                self.ancestor_stride,
                1.0 / (HEAD_DIM as f32).sqrt(),
                &mut encoder,
            );
        }
        encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
    }
}

#[uzu_test]
fn attention_last_query_matches_cpu() {
    let mut cpu = Runner::<Cpu>::new(4, 5, 3, 16);
    let initial_nodes = allocation_to_vec::<Cpu, bf16>(&cpu.node_qkv);
    let current_qkv = allocation_to_vec::<Cpu, bf16>(&cpu.current_qkv);
    let node_indices = allocation_to_vec::<Cpu, u32>(&cpu.node_indices);
    cpu.encode(1);
    let expected_output = allocation_to_vec::<Cpu, bf16>(&cpu.output);
    let expected_nodes = allocation_to_vec::<Cpu, bf16>(&cpu.node_qkv);

    let packed_width = 3 * NUM_HEADS as usize * HEAD_DIM;
    let q_width = NUM_HEADS as usize * HEAD_DIM;
    for node in 0..expected_nodes.len() / packed_width {
        let range = node * packed_width..(node + 1) * packed_width;
        if let Some(row) = node_indices.iter().position(|&index| index as usize == node) {
            let current = row * packed_width;
            assert_eq!(
                expected_nodes[range.start..range.start + q_width],
                initial_nodes[range.start..range.start + q_width]
            );
            assert_eq!(
                expected_nodes[range.start + q_width..range.end],
                current_qkv[current + q_width..current + packed_width],
            );
        } else {
            assert_eq!(expected_nodes[range.clone()], initial_nodes[range]);
        }
    }

    let mut metal = Runner::<Metal>::new(4, 5, 3, 16);
    metal.encode(1);
    let actual_output = allocation_to_vec::<Metal, bf16>(&metal.output);
    let actual_nodes = allocation_to_vec::<Metal, bf16>(&metal.node_qkv);

    assert_eq_float(&expected_output, &actual_output, 0.02, "AttentionLastQuery output");
    assert_eq!(expected_nodes, actual_nodes);
}

#[uzu_test]
#[ignore = "benchmark"]
fn benchmark_attention_last_query() {
    const BATCH: u32 = 32;
    const SAMPLES: usize = 50;

    let mut runner = Runner::<Metal>::new(8, 16, 8, 65);
    let mut run = || runner.encode(BATCH).div_f64(BATCH as f64);
    for _ in 0..5 {
        run();
    }
    let mut samples = (0..SAMPLES).map(|_| run()).collect::<Vec<_>>();
    samples.sort_unstable();
    eprintln!("attention_last_query gpu={:?}", samples[SAMPLES / 2]);
}
