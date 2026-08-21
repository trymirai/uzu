#![cfg(backend = "metal")]

use std::{sync::Arc, time::Duration};

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{
            Allocation, Backend, Encoder, Kernels, gpu_types::weaver::MetadataIdx, kernel::AncestorAttentionKernel,
        },
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
const MAX_DEPTH: u32 = 8;

struct Runner<B: Backend> {
    context: Arc<B::Context>,
    kernel: <B::Kernels as Kernels>::AncestorAttentionKernel,
    prefix_kv: Allocation<B>,
    node_kv: Allocation<B>,
    current_qkv: Allocation<B>,
    cosines: Allocation<B>,
    sines: Allocation<B>,
    node_metadata: Allocation<B>,
    ancestor_indices: Allocation<B>,
    ancestor_counts: Allocation<B>,
    node_indices: Allocation<B>,
    output: Allocation<B>,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    node_capacity: u32,
}

impl<B: Backend> Runner<B> {
    fn new(
        rows: usize,
        prefix_length: usize,
        ancestor_stride: usize,
        nodes: usize,
    ) -> Self {
        assert!(rows < nodes && ancestor_stride > 0);
        let model_dim = NUM_HEADS as usize * HEAD_DIM;
        let qkv_width = 3 * model_dim;
        let kv_width = 2 * model_dim;
        let values = |length, offset| {
            (0..length)
                .map(|index| bf16::from_f32((((index + offset) * 17 % 251) as f32 - 125.0) / 128.0))
                .collect::<Vec<_>>()
        };
        let first_output_node = nodes - rows;
        let node_indices = (first_output_node..nodes).map(|node| node as u32).collect::<Vec<_>>();
        let ancestor_counts = (0..rows).map(|row| (row % (ancestor_stride + 1)) as u32).collect::<Vec<_>>();
        let mut ancestor_indices = vec![0; rows * ancestor_stride];
        for row in 0..rows {
            for offset in 0..ancestor_counts[row] as usize {
                ancestor_indices[row * ancestor_stride + offset] =
                    ((row * ancestor_stride + offset) % first_output_node) as u32;
            }
        }

        let half_dim = HEAD_DIM / 2;
        let mut cosines = vec![0.0f32; (MAX_DEPTH as usize + 1) * HEAD_DIM];
        let mut sines = vec![0.0f32; (MAX_DEPTH as usize + 1) * HEAD_DIM];
        for position in 0..=MAX_DEPTH as usize {
            for pair in 0..half_dim {
                let angle = position as f32 / 10_000f32.powf(2.0 * pair as f32 / HEAD_DIM as f32);
                let row_base = position * HEAD_DIM;
                cosines[row_base + pair] = angle.cos();
                cosines[row_base + half_dim + pair] = angle.cos();
                sines[row_base + pair] = angle.sin();
                sines[row_base + half_dim + pair] = angle.sin();
            }
        }
        let mut node_metadata = vec![0u32; rows * MetadataIdx::COUNT];
        for row in 0..rows {
            node_metadata[MetadataIdx::Depth as usize * rows + row] = (row as u32 * 3) % MAX_DEPTH;
        }

        let context = create_context::<B>();
        let kernel =
            <B::Kernels as Kernels>::AncestorAttentionKernel::new(context.as_ref(), HEAD_DIM as u32, NUM_HEADS)
                .unwrap();
        Self {
            prefix_kv: alloc_allocation_with_data::<B, bf16>(&context, &values(prefix_length * kv_width, 0)),
            node_kv: alloc_allocation_with_data::<B, bf16>(&context, &values(nodes * kv_width, 11)),
            current_qkv: alloc_allocation_with_data::<B, bf16>(&context, &values(rows * qkv_width, 29)),
            cosines: alloc_allocation_with_data::<B, f32>(&context, &cosines),
            sines: alloc_allocation_with_data::<B, f32>(&context, &sines),
            node_metadata: alloc_allocation_with_data::<B, u32>(&context, &node_metadata),
            ancestor_indices: alloc_allocation_with_data::<B, u32>(&context, &ancestor_indices),
            ancestor_counts: alloc_allocation_with_data::<B, u32>(&context, &ancestor_counts),
            node_indices: alloc_allocation_with_data::<B, u32>(&context, &node_indices),
            output: alloc_allocation::<B, bf16>(&context, rows * NUM_HEADS as usize * HEAD_DIM),
            context,
            kernel,
            rows: rows as u32,
            prefix_length: prefix_length as u32,
            ancestor_stride: ancestor_stride as u32,
            node_capacity: nodes as u32,
        }
    }

    fn encode(
        &mut self,
        repetitions: u32,
    ) -> Duration {
        let mut encoder = Encoder::new(self.context.as_ref()).unwrap();
        for _ in 0..repetitions {
            self.kernel.encode(
                &self.prefix_kv,
                &mut self.node_kv,
                &self.current_qkv,
                &self.cosines,
                &self.sines,
                &self.node_metadata,
                &self.ancestor_indices,
                &self.ancestor_counts,
                &self.node_indices,
                &mut self.output,
                self.rows,
                self.prefix_length,
                self.ancestor_stride,
                self.node_capacity,
                MAX_DEPTH,
                1.0 / (HEAD_DIM as f32).sqrt(),
                &mut encoder,
            );
        }
        encoder.end_encoding().submit().wait_until_completed().unwrap().gpu_execution_time()
    }
}

#[uzu_test]
fn ancestor_attention_matches_cpu() {
    let mut cpu = Runner::<Cpu>::new(4, 5, 3, 16);
    cpu.encode(1);
    let expected_output = allocation_to_vec::<Cpu, bf16>(&cpu.output);
    let expected_node_kv = allocation_to_vec::<Cpu, bf16>(&cpu.node_kv);

    let mut metal = Runner::<Metal>::new(4, 5, 3, 16);
    metal.encode(1);
    let actual_output = allocation_to_vec::<Metal, bf16>(&metal.output);
    let actual_node_kv = allocation_to_vec::<Metal, bf16>(&metal.node_kv);

    assert_eq_float(&expected_output, &actual_output, 0.02, "AncestorAttention output");
    assert_eq!(actual_node_kv, expected_node_kv);
}

#[uzu_test]
#[ignore = "benchmark"]
fn benchmark_ancestor_attention() {
    const BATCH: u32 = 32;
    const SAMPLES: usize = 50;

    let mut runner = Runner::<Metal>::new(8, 16, 8, 65);
    let mut run = || runner.encode(BATCH).div_f64(BATCH as f64);
    let warmup = std::time::Instant::now();
    while warmup.elapsed() < std::time::Duration::from_millis(500) {
        run();
    }
    let mut samples = (0..SAMPLES).map(|_| run()).collect::<Vec<_>>();
    samples.sort_unstable();
    eprintln!("ancestor_attention gpu={:?}", samples[SAMPLES / 2]);
}
