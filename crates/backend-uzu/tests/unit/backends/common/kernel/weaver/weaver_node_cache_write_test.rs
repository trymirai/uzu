#![cfg(metal_backend)]

use half::bf16;
use proc_macros::uzu_test;

use crate::{
    backends::{
        common::{Allocation, Backend, Encoder, Kernels, kernel::WeaverNodeCacheWriteKernel},
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
    tests::helpers::{alloc_allocation_with_data, allocation_to_vec, create_context},
};

const MODEL_DIM: usize = 64;
const ROWS: usize = 4;
const NODES: usize = 9;

struct Runner<B: Backend> {
    context: std::sync::Arc<B::Context>,
    kernel: <B::Kernels as Kernels>::WeaverNodeCacheWriteKernel,
    current_qkv: Allocation<B>,
    node_kv: Allocation<B>,
    node_indices: Allocation<B>,
}

fn values(
    length: usize,
    offset: usize,
) -> Vec<bf16> {
    (0..length).map(|index| bf16::from_f32((((index + offset) * 17 % 251) as f32 - 125.0) / 128.0)).collect()
}

/// Slots are deliberately non-contiguous and out of order so the scatter is
/// exercised rather than a coincidentally-contiguous copy.
const NODE_INDICES: [u32; ROWS] = [7, 1, 8, 4];

impl<B: Backend> Runner<B> {
    fn new() -> Self {
        let qkv_width = 3 * MODEL_DIM;
        let kv_width = 2 * MODEL_DIM;
        let context = create_context::<B>();
        let kernel =
            <B::Kernels as Kernels>::WeaverNodeCacheWriteKernel::new(context.as_ref(), DataType::BF16).unwrap();
        Self {
            current_qkv: alloc_allocation_with_data::<B, bf16>(&context, &values(ROWS * qkv_width, 29)),
            node_kv: alloc_allocation_with_data::<B, bf16>(&context, &values(NODES * kv_width, 11)),
            node_indices: alloc_allocation_with_data::<B, u32>(&context, &NODE_INDICES),
            context,
            kernel,
        }
    }

    fn run(&mut self) -> Vec<bf16> {
        let mut encoder = Encoder::new(self.context.as_ref()).unwrap();
        self.kernel.encode(
            &self.current_qkv,
            &mut self.node_kv,
            &self.node_indices,
            MODEL_DIM as u32,
            NODES as u32,
            (ROWS * 2 * MODEL_DIM) as u32,
            &mut encoder,
        );
        encoder.end_encoding().submit().wait_until_completed().unwrap();
        allocation_to_vec::<B, bf16>(&self.node_kv)
    }
}

#[uzu_test]
fn weaver_node_cache_write_matches_cpu() {
    let qkv_width = 3 * MODEL_DIM;

    let mut cpu = Runner::<Cpu>::new();
    let initial_nodes = allocation_to_vec::<Cpu, bf16>(&cpu.node_kv);
    let current_qkv = allocation_to_vec::<Cpu, bf16>(&cpu.current_qkv);
    let expected = cpu.run();

    for component in 0..2 {
        for node in 0..NODES {
            let destination = component * NODES * MODEL_DIM + node * MODEL_DIM;
            match NODE_INDICES.iter().position(|&index| index as usize == node) {
                Some(row) => {
                    let source = row * qkv_width + (component + 1) * MODEL_DIM;
                    assert_eq!(expected[destination..destination + MODEL_DIM], current_qkv[source..source + MODEL_DIM],);
                },
                None => assert_eq!(
                    expected[destination..destination + MODEL_DIM],
                    initial_nodes[destination..destination + MODEL_DIM],
                ),
            }
        }
    }

    let mut metal = Runner::<Metal>::new();
    assert_eq!(metal.run(), expected);
}
