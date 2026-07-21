#![cfg(metal_backend)]

use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildTreePrefixKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

fn run<B: Backend>(tree_size: usize) -> Vec<f32> {
    const HEADS: usize = 5;
    let context = B::Context::new().unwrap();
    let trie = (0..tree_size as u32).flat_map(|node| [node, tree_size as u32 - 1, node]).collect::<Vec<_>>();
    let log_decay = (0..tree_size * HEADS).map(|i| -0.001 - i as f32 * 0.0001).collect::<Vec<_>>();
    let trie = alloc_allocation_with_data::<B, u32>(&context, &trie);
    let log_decay = alloc_allocation_with_data::<B, f32>(&context, &log_decay);
    let mut prefix = alloc_allocation::<B, f32>(&context, tree_size * HEADS);
    let kernel = <B::Kernels as Kernels>::BuildTreePrefixKernel::new(&context).unwrap();
    let mut encoder = Encoder::new(context.as_ref()).unwrap();
    kernel.encode(&trie, &log_decay, &mut prefix, 1, tree_size as u32, HEADS as u32, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();
    allocation_to_vec(&prefix)
}

#[uzu_test]
fn test_build_tree_prefix() {
    const HEADS: usize = 5;
    for tree_size in [49, 64, 128] {
        let expected = (0..tree_size)
            .flat_map(|row| {
                (0..HEADS).map(move |head| {
                    (0..=row).map(|token| -0.001 - (token * HEADS + head) as f32 * 0.0001).sum::<f32>()
                })
            })
            .collect::<Vec<_>>();
        assert_eq_float(&expected, &run::<Cpu>(tree_size), 1e-6, "CPU");
        for_each_non_cpu_backend!(|B| assert_eq_float(&expected, &run::<B>(tree_size), 1e-6, "backend"));
    }
}
