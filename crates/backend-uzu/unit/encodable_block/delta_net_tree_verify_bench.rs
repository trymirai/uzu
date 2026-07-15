#![cfg(metal_backend)]

use std::time::{Duration, Instant};

use criterion::Criterion;
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{Backend, Context, Kernels, kernel::delta_net_tree_verify::DeltaNetTreeVerify},
        metal::Metal,
    },
    data_type::DataType,
    encodable_block::mixer::delta_net::tree_verify::{TreeVerifyEncodeArguments, TreeVerifyNewArguments},
    tests::{helpers::alloc_allocation_with_data, matmul::iter_encode_loop_named},
};

#[uzu_bench]
fn bench_delta_net_tree_verify(c: &mut Criterion) {
    const K_HEADS: usize = 16;
    const V_HEADS: usize = 48;
    const HEAD_DIM: usize = 128;
    let arguments = TreeVerifyNewArguments {
        data_type: DataType::BF16,
        num_k_heads: K_HEADS,
        num_v_heads: V_HEADS,
        head_k_dim: HEAD_DIM,
        head_v_dim: HEAD_DIM,
    };
    let context = <Metal as Backend>::Context::new().unwrap();
    let started = Instant::now();
    let tree_verify =
        <<Metal as Backend>::Kernels as Kernels>::DeltaNetTreeVerify::new(context.as_ref(), &arguments).unwrap();
    eprintln!("GDN tree verification PSO construction: {:.2} ms", started.elapsed().as_secs_f64() * 1e3);
    let h0 = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.001; V_HEADS * HEAD_DIM * HEAD_DIM]);
    let mut group = c.benchmark_group("Metal/Kernel/GDNTreeVerify");
    group.sample_size(20).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));

    for tree_size in [32usize, 49, 64, 128] {
        let q = alloc_allocation_with_data::<Metal, bf16>(
            &context,
            &vec![bf16::from_f32(0.01); tree_size * K_HEADS * HEAD_DIM],
        );
        let k = alloc_allocation_with_data::<Metal, bf16>(
            &context,
            &vec![bf16::from_f32(0.02); tree_size * K_HEADS * HEAD_DIM],
        );
        let v = alloc_allocation_with_data::<Metal, bf16>(
            &context,
            &vec![bf16::from_f32(0.03); tree_size * V_HEADS * HEAD_DIM],
        );
        let trie = alloc_allocation_with_data::<Metal, u32>(
            &context,
            &(0..tree_size as u32).flat_map(|node| [node, tree_size as u32 - 1, node]).collect::<Vec<_>>(),
        );
        let log_decay = alloc_allocation_with_data::<Metal, f32>(&context, &vec![-0.01; tree_size * V_HEADS]);
        let beta = alloc_allocation_with_data::<Metal, f32>(&context, &vec![0.2; tree_size * V_HEADS]);
        let benchmark_path = format!("Metal/Kernel/GDNTreeVerify/T{tree_size}");
        group.bench_function(format!("T{tree_size}"), |bencher| {
            iter_encode_loop_named(context.as_ref(), bencher, &benchmark_path, |encoder| {
                tree_verify
                    .encode(
                        TreeVerifyEncodeArguments {
                            q: &q,
                            k: &k,
                            v: &v,
                            trie: &trie,
                            log_decay: &log_decay,
                            beta: &beta,
                            h0: &h0,
                            tree_size,
                        },
                        encoder,
                    )
                    .unwrap();
            })
        });
    }
}
