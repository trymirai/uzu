#![cfg(metal_backend)]

use std::{mem::size_of, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use half::bf16;
use proc_macros::uzu_bench;

use crate::{
    backends::{
        common::{Allocation, Backend, Context, Encoder, Kernels, kernel::TreeUpdateSolveKernel},
        metal::{DeviceExt, Metal},
    },
    data_type::DataType,
    tests::{
        cold_pool::ColdPool,
        helpers::{alloc_allocation, alloc_allocation_with_data},
        matmul::iter_encode_loop_named,
    },
};

const NUM_V_HEADS: usize = 48;
const HEAD_V_DIM: usize = 128;
const BT: usize = 16;
const BVS: &[usize] = &[16, 32];
const MXU_BVS: &[usize] = &[32];

const BENCH_SHAPES: &[(usize, usize)] = &[
    (1, 33),
    (1, 49),
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 512),
    (2, 33),
    (2, 49),
    (2, 64),
    (2, 128),
    (2, 256),
    (2, 512),
    (4, 33),
    (4, 49),
    (4, 64),
    (4, 128),
    (4, 256),
    (4, 512),
    (8, 33),
    (8, 49),
    (8, 64),
    (8, 128),
    (8, 256),
    (8, 512),
];

struct TreeUpdateSolveBuffers {
    kh0: Allocation<Metal>,
    v: Allocation<Metal>,
    prefix: Allocation<Metal>,
    beta: Allocation<Metal>,
    a: Allocation<Metal>,
    a_inv: Allocation<Metal>,
    h0_idx: Allocation<Metal>,
    u: Allocation<Metal>,
}

fn matrix_value(
    batch_head_matrix_idx: usize,
    row: usize,
    col: usize,
    tree_size: usize,
) -> f32 {
    if col < row {
        ((batch_head_matrix_idx * tree_size * tree_size + row * tree_size + col) as f32 * 0.011).sin() * 0.01
    } else {
        0.0
    }
}

// Compact [B * HV, ceil(T/BT), BT, BT] diagonal blocks.
fn inverse_value(
    batch_head_matrix_idx: usize,
    block: usize,
    row: usize,
    col: usize,
    tree_size: usize,
) -> f32 {
    if row == col {
        1.0
    } else if col < row {
        -matrix_value(batch_head_matrix_idx, block * BT + row, block * BT + col, tree_size)
    } else {
        0.0
    }
}

fn make_buffers(
    context: &<Metal as Backend>::Context,
    batch_size: usize,
    tree_size: usize,
) -> TreeUpdateSolveBuffers {
    let v_len = batch_size * tree_size * NUM_V_HEADS * HEAD_V_DIM;
    let scalar_len = batch_size * tree_size * NUM_V_HEADS;
    let num_blocks = tree_size.div_ceil(BT);
    let num_col_pairs = num_blocks.div_ceil(2);
    let a_len = batch_size * NUM_V_HEADS * num_blocks * num_col_pairs * BT * 2 * BT;
    let u_len = batch_size * NUM_V_HEADS * tree_size * HEAD_V_DIM;

    let kh0 = (0..v_len).map(|i| ((i as f32 * 0.019).sin() * 0.2) + 0.01).collect::<Vec<f32>>();
    let v = (0..v_len).map(|i| bf16::from_f32(((i as f32 * 0.017).cos() * 0.18) - 0.02)).collect::<Vec<_>>();
    let prefix = (0..scalar_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % NUM_V_HEADS) as f32) * 0.003)
        .collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();
    // Packed [B * HV, NB, ceil(NB/2), BT, 2*BT] block-pair tiles.
    let a_f32 = (0..a_len)
        .map(|i| {
            let local_col = i % (2 * BT);
            let local_row = (i / (2 * BT)) % BT;
            let pair = (i / (BT * 2 * BT)) % num_col_pairs;
            let block = (i / (BT * 2 * BT * num_col_pairs)) % num_blocks;
            let batch_head_matrix_idx = i / (BT * 2 * BT * num_col_pairs * num_blocks);
            let row = block * BT + local_row;
            let col = pair * 2 * BT + local_col;
            if col < tree_size && row < tree_size && col / BT < block {
                matrix_value(batch_head_matrix_idx, row, col, tree_size)
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let a = alloc_allocation_with_data::<Metal, f32>(context, &a_f32);
    let inv_len = batch_size * NUM_V_HEADS * tree_size.div_ceil(BT) * BT * BT;
    let a_inv = (0..inv_len)
        .map(|i| {
            let col = i % BT;
            let row = (i / BT) % BT;
            let block = (i / (BT * BT)) % tree_size.div_ceil(BT);
            let batch_head_matrix_idx = i / (BT * BT * tree_size.div_ceil(BT));
            inverse_value(batch_head_matrix_idx, block, row, col, tree_size)
        })
        .collect::<Vec<_>>();
    let h0_idx = (0..batch_size).map(|i| i as i32).collect::<Vec<_>>();

    TreeUpdateSolveBuffers {
        kh0: alloc_allocation_with_data::<Metal, f32>(context, &kh0),
        v: alloc_allocation_with_data::<Metal, bf16>(context, &v),
        prefix: alloc_allocation_with_data::<Metal, f32>(context, &prefix),
        beta: alloc_allocation_with_data::<Metal, f32>(context, &beta),
        a,
        a_inv: alloc_allocation_with_data::<Metal, f32>(context, &a_inv),
        h0_idx: alloc_allocation_with_data::<Metal, i32>(context, &h0_idx),
        u: alloc_allocation::<Metal, f32>(context, u_len),
    }
}

fn buffers_bytes(
    batch_size: usize,
    tree_size: usize,
) -> usize {
    let v_len = batch_size * tree_size * NUM_V_HEADS * HEAD_V_DIM;
    let scalar_len = batch_size * tree_size * NUM_V_HEADS;
    let num_blocks = tree_size.div_ceil(BT);
    let a_len = batch_size * NUM_V_HEADS * num_blocks * num_blocks.div_ceil(2) * BT * 2 * BT;
    let u_len = batch_size * NUM_V_HEADS * tree_size * HEAD_V_DIM;

    let inv_len = batch_size * NUM_V_HEADS * num_blocks * BT * BT;
    v_len * size_of::<bf16>()
        + (scalar_len * 2 + a_len + inv_len + u_len + v_len) * size_of::<f32>()
        + batch_size * size_of::<i32>()
}

#[uzu_bench]
fn bench_tree_update_solve(c: &mut Criterion) {
    let context = <Metal as Backend>::Context::new().expect("metal context");
    let kernel_paths = if context.device.supports_mxu() {
        &[("Simdgroup", false, BVS), ("MXU", true, MXU_BVS)][..]
    } else {
        &[("Simdgroup", false, BVS)][..]
    };

    for &(kernel_path, use_mxu, bvs) in kernel_paths {
        for &bv in bvs {
            let kernel = <<Metal as Backend>::Kernels as Kernels>::TreeUpdateSolveKernel::new(
                &context,
                DataType::BF16,
                bv as u32,
                use_mxu,
                true,
            )
            .expect("TreeUpdateSolveKernel");

            let mut group =
                c.benchmark_group(format!("Metal/Kernel/GDNTreeVerify/TreeUpdateSolve/{kernel_path}/BV{bv}"));
            // 500ms warmup ramps GPU clocks before measuring (µs-scale benches never
            // ramp otherwise and medians become run-composition artifacts).
            group.sample_size(20).warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(2));

            for &(batch_size, tree_size) in BENCH_SHAPES {
                let benchmark_path = format!(
                    "Metal/Kernel/GDNTreeVerify/TreeUpdateSolve/{kernel_path}/BV{bv}/B{batch_size}_T{tree_size}"
                );
                let benchmark_id = BenchmarkId::from_parameter(format!(
                    "B{batch_size}_T{tree_size}_HV{NUM_V_HEADS}_V{HEAD_V_DIM}_BV{bv}"
                ));
                let mut buffers = ColdPool::new(buffers_bytes(batch_size, tree_size), || {
                    make_buffers(&context, batch_size, tree_size)
                });

                group.throughput(Throughput::Elements((batch_size * tree_size * NUM_V_HEADS * HEAD_V_DIM) as u64));
                group.bench_function(benchmark_id, |bencher| {
                    iter_encode_loop_named::<Metal, _>(context.as_ref(), bencher, &benchmark_path, |encoder| {
                        let buffers = buffers.next_mut();
                        encode(&kernel, buffers, batch_size, tree_size, encoder);
                    });
                });
            }
            group.finish();
            test_runner::metrics::wait_gpu_cooldown();
        }
    }
}

fn encode(
    kernel: &impl TreeUpdateSolveKernel<Backend = Metal>,
    buffers: &mut TreeUpdateSolveBuffers,
    batch_size: usize,
    tree_size: usize,
    encoder: &mut Encoder<Metal>,
) {
    kernel.encode(
        Some(&buffers.kh0),
        &buffers.v,
        &buffers.prefix,
        &buffers.beta,
        &buffers.a,
        &buffers.a_inv,
        Some(&buffers.h0_idx),
        &mut buffers.u,
        batch_size as u32,
        tree_size as u32,
        NUM_V_HEADS as u32,
        HEAD_V_DIM as u32,
        encoder,
    );
}
