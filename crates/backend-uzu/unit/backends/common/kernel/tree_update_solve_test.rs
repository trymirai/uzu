#![cfg(metal_backend)]

use half::bf16;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::TreeUpdateSolveKernel},
        cpu::Cpu,
        metal::Metal,
    },
    data_type::DataType,
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

const BT: u32 = 16;
const BVS: &[u32] = &[16, 32];

#[derive(Clone, Copy)]
struct SolveCase {
    name: &'static str,
    batch_size: u32,
    tree_size: u32,
    num_v_heads: u32,
    head_v_dim: u32,
}

const CASES: &[SolveCase] = &[
    SolveCase {
        name: "single_full_block",
        batch_size: 1,
        tree_size: 16,
        num_v_heads: 1,
        head_v_dim: 16,
    },
    SolveCase {
        name: "ragged_second_block",
        batch_size: 1,
        tree_size: 17,
        num_v_heads: 2,
        head_v_dim: 20,
    },
    SolveCase {
        name: "common_small_tree",
        batch_size: 1,
        tree_size: 33,
        num_v_heads: 4,
        head_v_dim: 128,
    },
    SolveCase {
        name: "ragged_four_blocks",
        batch_size: 1,
        tree_size: 49,
        num_v_heads: 4,
        head_v_dim: 128,
    },
    SolveCase {
        // NB=7: exercises pair_idx >= 1, tail blocks >= 4, and a ragged final block.
        name: "ragged_seven_blocks",
        batch_size: 1,
        tree_size: 97,
        num_v_heads: 2,
        head_v_dim: 64,
    },
    SolveCase {
        // NB=16: the large-tree regime this kernel exists for; num_full_pairs
        // reaches 7, so the deep packed-pair accumulation is exercised.
        name: "large_tree",
        batch_size: 1,
        tree_size: 256,
        num_v_heads: 2,
        head_v_dim: 64,
    },
    SolveCase {
        name: "batched_optional_h0",
        batch_size: 2,
        tree_size: 16,
        num_v_heads: 2,
        head_v_dim: 16,
    },
];

fn run_case<B: Backend, T: ArrayElement + Copy>(
    case: SolveCase,
    bv: u32,
    use_mxu: bool,
    use_h0: bool,
    data_type: DataType,
    cast: impl Fn(f32) -> T + Copy,
) -> Vec<f32> {
    let context = B::Context::new().expect("context");
    let kernel =
        <<B as Backend>::Kernels as Kernels>::TreeUpdateSolveKernel::new(&context, data_type, bv, use_mxu, use_h0)
            .expect("kernel");

    let batch_size = case.batch_size as usize;
    let tree_size = case.tree_size as usize;
    let num_v_heads = case.num_v_heads as usize;
    let head_v_dim = case.head_v_dim as usize;

    let v_len = batch_size * tree_size * num_v_heads * head_v_dim;
    let scalar_len = batch_size * tree_size * num_v_heads;
    let num_blocks = tree_size.div_ceil(BT as usize);
    let num_col_pairs = num_blocks.div_ceil(2);
    let a_len = batch_size * num_v_heads * num_blocks * num_col_pairs * (BT * 2 * BT) as usize;
    let inv_len = batch_size * num_v_heads * num_blocks * (BT * BT) as usize;
    let u_len = batch_size * num_v_heads * tree_size * head_v_dim;

    let kh0 = (0..v_len).map(|i| ((i as f32 * 0.019).sin() * 0.2) + 0.01).collect::<Vec<f32>>();
    let v = (0..v_len).map(|i| cast(((i as f32 * 0.017).cos() * 0.18) - 0.02)).collect::<Vec<_>>();
    let prefix = (0..scalar_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % num_v_heads) as f32) * 0.003)
        .collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();
    // Packed [B * HV, NB, ceil(NB/2), BT, 2*BT] block-pair tiles; strictly-lower
    // blocks get values, everything else zero (the kernel never reads it).
    let a_f32 = (0..a_len)
        .map(|i| {
            let local_col = i % (2 * BT as usize);
            let local_row = (i / (2 * BT as usize)) % BT as usize;
            let pair = (i / (BT * 2 * BT) as usize) % num_col_pairs;
            let block = (i / ((BT * 2 * BT) as usize * num_col_pairs)) % num_blocks;
            let row = block * BT as usize + local_row;
            let col = pair * 2 * BT as usize + local_col;
            if col < tree_size && row < tree_size && col / (BT as usize) < block {
                // ~0.05 per entry: with the dense lower-triangle fill this puts the
                // per-row A magnitude near the real l2-normed-key / softplus-decay regime.
                ((i as f32 * 0.011).sin()) * 0.05
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    // Compact [B * HV, ceil(T/BT), BT, BT] diagonal blocks.
    let a_inv = (0..inv_len)
        .map(|i| {
            let col = i % BT as usize;
            let row = (i / BT as usize) % BT as usize;
            if row == col {
                1.0
            } else if col < row {
                -((i as f32 * 0.011).sin()) * 0.05
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let h0_idx = if case.name == "batched_optional_h0" {
        vec![0, -1]
    } else {
        (0..batch_size).map(|batch| batch as i32).collect::<Vec<_>>()
    };

    let kh0 = alloc_allocation_with_data::<B, f32>(&context, &kh0);
    let v = alloc_allocation_with_data::<B, T>(&context, &v);
    let prefix = alloc_allocation_with_data::<B, f32>(&context, &prefix);
    let beta = alloc_allocation_with_data::<B, f32>(&context, &beta);
    let a = alloc_allocation_with_data::<B, f32>(&context, &a_f32);
    let a_inv = alloc_allocation_with_data::<B, f32>(&context, &a_inv);
    let h0_idx = alloc_allocation_with_data::<B, i32>(&context, &h0_idx);
    let mut u = alloc_allocation::<B, f32>(&context, u_len);

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode(
        use_h0.then_some(&kh0),
        &v,
        &prefix,
        &beta,
        &a,
        &a_inv,
        use_h0.then_some(&h0_idx),
        &mut u,
        case.batch_size,
        case.tree_size,
        case.num_v_heads,
        case.head_v_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&u)
}

#[uzu_test]
fn test_tree_update_solve_cases() {
    for &bv in BVS {
        for &use_h0 in &[true, false] {
            for &case in CASES {
                let expected = run_case::<Cpu, f32>(case, bv, false, use_h0, DataType::F32, |x| x);
                for_each_non_cpu_backend!(|B| {
                    let output = run_case::<B, f32>(case, bv, false, use_h0, DataType::F32, |x| x);
                    assert_eq_float(&expected, &output, 1e-4, &format!("{} BV{bv} h0={use_h0}", case.name));
                });
            }
        }
    }

    {
        let supports_mxu = <Metal as Backend>::Context::new().expect("context").supports_mxu();
        let paths: &[(&str, bool)] = if supports_mxu {
            &[("Simdgroup", false), ("MXU", true)]
        } else {
            &[("Simdgroup", false)]
        };

        // MXU needs an even number of column fragments or MPP row pairing; BV16
        // (a single 16x16 fragment) is not supported.
        for &bv in &[32] {
            for &use_h0 in &[true, false] {
                for &case in CASES {
                    let expected = run_case::<Cpu, bf16>(case, bv, false, use_h0, DataType::BF16, bf16::from_f32);
                    for &(path, use_mxu) in paths {
                        let output = run_case::<Metal, bf16>(case, bv, use_mxu, use_h0, DataType::BF16, bf16::from_f32);
                        assert_eq_float(&expected, &output, 1e-2, &format!("{} {path} BV{bv} h0={use_h0}", case.name));
                    }
                }
            }
        }
    }
}
