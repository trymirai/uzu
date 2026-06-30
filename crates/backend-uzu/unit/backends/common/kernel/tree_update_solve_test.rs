use half::bf16;
use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::{ArrayContextExt, ArrayElement},
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::GdnTreeUpdateSolveKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
};

const HEAD_K_DIM: usize = 128;
const BT: u32 = 16;
const BVS: &[u32] = &[16, 32, 64];

#[derive(Clone, Copy)]
struct SolveCase {
    name: &'static str,
    batch_size: u32,
    tree_size: u32,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
}

const CASES: &[SolveCase] = &[
    SolveCase {
        name: "single_full_block",
        batch_size: 1,
        tree_size: 16,
        num_v_heads: 1,
        num_k_heads: 1,
        head_v_dim: 16,
    },
    SolveCase {
        name: "ragged_second_block",
        batch_size: 1,
        tree_size: 17,
        num_v_heads: 2,
        num_k_heads: 1,
        head_v_dim: 20,
    },
    SolveCase {
        name: "common_small_tree",
        batch_size: 1,
        tree_size: 33,
        num_v_heads: 4,
        num_k_heads: 2,
        head_v_dim: 128,
    },
    SolveCase {
        name: "ragged_four_blocks",
        batch_size: 1,
        tree_size: 49,
        num_v_heads: 4,
        num_k_heads: 2,
        head_v_dim: 128,
    },
    SolveCase {
        name: "batched_optional_h0",
        batch_size: 2,
        tree_size: 16,
        num_v_heads: 2,
        num_k_heads: 1,
        head_v_dim: 16,
    },
];

fn run_case<B: Backend, T: ArrayElement + Copy>(
    case: SolveCase,
    bv: u32,
    use_mxu: bool,
    data_type: DataType,
    cast: impl Fn(f32) -> T + Copy,
) -> Vec<f32> {
    let context = B::Context::new().expect("context");
    let kernel = <<B as Backend>::Kernels as Kernels>::GdnTreeUpdateSolveKernel::new(
        &context,
        data_type,
        HEAD_K_DIM as u32,
        BT,
        bv,
        use_mxu,
    )
    .expect("kernel");

    let batch_size = case.batch_size as usize;
    let tree_size = case.tree_size as usize;
    let num_v_heads = case.num_v_heads as usize;
    let num_k_heads = case.num_k_heads as usize;
    let head_v_dim = case.head_v_dim as usize;

    let k_len = batch_size * tree_size * num_k_heads * HEAD_K_DIM;
    let v_len = batch_size * tree_size * num_v_heads * head_v_dim;
    let scalar_len = batch_size * tree_size * num_v_heads;
    let matrix_len = batch_size * num_v_heads * tree_size * tree_size;
    let h0_len = batch_size * num_v_heads * head_v_dim * HEAD_K_DIM;
    let u_len = batch_size * num_v_heads * tree_size * head_v_dim;

    let k = (0..k_len).map(|i| cast(((i as f32 * 0.019).sin() * 0.2) + 0.01)).collect::<Vec<_>>();
    let v = (0..v_len).map(|i| cast(((i as f32 * 0.017).cos() * 0.18) - 0.02)).collect::<Vec<_>>();
    let prefix = (0..scalar_len)
        .map(|i| -((i % tree_size) as f32) * 0.01 - ((i % num_v_heads) as f32) * 0.003)
        .collect::<Vec<_>>();
    let beta = (0..scalar_len).map(|i| 0.25 + ((i as f32 * 0.013).sin() + 1.0) * 0.2).collect::<Vec<_>>();
    let a = (0..matrix_len)
        .map(|i| {
            let col = i % tree_size;
            let row = (i / tree_size) % tree_size;
            if col < row {
                ((i as f32 * 0.011).sin()) * 0.01
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let a_inv = (0..matrix_len)
        .map(|i| {
            let col = i % tree_size;
            let row = (i / tree_size) % tree_size;
            if row == col {
                1.0
            } else if row / BT as usize == col / BT as usize && col < row {
                -((i as f32 * 0.011).sin()) * 0.01
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let h0 = (0..h0_len).map(|i| cast(((i as f32 * 0.007).sin() * 0.05) - 0.01)).collect::<Vec<_>>();
    let h0_idx = if case.name == "batched_optional_h0" {
        vec![0, -1]
    } else {
        (0..batch_size).map(|batch| batch as i32).collect::<Vec<_>>()
    };

    let k = context.create_array_from(&[k.len()], &k);
    let v = context.create_array_from(&[v.len()], &v);
    let prefix = context.create_array_from(&[prefix.len()], &prefix);
    let beta = context.create_array_from(&[beta.len()], &beta);
    let a = context.create_array_from(&[a.len()], &a);
    let a_inv = context.create_array_from(&[a_inv.len()], &a_inv);
    let h0 = context.create_array_from(&[h0.len()], &h0);
    let h0_idx = context.create_array_from(&[h0_idx.len()], &h0_idx);
    let mut u = context.create_array_uninitialized(&[u_len], DataType::F32).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("encoder");
    kernel.encode(
        k.allocation(),
        v.allocation(),
        prefix.allocation(),
        beta.allocation(),
        a.allocation(),
        a_inv.allocation(),
        h0.allocation(),
        h0_idx.allocation(),
        &mut u,
        case.batch_size,
        case.tree_size,
        case.num_v_heads,
        case.num_k_heads,
        case.head_v_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&u)
}

#[uzu_test]
fn test_gdn_tree_update_solve_cases() {
    for &bv in BVS {
        for &case in CASES {
            let expected = run_case::<Cpu, f32>(case, bv, false, DataType::F32, |x| x);
            for_each_non_cpu_backend!(|B| {
                let output = run_case::<B, f32>(case, bv, false, DataType::F32, |x| x);
                assert_eq_float(&expected, &output, 1e-4, &format!("{} BV{bv}", case.name));
            });
        }
    }
}

#[uzu_test]
fn test_gdn_tree_update_solve_mxu_cases() {
    for &bv in &[32] {
        for &case in CASES {
            let expected = run_case::<Cpu, bf16>(case, bv, false, DataType::BF16, bf16::from_f32);
            for_each_non_cpu_backend!(|B| {
                if <B as Backend>::Context::new().expect("context").supports_mxu() {
                    let output = run_case::<B, bf16>(case, bv, true, DataType::BF16, bf16::from_f32);
                    assert_eq_float(&expected, &output, 1e-2, &format!("{} MXU BV{bv}", case.name));
                }
            });
        }
    }
}
