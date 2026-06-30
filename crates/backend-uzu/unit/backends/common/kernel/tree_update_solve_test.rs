use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::GdnTreeUpdateSolveKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
};

const K: usize = 128;

fn run_tiny_block<B: Backend>() -> Vec<f32> {
    let context = B::Context::new().expect("context");
    let kernel = <<B as Backend>::Kernels as Kernels>::GdnTreeUpdateSolveKernel::new(
        &context,
        DataType::F32,
        K as u32,
        16,
        16,
        false,
    )
    .expect("kernel");

    let batch_size = 1u32;
    let tree_size = 2u32;
    let num_v_heads = 1u32;
    let num_k_heads = 1u32;
    let head_v_dim = 2u32;

    // k: [B, T, num_k_heads, K]. Only k[0] and k[1] are non-zero.
    let mut k = vec![0.0f32; tree_size as usize * K];
    k[0] = 1.0;
    k[K + 1] = 1.0;

    // v: [B, T, num_v_heads, head_v_dim].
    let v = vec![10.0f32, 20.0, 30.0, 40.0];
    let prefix = vec![0.0f32, 0.0];
    let beta = vec![1.0f32, 1.0];

    // A: [B * HV, T, T]. Row 1 depends on row 0 with weight 0.5.
    let a = vec![0.0f32, 0.0, 0.5, 0.0];
    // inverse(I + A) for [[1, 0], [0.5, 1]].
    let a_inv = vec![1.0f32, 0.0, -0.5, 1.0];

    // h0: [pool, num_v_heads, head_v_dim, K].
    let mut h0 = vec![0.0f32; head_v_dim as usize * K];
    h0[0] = 1.0;
    h0[1] = 2.0;
    h0[K] = -1.0;
    h0[K + 1] = 0.5;
    let h0_idx = vec![0i32];

    let k = context.create_array_from(&[k.len()], &k);
    let v = context.create_array_from(&[v.len()], &v);
    let prefix = context.create_array_from(&[prefix.len()], &prefix);
    let beta = context.create_array_from(&[beta.len()], &beta);
    let a = context.create_array_from(&[a.len()], &a);
    let a_inv = context.create_array_from(&[a_inv.len()], &a_inv);
    let h0 = context.create_array_from(&[h0.len()], &h0);
    let h0_idx = context.create_array_from(&[h0_idx.len()], &h0_idx);
    let mut u = context.create_array_uninitialized(&[4], DataType::F32).into_allocation();

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
        batch_size,
        tree_size,
        num_v_heads,
        num_k_heads,
        head_v_dim,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&u)
}

#[uzu_test]
fn test_gdn_tree_update_solve_tiny_block() {
    let expected = run_tiny_block::<Cpu>();
    assert_eq_float(&[9.0f32, 21.0, 23.5, 29.0], &expected, 1e-6, "cpu u");

    for_each_non_cpu_backend!(|B| {
        let output = run_tiny_block::<B>();
        assert_eq_float(&expected, &output, 1e-5, "backend u");
    });
}
