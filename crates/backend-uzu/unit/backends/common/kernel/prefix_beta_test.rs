use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildPrefixBetaKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::{assert::assert_eq_float, helpers::allocation_to_vec},
};

fn get_output<B: Backend>(
    path_matrix: &[u8],
    a_transposed: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildPrefixBetaKernel::new(&context, DataType::F32)
        .expect("Failed to create BuildPrefixBetaKernel");

    let output_len = a_transposed.len();
    let path_matrix = context.create_array_from(&[path_matrix.len()], path_matrix);
    let a_transposed = context.create_array_from(&[a_transposed.len()], a_transposed);
    let b = context.create_array_from(&[b.len()], b);
    let a_log = context.create_array_from(&[a_log.len()], a_log);
    let dt_bias = context.create_array_from(&[dt_bias.len()], dt_bias);
    let mut prefix = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();
    let mut beta = context.create_array_uninitialized(&[output_len], DataType::F32).into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(
        path_matrix.allocation(),
        a_transposed.allocation(),
        b.allocation(),
        a_log.allocation(),
        dt_bias.allocation(),
        &mut prefix,
        &mut beta,
        batch_size,
        tree_size,
        value_heads,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec(&prefix), allocation_to_vec(&beta))
}

fn path_matrix_from_parent(
    parent: &[i32],
    batch_size: usize,
    tree_size: usize,
) -> Vec<u8> {
    let mut path_matrix = vec![0; batch_size * tree_size * tree_size];
    for batch in 0..batch_size {
        for row in 0..tree_size {
            let base = batch * tree_size * tree_size + row * tree_size;
            let mut cur = row as i32;
            while cur >= 0 {
                path_matrix[base + cur as usize] = 1;
                cur = parent[batch * tree_size + cur as usize];
            }
        }
    }
    path_matrix
}

#[uzu_test]
fn test_build_prefix_beta_matches_cpu() {
    let batch_size = 2;
    let value_heads = 5;

    for tree_size in [5, 33, 129] {
        let len = batch_size * tree_size * value_heads;
        let mut parent = vec![-1; batch_size * tree_size];
        for batch in 0..batch_size {
            for node in 1..tree_size {
                parent[batch * tree_size + node] = if batch == 0 {
                    (node - 1) as i32
                } else {
                    ((node - 1) / 2) as i32
                };
            }
        }

        let path_matrix = path_matrix_from_parent(&parent, batch_size, tree_size);
        let a_transposed = (0..len).map(|i| (i as f32 * 0.13).sin() - 0.4).collect::<Vec<_>>();
        let b = (0..len).map(|i| (i as f32 * 0.11).cos() * 0.8).collect::<Vec<_>>();
        let a_log = (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect::<Vec<_>>();
        let dt_bias = (0..value_heads).map(|i| 0.1 - i as f32 * 0.02).collect::<Vec<_>>();
        let expected = get_output::<Cpu>(
            &path_matrix,
            &a_transposed,
            &b,
            &a_log,
            &dt_bias,
            batch_size as u32,
            tree_size as u32,
            value_heads as u32,
        );

        for_each_non_cpu_backend!(|B| {
            let output = get_output::<B>(
                &path_matrix,
                &a_transposed,
                &b,
                &a_log,
                &dt_bias,
                batch_size as u32,
                tree_size as u32,
                value_heads as u32,
            );
            let msg = format!("backend {} tree_size {tree_size}", std::any::type_name::<B>());
            assert_eq_float::<f32>(&expected.0, &output.0, 1e-4, &format!("prefix {msg}"));
            assert_eq_float::<f32>(&expected.1, &output.1, 1e-5, &format!("beta {msg}"));
        });
    }
}
