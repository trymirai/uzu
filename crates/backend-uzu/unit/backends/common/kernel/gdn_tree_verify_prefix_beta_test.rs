use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayElement,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildPrefixBetaKernel},
        cpu::Cpu,
    },
    tests::{
        assert::assert_eq_float,
        helpers::{alloc_allocation, alloc_allocation_with_data, allocation_to_vec},
    },
};

struct Input {
    path_matrix: Vec<u8>,
    a: Vec<f32>,
    b: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
}

fn path_matrix_from_parent(
    parent: &[i32],
    batch_size: usize,
    tree_size: usize,
) -> Vec<u8> {
    let mut path_matrix = vec![0; batch_size * tree_size * tree_size];

    for batch in 0..batch_size {
        for row in 0..tree_size {
            let row_base = batch * tree_size * tree_size + row * tree_size;
            let mut cur = row as i32;
            for _ in 0..tree_size {
                if cur < 0 {
                    break;
                }
                path_matrix[row_base + cur as usize] = 1;
                cur = parent[batch * tree_size + cur as usize];
            }
        }
    }

    path_matrix
}

fn chain_parent(tree_size: usize) -> Vec<i32> {
    let mut parent = vec![-1; tree_size];
    for (i, value) in parent.iter_mut().enumerate().skip(1) {
        *value = (i - 1) as i32;
    }
    parent
}

fn make_input(
    parent: Vec<i32>,
    batch_size: usize,
    tree_size: usize,
    value_heads: usize,
) -> Input {
    let len = batch_size * tree_size * value_heads;
    Input {
        path_matrix: path_matrix_from_parent(&parent, batch_size, tree_size),
        a: (0..len).map(|i| (i as f32 * 0.17).sin() - 0.4).collect(),
        b: (0..len).map(|i| (i as f32 * 0.11).cos() * 0.8).collect(),
        a_log: (0..value_heads).map(|i| -0.2 + i as f32 * 0.03).collect(),
        dt_bias: (0..value_heads).map(|i| 0.1 - i as f32 * 0.02).collect(),
        batch_size: batch_size as u32,
        tree_size: tree_size as u32,
        value_heads: value_heads as u32,
    }
}

fn run<B: Backend>(input: &Input) -> (Vec<f32>, Vec<f32>) {
    let context = B::Context::new().expect("create context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildPrefixBetaKernel::new(&context, f32::data_type())
        .expect("create BuildPrefixBetaKernel");

    let output_len = input.a.len();
    let path_matrix = alloc_allocation_with_data::<B, u8>(&context, &input.path_matrix);
    let a = alloc_allocation_with_data::<B, f32>(&context, &input.a);
    let b = alloc_allocation_with_data::<B, f32>(&context, &input.b);
    let a_log = alloc_allocation_with_data::<B, f32>(&context, &input.a_log);
    let dt_bias = alloc_allocation_with_data::<B, f32>(&context, &input.dt_bias);
    let mut prefix = alloc_allocation::<B, f32>(&context, output_len);
    let mut beta = alloc_allocation::<B, f32>(&context, output_len);

    let mut encoder = Encoder::new(context.as_ref()).expect("create encoder");
    kernel.encode(
        &path_matrix,
        &a,
        &b,
        &a_log,
        &dt_bias,
        &mut prefix,
        &mut beta,
        input.batch_size,
        input.tree_size,
        input.value_heads,
        &mut encoder,
    );
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    (allocation_to_vec::<B, f32>(&prefix), allocation_to_vec::<B, f32>(&beta))
}

#[uzu_test]
fn test_build_prefix_beta_matches_cpu() {
    let cases = [
        make_input(
            vec![
                -1, 0, 0, 1, 1, //
                -1, 0, 1, 2, 2,
            ],
            2,
            5,
            6,
        ),
        make_input(chain_parent(33), 1, 33, 5),
    ];

    for input in &cases {
        let (expected_prefix, expected_beta) = run::<Cpu>(input);
        for_each_non_cpu_backend!(|B| {
            let (prefix, beta) = run::<B>(input);
            let backend = std::any::type_name::<B>();
            assert_eq_float::<f32>(&expected_prefix, &prefix, 1e-4, &format!("prefix backend={backend}"));
            assert_eq_float::<f32>(&expected_beta, &beta, 1e-5, &format!("beta backend={backend}"));
        });
    }
}
