use proc_macros::uzu_test;
use test_runner::for_each_non_cpu_backend;

use crate::{
    array::ArrayContextExt,
    backends::{
        common::{Backend, Context, Encoder, Kernels, kernel::BuildPathMatrixKernel},
        cpu::Cpu,
    },
    data_type::DataType,
    tests::helpers::allocation_to_vec,
};

fn get_output<B: Backend>(
    parent: &[i32],
    batch_size: u32,
    tree_size: u32,
) -> Vec<u8> {
    let context = B::Context::new().expect("Failed to create Context");
    let kernel = <<B as Backend>::Kernels as Kernels>::BuildPathMatrixKernel::new(&context)
        .expect("Failed to create BuildPathMatrixKernel");

    let parent = context.create_array_from(&[parent.len()], parent);
    let mut path_matrix = context
        .create_array_uninitialized(&[(batch_size * tree_size * tree_size) as usize], DataType::U8)
        .into_allocation();

    let mut encoder = Encoder::new(context.as_ref()).expect("Failed to create encoder");
    kernel.encode(parent.allocation(), &mut path_matrix, batch_size, tree_size, &mut encoder);
    encoder.end_encoding().submit().wait_until_completed().unwrap();

    allocation_to_vec(&path_matrix)
}

fn chain_parent(tree_size: usize) -> Vec<i32> {
    let mut parent = vec![-1; tree_size];
    for i in 1..tree_size {
        parent[i] = (i - 1) as i32;
    }
    parent
}

#[uzu_test]
fn test_build_path_matrix_matches_cpu() {
    for tree_size in [5, 129, 257] {
        let parent = chain_parent(tree_size);
        let expected = get_output::<Cpu>(&parent, 1, tree_size as u32);

        for_each_non_cpu_backend!(|B| {
            let output = get_output::<B>(&parent, 1, tree_size as u32);
            assert_eq!(output, expected, "backend {} tree_size {tree_size}", std::any::type_name::<B>());
        });
    }
}
