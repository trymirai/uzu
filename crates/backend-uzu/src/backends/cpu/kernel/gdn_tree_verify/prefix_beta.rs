use half::{bf16, f16};
use num_traits::Float;
use proc_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ActivationType};

#[kernel(BuildPrefixBeta)]
#[variants(T, f32, f16, bf16)]
pub fn build_prefix_beta<T: ArrayElement + Float>(
    path_matrix: *const u8,
    a: *const T,
    b: *const T,
    a_log: *const f32,
    dt_bias: *const f32,
    prefix: *mut f32,
    beta: *mut f32,
    batch_size: u32,
    tree_size: u32,
    value_heads: u32,
) {
    let batch_size = batch_size as usize;
    let tree_size = tree_size as usize;
    let value_heads = value_heads as usize;

    unsafe {
        for batch in 0..batch_size {
            let path_batch = batch * tree_size * tree_size;
            let batch_offset = batch * tree_size * value_heads;

            for row in 0..tree_size {
                for head in 0..value_heads {
                    let out_idx = batch_offset + row * value_heads + head;
                    let b_val = (*b.add(out_idx)).to_f32().unwrap();
                    *beta.add(out_idx) = 1.0 / (1.0 + (-b_val).exp());

                    let mut sum = 0.0f32;
                    for col in 0..tree_size {
                        if *path_matrix.add(path_batch + row * tree_size + col) == 0 {
                            continue;
                        }
                        let a_val = (*a.add(batch_offset + head * tree_size + col)).to_f32().unwrap();
                        let sp = ActivationType::SOFTPLUS.activate(a_val + *dt_bias.add(head));
                        sum -= a_log.add(head).read().exp() * sp;
                    }
                    *prefix.add(out_idx) = sum;
                }
            }
        }
    }
}
