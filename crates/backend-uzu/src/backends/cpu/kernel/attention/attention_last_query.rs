use half::bf16;
use proc_macros::kernel;

use super::attention_single_pass::attention_single_pass;

#[kernel(AttentionLastQuery)]
#[variants(HEAD_DIM, 128)]
fn attention_last_query<const HEAD_DIM: u32>(
    prefix_kv: *const bf16,
    node_kv: *const bf16,
    current_qkv: *const bf16,
    ancestor_indices: *const u32,
    ancestor_counts: *const u32,
    output: *mut bf16,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    node_capacity: u32,
    scale: f32,
    #[specialize] num_heads: u32,
) {
    const QKV_COMPONENTS: usize = 3;
    const KEY_COMPONENT: usize = 1;
    const VALUE_COMPONENT: usize = 2;
    let head_dim = HEAD_DIM as usize;
    let num_heads = num_heads as usize;
    let model_dim = num_heads * head_dim;
    let qkv_width = QKV_COMPONENTS * model_dim;
    let prefix_length = prefix_length as usize;
    let node_capacity = node_capacity as usize;
    let last_node = node_capacity.saturating_sub(1);

    for row in 0..rows as usize {
        unsafe {
            let current_row = current_qkv.add(row * qkv_width);

            let ancestor_count = *ancestor_counts.add(row) as usize;
            let length = prefix_length + ancestor_count + 1;
            let mut keys = vec![bf16::ZERO; length * model_dim];
            let mut values = vec![bf16::ZERO; length * model_dim];
            std::ptr::copy_nonoverlapping(prefix_kv, keys.as_mut_ptr(), prefix_length * model_dim);
            std::ptr::copy_nonoverlapping(
                prefix_kv.add(prefix_length * model_dim),
                values.as_mut_ptr(),
                prefix_length * model_dim,
            );
            for offset in 0..ancestor_count {
                let ancestor = (*ancestor_indices.add(row * ancestor_stride as usize + offset) as usize).min(last_node);
                std::ptr::copy_nonoverlapping(
                    node_kv.add(ancestor * model_dim),
                    keys.as_mut_ptr().add((prefix_length + offset) * model_dim),
                    model_dim,
                );
                std::ptr::copy_nonoverlapping(
                    node_kv.add(node_capacity * model_dim + ancestor * model_dim),
                    values.as_mut_ptr().add((prefix_length + offset) * model_dim),
                    model_dim,
                );
            }
            std::ptr::copy_nonoverlapping(
                current_row.add(KEY_COMPONENT * model_dim),
                keys.as_mut_ptr().add((length - 1) * model_dim),
                model_dim,
            );
            std::ptr::copy_nonoverlapping(
                current_row.add(VALUE_COMPONENT * model_dim),
                values.as_mut_ptr().add((length - 1) * model_dim),
                model_dim,
            );
            attention_single_pass::<bf16, HEAD_DIM>(
                current_row,
                keys.as_ptr(),
                values.as_ptr(),
                output.add(row * num_heads * head_dim),
                1,
                length as u32,
                HEAD_DIM,
                model_dim as u32,
                HEAD_DIM,
                model_dim as u32,
                None,
                scale,
                None,
                None,
                None,
                num_heads as u32,
                1,
                false,
                false,
                false,
                false,
                false,
            );
        }
    }
}
