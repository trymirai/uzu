use half::bf16;
use proc_macros::kernel;

use super::attention_single_pass::attention_single_pass;

#[kernel(AttentionLastQuery)]
#[variants(HEAD_DIM, 128)]
fn attention_last_query<const HEAD_DIM: u32>(
    prefix_qkv: *const bf16,
    node_qkv: *const bf16,
    current_qkv: *const bf16,
    ancestor_indices: *const u32,
    ancestor_counts: *const u32,
    output: *mut bf16,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    scale: f32,
    #[specialize] num_heads: u32,
) {
    const QKV_COMPONENTS: usize = 3;
    const KV_COMPONENTS: usize = 2;
    const KEY_COMPONENT: usize = 1;
    const VALUE_COMPONENT: usize = 2;
    let head_dim = HEAD_DIM as usize;
    let num_heads = num_heads as usize;
    let qkv_width = QKV_COMPONENTS * num_heads * head_dim;
    let kv_offset = KEY_COMPONENT * num_heads * head_dim;
    let kv_width = KV_COMPONENTS * num_heads * head_dim;

    for row in 0..rows as usize {
        unsafe {
            let current_row = current_qkv.add(row * qkv_width);

            let ancestor_count = *ancestor_counts.add(row) as usize;
            let length = prefix_length as usize + ancestor_count + 1;
            let mut sequence = vec![bf16::ZERO; length * qkv_width];
            std::ptr::copy_nonoverlapping(prefix_qkv, sequence.as_mut_ptr(), prefix_length as usize * qkv_width);
            for offset in 0..ancestor_count {
                let ancestor = *ancestor_indices.add(row * ancestor_stride as usize + offset) as usize;
                std::ptr::copy_nonoverlapping(
                    node_qkv.add(ancestor * qkv_width + kv_offset),
                    sequence.as_mut_ptr().add((prefix_length as usize + offset) * qkv_width + kv_offset),
                    kv_width,
                );
            }
            std::ptr::copy_nonoverlapping(
                current_row.add(kv_offset),
                sequence.as_mut_ptr().add((length - 1) * qkv_width + kv_offset),
                kv_width,
            );
            attention_single_pass::<bf16, HEAD_DIM>(
                current_row,
                sequence.as_ptr().add(kv_offset),
                sequence.as_ptr().add(VALUE_COMPONENT * num_heads * head_dim),
                output.add(row * num_heads * head_dim),
                1,
                length as u32,
                HEAD_DIM,
                qkv_width as u32,
                HEAD_DIM,
                qkv_width as u32,
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
