use half::bf16;
use proc_macros::kernel;

use super::attention_single_pass::attention_single_pass;

// Packed BF16 MHA; destinations must be unique and absent from this dispatch's ancestors.
#[kernel(AttentionLastQuery)]
#[variants(HEAD_DIM, 128)]
fn attention_last_query<const HEAD_DIM: u32>(
    prefix_qkv: *const bf16,
    node_qkv: *mut bf16,
    current_qkv: *const bf16,
    ancestor_indices: *const u32,
    ancestor_counts: *const u32,
    node_indices: *const u32,
    output: *mut bf16,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    scale: f32,
    #[specialize] num_heads: u32,
) {
    const QKV_COMPONENTS: usize = 3;
    const KEY_COMPONENT: usize = 1;
    let head_dim = HEAD_DIM as usize;
    let num_heads = num_heads as usize;
    let packed_width = QKV_COMPONENTS * num_heads * head_dim;
    let kv_offset = KEY_COMPONENT * num_heads * head_dim;
    let kv_width = 2 * num_heads * head_dim;

    for row in 0..rows as usize {
        unsafe {
            let current = current_qkv.add(row * packed_width);
            let node = node_qkv.add(*node_indices.add(row) as usize * packed_width);

            let ancestor_count = *ancestor_counts.add(row) as usize;
            let length = prefix_length as usize + ancestor_count + 1;
            let mut sequence = vec![bf16::ZERO; length * packed_width];
            std::ptr::copy_nonoverlapping(prefix_qkv, sequence.as_mut_ptr(), prefix_length as usize * packed_width);
            for offset in 0..ancestor_count {
                let ancestor = *ancestor_indices.add(row * ancestor_stride as usize + offset) as usize;
                std::ptr::copy_nonoverlapping(
                    node_qkv.add(ancestor * packed_width + kv_offset),
                    sequence.as_mut_ptr().add((prefix_length as usize + offset) * packed_width + kv_offset),
                    kv_width,
                );
            }
            std::ptr::copy_nonoverlapping(
                current.add(kv_offset),
                sequence.as_mut_ptr().add((length - 1) * packed_width + kv_offset),
                kv_width,
            );
            attention_single_pass::<bf16, HEAD_DIM>(
                current,
                sequence.as_ptr().add(kv_offset),
                sequence.as_ptr().add(2 * num_heads * head_dim),
                output.add(row * num_heads * head_dim),
                1,
                length as u32,
                HEAD_DIM,
                packed_width as u32,
                HEAD_DIM,
                packed_width as u32,
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
            std::ptr::copy_nonoverlapping(current.add(kv_offset), node.add(kv_offset), kv_width);
        }
    }
}
