use half::bf16;
use proc_macros::kernel;

use super::attention_single_pass::attention_single_pass;
use crate::backends::common::gpu_types::weaver::MetadataIdx;

#[kernel(AncestorAttention)]
#[variants(HEAD_DIM, 128)]
fn ancestor_attention<const HEAD_DIM: u32>(
    prefix_kv: *const bf16,
    node_kv: *mut bf16,
    current_qkv: *const bf16,
    cosines: *const f32,
    sines: *const f32,
    node_metadata: *const u32,
    ancestor_indices: *const u32,
    ancestor_counts: *const u32,
    node_indices: *const u32,
    output: *mut bf16,
    rows: u32,
    prefix_length: u32,
    ancestor_stride: u32,
    node_capacity: u32,
    max_depth: u32,
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
    let half_dim = head_dim / 2;
    let row_count = rows as usize;

    for row in 0..row_count {
        unsafe {
            let current_row = current_qkv.add(row * qkv_width);

            let depth = *node_metadata.add(MetadataIdx::Depth as usize * row_count + row);
            assert!(depth < max_depth, "node metadata depth must be rope-safe");
            let position = depth as usize + 1;
            let rotate = |component: usize| {
                let mut rotated = vec![bf16::ZERO; model_dim];
                for head in 0..num_heads {
                    let base = component * model_dim + head * head_dim;
                    for pair in 0..half_dim {
                        let low = (*current_row.add(base + pair)).to_f32();
                        let high = (*current_row.add(base + half_dim + pair)).to_f32();
                        let index = position * head_dim + pair;
                        let low_cosine = *cosines.add(index);
                        let low_sine = *sines.add(index);
                        let high_cosine = *cosines.add(index + half_dim);
                        let high_sine = *sines.add(index + half_dim);
                        rotated[head * head_dim + pair] = bf16::from_f32(low * low_cosine - high * low_sine);
                        rotated[head * head_dim + half_dim + pair] =
                            bf16::from_f32(high * high_cosine + low * high_sine);
                    }
                }
                rotated
            };
            let rotated_queries = rotate(0);
            let rotated_keys = rotate(KEY_COMPONENT);

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
                let ancestor = *ancestor_indices.add(row * ancestor_stride as usize + offset) as usize;
                assert!(ancestor < node_capacity, "ancestor slot exceeds tree slot count");
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
                rotated_keys.as_ptr(),
                keys.as_mut_ptr().add((length - 1) * model_dim),
                model_dim,
            );
            std::ptr::copy_nonoverlapping(
                current_row.add(VALUE_COMPONENT * model_dim),
                values.as_mut_ptr().add((length - 1) * model_dim),
                model_dim,
            );
            attention_single_pass::<bf16, HEAD_DIM>(
                rotated_queries.as_ptr(),
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
            if node_capacity > 0 {
                let node = *node_indices.add(row) as usize;
                assert!(node < node_capacity, "node slot exceeds tree slot count");
                std::ptr::copy_nonoverlapping(rotated_keys.as_ptr(), node_kv.add(node * model_dim), model_dim);
                std::ptr::copy_nonoverlapping(
                    current_row.add(VALUE_COMPONENT * model_dim),
                    node_kv.add(node_capacity * model_dim + node * model_dim),
                    model_dim,
                );
            }
        }
    }
}
