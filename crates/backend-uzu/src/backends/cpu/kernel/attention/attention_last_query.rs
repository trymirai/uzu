use half::bf16;
use proc_macros::kernel;

#[kernel(AttentionLastQuery)]
pub fn attention_last_query(
    qkv: *const bf16,
    lengths: *const u32,
    output: *mut bf16,
    rows: u32,
    sequence_length: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
) {
    let rows = rows as usize;
    let sequence_length = sequence_length as usize;
    let num_heads = num_heads as usize;
    let head_dim = head_dim as usize;
    let qkv_width = 3 * num_heads * head_dim;
    unsafe {
        for row in 0..rows {
            let length = (*lengths.add(row) as usize).min(sequence_length);
            let query_row = qkv.add((row * sequence_length + length - 1) * qkv_width);
            for head in 0..num_heads {
                let query = query_row.add(head * head_dim);
                let mut max_score = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                let mut values = vec![0.0f32; head_dim];
                for position in 0..length {
                    let row_qkv = qkv.add((row * sequence_length + position) * qkv_width);
                    let key = row_qkv.add((num_heads + head) * head_dim);
                    let value = row_qkv.add((2 * num_heads + head) * head_dim);
                    let mut score = 0.0f32;
                    for dim in 0..head_dim {
                        score += (*query.add(dim)).to_f32() * (*key.add(dim)).to_f32();
                    }
                    score *= scale;
                    let new_max = max_score.max(score);
                    let old_factor = (max_score - new_max).exp();
                    let factor = (score - new_max).exp();
                    sum = sum * old_factor + factor;
                    for dim in 0..head_dim {
                        values[dim] = values[dim] * old_factor + factor * (*value.add(dim)).to_f32();
                    }
                    max_score = new_max;
                }
                for dim in 0..head_dim {
                    *output.add((row * num_heads + head) * head_dim + dim) = bf16::from_f32(values[dim] / sum);
                }
            }
        }
    }
}
