use num_traits::Float;

use crate::data_type::DataTypeRepr;

pub(super) fn apply_rope<ElementT: DataTypeRepr + Float>(
    head: *const ElementT,
    cosines: *const (impl DataTypeRepr + Float),
    sines: *const (impl DataTypeRepr + Float),
    batch_idx: usize,
    head_dim_idx: usize,
    rope_dim: usize,
) -> ElementT {
    let half_rope_dim = rope_dim / 2;
    let paired_idx = if head_dim_idx < half_rope_dim {
        head_dim_idx + half_rope_dim
    } else {
        head_dim_idx - half_rope_dim
    };
    let input = unsafe { (*head.add(head_dim_idx)).to_f32().unwrap() };
    let paired = unsafe { (*head.add(paired_idx)).to_f32().unwrap() };
    let signed_paired = if head_dim_idx < half_rope_dim {
        -paired
    } else {
        paired
    };
    let cos_val = unsafe { (*cosines.add(batch_idx * rope_dim + head_dim_idx)).to_f32().unwrap() };
    let sin_val = unsafe { (*sines.add(batch_idx * rope_dim + head_dim_idx)).to_f32().unwrap() };

    ElementT::from(input * cos_val + signed_paired * sin_val).unwrap()
}
