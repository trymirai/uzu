use num_traits::Float;

use crate::data_type::DataTypeRepr;

pub(super) fn apply_rope<ElementT: DataTypeRepr + Float>(
    head: *const ElementT,
    inverse_frequencies: *const f32,
    scale: f32,
    position: u32,
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
    let angle = position as f32 * unsafe { *inverse_frequencies.add(head_dim_idx % half_rope_dim) };
    let (sin_val, cos_val) = angle.sin_cos();

    ElementT::from(input * (cos_val * scale) + signed_paired * (sin_val * scale)).unwrap()
}
