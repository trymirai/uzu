use num_traits::Float;

use crate::data_type::DataTypeRepr;

/// Interleaved M-RoPE: pairs 1, 4, 7, ... add the grid row, pairs 2, 5, 8, ... the grid column, while the axis has
/// pairs left. The rest stay at the grid's position.
pub(super) fn mrope_axis_offset(
    pair: u32,
    grid_row: u32,
    grid_column: u32,
    height_pairs: u32,
    width_pairs: u32,
) -> u32 {
    if pair % 3 == 1 && pair < 3 * height_pairs {
        grid_row
    } else if pair % 3 == 2 && pair < 3 * width_pairs {
        grid_column
    } else {
        0
    }
}

pub(super) fn apply_rope<ResultT: DataTypeRepr + Float, ElementT: DataTypeRepr + Float>(
    head: *const ElementT,
    inverse_frequencies: *const f32,
    scale: f32,
    position: u32,
    head_dim_idx: usize,
    rope_dim: usize,
) -> ResultT {
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

    ResultT::from(input * (cos_val * scale) + signed_paired * (sin_val * scale)).unwrap()
}
