use super::patch_coordinates;
use crate::{
    backends::common::{Backend, CommandBuffer, CommandBufferEncoding},
    config::vision::VisionRoPEConfig,
    encodable_block::mixer::attention::rope::PrecalculatedRoPE,
};

/// 2D RoPE tables for one image of `height x width` patches, in HF's patch order:
/// each `spatial_merge_size x spatial_merge_size` block of patches is contiguous.
///
/// Each patch gets `head_dim` angles `[row * f, col * f, row * f, col * f]`, with `f_i = base^(-2i / (head_dim / 2))`;
/// the attention applies them rotate-half style, pairing channel `c` with `c +- head_dim / 2`.
pub fn get_precalculated_vision_rope<B: Backend>(
    config: &VisionRoPEConfig,
    height: u32,
    width: u32,
    spatial_merge_size: u32,
    command_buffer: &mut <B::CommandBuffer as CommandBuffer>::Encoding,
) -> Result<PrecalculatedRoPE<B>, B::Error> {
    let axis_dim = config.head_dim / 2;
    let inverse_frequencies: Vec<f32> =
        (0..axis_dim).step_by(2).map(|i| 1.0 / config.base.powf(i as f32 / axis_dim as f32)).collect();

    let elements = (height * width * config.head_dim) as usize;
    let mut cosines = Vec::with_capacity(elements);
    let mut sines = Vec::with_capacity(elements);
    for (row, col) in patch_coordinates(height, width, spatial_merge_size) {
        let angles = || [row, col].into_iter().flat_map(|p| inverse_frequencies.iter().map(move |f| p as f32 * f));
        for angle in angles().chain(angles()) {
            cosines.push(angle.cos());
            sines.push(angle.sin());
        }
    }

    Ok(PrecalculatedRoPE {
        cosines: command_buffer.allocate_constant_from_slice(&cosines)?,
        sines: command_buffer.allocate_constant_from_slice(&sines)?,
        dim: config.head_dim,
    })
}
