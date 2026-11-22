use bytemuck::NoUninit;

/// The position-dependent scale `1 + alpha * ln(max((position + 1) / n_floor, 1))`.
#[derive(Clone, Copy, Debug, Default, NoUninit)]
#[repr(C)]
pub struct LogScalingParams {
    pub n_floor: f32,
    pub alpha: f32,
    /// The position of the first suffix token; a trie node adds its height.
    pub position_offset: u32,
}
