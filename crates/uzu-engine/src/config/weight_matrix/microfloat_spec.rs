use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::Layout;

/// How the per-group scales of a microfloat matrix are encoded.
#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatScaleMode {
    /// One E8M0 power-of-two scale per group (stored as `U8`), no global scale.
    Mxfp4,
    /// One E4M3 scale per group (stored as `F8_E4M3`) times one float32 global scale per matrix.
    Nvfp4,
}

/// Lalamo's `MicrofloatSpec`: 4-bit E2M1 codes packed two per byte, low nibble first, with group scales along the
/// input dimension. A weight is `e2m1(code) * scale[group] * global_scale`.
///
/// Tensors under the matrix prefix: `weights` (`U8`, `[*components, rows, cols / 2]`), `scales`
/// (`[*components, rows, cols / group_size]`) and `global_scale` (`F32`, `[*components]`).
#[uzu_config(super::WeightMatrixSpec)]
pub struct MicrofloatSpec {
    pub bits: u32,
    pub group_size: u32,
    pub scale_mode: MicrofloatScaleMode,
    pub layout: Layout,
}
