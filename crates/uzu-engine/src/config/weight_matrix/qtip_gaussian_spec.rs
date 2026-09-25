use uzu_engine_macros::uzu_config;

use crate::{config::weight_matrix::Layout, data_type::DataType};

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum PostGainAxis {
    Row,
}

/// Mirai S trellis leaf: codes decode through a 16-bit state trellis into a shared Gaussian codebook,
/// scaled per row (`scales`, `gains`, then one `post_gains.<index>` tensor per axis).
#[uzu_config(super::WeightMatrixSpec)]
pub struct QtipGaussianSpec {
    pub layout: Layout,
    pub vector_width: u32,
    pub transition_bits: u32,
    pub restart_columns: u32,
    pub scale_dtype: DataType,
    pub post_gain_axes: Box<[PostGainAxis]>,
}
