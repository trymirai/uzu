use uzu_engine_macros::uzu_config;

use crate::config::weight_matrix::{Layout, qtip_gaussian_spec::QtipGaussianSpec};

/// Row-wise concatenation of `(rows, spec)` parts, stored under `parts.<index>`.
#[uzu_config(super::WeightMatrixSpec)]
pub struct RowStackSpec {
    pub parts: Box<[(u32, QtipGaussianSpec)]>,
    pub layout: Layout,
}
