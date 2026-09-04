use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct D4S4Spec {
    pub layout: Layout,
}
