use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct I3S4Spec {
    pub layout: Layout,
}
