use super::{super::grid_size::GridSize, specialization::Specialization};
use crate::backends::common::gpu_types::GEMMParams;

#[derive(Debug, Clone)]
pub struct DispatchDescriptor {
    pub specialization: Specialization,
    pub params: GEMMParams,
    pub threadgroups: GridSize,
}
