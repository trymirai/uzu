use super::super::grid_size::GridSize;
use crate::backends::common::gpu_types::GEMMParams;

#[derive(Debug, Clone)]
pub struct DispatchDescriptor {
    pub params: GEMMParams,
    pub threadgroups: GridSize,
}
