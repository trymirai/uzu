use super::tile_configuration::TileConfiguration;
use crate::backends::metal::kernel::mlp::MlpActivationType;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub tile: TileConfiguration,
    pub mn_aligned: bool,
    pub k_aligned: bool,
    pub activation: MlpActivationType,
}
