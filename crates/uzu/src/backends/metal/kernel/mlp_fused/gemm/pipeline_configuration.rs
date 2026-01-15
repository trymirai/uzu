use super::tile_configuration::TileConfiguration;
use crate::backends::metal::kernel::mlp::MlpActivationType;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub tile: TileConfiguration,
    pub weights_transposed: bool,
    pub mn_aligned: bool,
    pub k_aligned: bool,
    pub activation: MlpActivationType,
}
