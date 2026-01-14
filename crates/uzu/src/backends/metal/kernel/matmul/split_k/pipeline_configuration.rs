use super::tile_configuration::TileConfiguration;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub tile: TileConfiguration,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub mn_aligned: bool,
    pub k_aligned: bool,
}
