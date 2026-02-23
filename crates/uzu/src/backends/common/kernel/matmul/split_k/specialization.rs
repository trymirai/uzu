use super::tile_configuration::TileConfiguration;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Specialization {
    pub tile: TileConfiguration,
    pub transpose_b: bool,
    pub mn_aligned: bool,
    pub k_aligned: bool,
}
