use super::tile_configuration::TileConfiguration;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineConfiguration {
    pub tile: TileConfiguration,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub align_m: bool,
    pub align_n: bool,
    pub align_k: bool,
    pub has_batch: bool,
    pub use_out_source: bool,
    pub do_axpby: bool,
}
