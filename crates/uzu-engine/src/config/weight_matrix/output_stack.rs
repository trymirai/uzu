use uzu_engine_macros::uzu_config;

use super::AnyWeightMatrixSpec;

#[uzu_config(super::WeightMatrixSpec)]
pub struct OutputStackSpec {
    pub parts: Box<[OutputStackPart]>,
}

#[uzu_config]
pub struct OutputStackPart {
    pub rows: u32,
    pub spec: AnyWeightMatrixSpec,
}
