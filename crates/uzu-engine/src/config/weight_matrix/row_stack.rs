use uzu_engine_macros::uzu_config;

use super::AnyWeightMatrixSpec;

#[uzu_config(super::WeightMatrixSpec)]
pub struct RowStackSpec {
    pub parts: Box<[(u32, AnyWeightMatrixSpec)]>,
}
