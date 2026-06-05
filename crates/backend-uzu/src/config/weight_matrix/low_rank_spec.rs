use proc_macros::uzu_config;

#[uzu_config(super::WeightMatrixSpec)]
pub struct LowRankSpec {
    pub rank: usize,
}
