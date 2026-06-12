use monostate::MustBe;
use proc_macros::uzu_config;

#[uzu_config]
pub struct CausalConv1dConfig {
    pub has_biases: MustBe!(true),
}

#[uzu_config]
pub struct CausalTransposeConv1dConfig {
    pub has_biases: MustBe!(true),
}

#[uzu_config]
pub struct Snake1dConfig {}
