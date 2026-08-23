use backend_uzu_macros::uzu_config;

#[uzu_config]
pub struct SeparableCausalConvConfig {
    pub has_biases: bool,
}
