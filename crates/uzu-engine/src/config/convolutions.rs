use uzu_engine_macros::uzu_config;

#[uzu_config]
pub struct SeparableCausalConvConfig {
    pub kernel_size: u32,
    pub coefficient_group_size: Option<u32>,
    pub has_biases: bool,
}
