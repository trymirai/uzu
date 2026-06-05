use proc_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum UpcastMode {
    OnlyNormalization,
    FullLayer,
}

#[uzu_config]
pub struct NormalizationConfig {
    pub epsilon: f32,
    pub scale_offset: Option<f32>,
    pub upcast_mode: UpcastMode,
    pub subtract_mean: bool,
    pub has_biases: bool,
}
