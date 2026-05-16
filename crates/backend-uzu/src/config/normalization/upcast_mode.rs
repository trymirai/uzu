use proc_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum UpcastMode {
    OnlyNormalization,
    FullLayer,
}
