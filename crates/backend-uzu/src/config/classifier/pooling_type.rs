use proc_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "lowercase")]
pub enum PoolingType {
    Cls,
    Mean,
}
