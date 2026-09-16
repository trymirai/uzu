use uzu_engine_macros::uzu_config;

#[derive(Copy, Eq)]
#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum MicrofloatFormat {
    Mxfp4,
}
