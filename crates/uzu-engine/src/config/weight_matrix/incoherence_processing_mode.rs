use uzu_engine_macros::uzu_config;

#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum IncoherenceProcessingMode {
    Input,
    Output,
    InputOutput,
}
