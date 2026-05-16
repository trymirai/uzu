use proc_macros::uzu_config;

#[derive(Eq)]
#[uzu_config]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    LanguageModel,
    ClassifierModel,
    TtsModel,
}
