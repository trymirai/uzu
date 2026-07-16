use serde::{Deserialize, Serialize};

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EncodingConfig {
    Hanashi {
        #[serde(flatten)]
        config: HanashiConfig,
    },
    Harmony {
        #[serde(flatten)]
        config: HarmonyConfig,
    },
}

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum HanashiConfig {
    #[serde(rename = "functiongemma")]
    FunctionGemma,
    #[serde(rename = "gemma-3")]
    Gemma3,
    #[serde(rename = "gemma-4")]
    Gemma4,
    #[serde(rename = "gpt-oss")]
    GptOss,
    #[serde(rename = "lfm2")]
    Lfm2,
    #[serde(rename = "lfm2.5-instruct")]
    Lfm25Instruct,
    #[serde(rename = "lfm2.5-thinking")]
    Lfm25Thinking,
    #[serde(rename = "llama-3.2")]
    Llama32,
    #[serde(rename = "qwen3")]
    Qwen3,
    #[serde(rename = "qwen3-instruct")]
    Qwen3Instruct,
    #[serde(rename = "qwen3-thinking")]
    Qwen3Thinking,
    #[serde(rename = "qwen3.5")]
    Qwen35,
    #[serde(rename = "qwen3.6")]
    Qwen36,
}

#[bindings::export(Enumeration)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum HarmonyConfig {
    #[serde(rename = "gpt-oss")]
    GptOss,
}
