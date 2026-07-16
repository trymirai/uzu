use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EncodingConfig {
    Hanashi(HanashiConfig),
    Harmony(HarmonyConfig),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum HanashiConfig {
    #[serde(rename = "gpt-oss")]
    GptOss,
    #[serde(rename = "llama-3.1")]
    Llama31,
    #[serde(rename = "llama-3.2")]
    Llama32,
    #[serde(rename = "qwen2.5")]
    Qwen25,
    #[serde(rename = "qwen2.5-coder")]
    Qwen25Coder,
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
    #[serde(rename = "lfm2")]
    Lfm2,
    #[serde(rename = "lfm2.5-instruct")]
    Lfm25Instruct,
    #[serde(rename = "lfm2.5-thinking")]
    Lfm25Thinking,
    #[serde(rename = "deepseek-r1-distill-qwen")]
    DeepseekR1DistillQwen,
    #[serde(rename = "llamba")]
    Llamba,
    #[serde(rename = "rnj-1")]
    Rnj1,
    #[serde(rename = "gemma-2")]
    Gemma2,
    #[serde(rename = "gemma-3")]
    Gemma3,
    #[serde(rename = "gemma-4")]
    Gemma4,
    #[serde(rename = "functiongemma")]
    FunctionGemma,
    #[serde(rename = "translategemma")]
    TranslateGemma,
    #[serde(rename = "smollm2")]
    SmolLm2,
    #[serde(rename = "smollm3")]
    SmolLm3,
    #[serde(rename = "codestral")]
    Codestral,
    #[serde(rename = "mistral-small-3")]
    MistralSmall3,
    #[serde(rename = "devstral-small")]
    DevstralSmall,
    #[serde(rename = "polaris")]
    Polaris,
    #[serde(rename = "bonsai")]
    Bonsai,
    #[serde(rename = "nanbeige")]
    Nanbeige,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum HarmonyConfig {
    #[serde(rename = "gpt-oss")]
    GptOss,
}
