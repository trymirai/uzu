use std::collections::HashMap;

use serde::Deserialize;
use tokenizers::AddedToken;

use crate::session::config::common::ValueOrToken;

#[derive(Clone, Deserialize, Debug)]
pub struct TokenizerConfig {
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub add_prefix_space: Option<bool>,
    pub added_tokens_decoder: Option<HashMap<u32, AddedToken>>,
    pub additional_special_tokens: Option<Vec<String>>,
    pub boi_token: Option<ValueOrToken>,
    pub bos_token: Option<ValueOrToken>,
    pub chat_template: Option<String>,
    pub clean_up_tokenization_spaces: Option<bool>,
    pub eoi_token: Option<ValueOrToken>,
    pub eos_token: Option<ValueOrToken>,
    pub errors: Option<String>,
    pub extra_special_tokens: Option<HashMap<String, String>>,
    pub image_token: Option<ValueOrToken>,
    pub pad_token: Option<ValueOrToken>,
    pub processor_class: Option<String>,
    pub spaces_between_special_tokens: Option<bool>,
    pub split_special_tokens: Option<bool>,
    pub tokenizer_class: Option<String>,
    pub unk_token: Option<ValueOrToken>,
    pub use_default_system_prompt: Option<bool>,
}
